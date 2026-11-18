"""Generation-2 reinforcement learning: Gumbel self-play from the pre-trained
clock-aware network, with two gates.

    self-play workers  Gumbel search + playout-cap randomisation with the
                       current *best* network (src/selfplay2.py)
    trainer (here)     trains the *candidate* on a mix of
                         * self-play positions (improved policy, result,
                           moves-left; clocks off), and
                         * strong human bullet positions (human move, result
                           including losses on time, moves-left; clocks on),
                       so the network gets stronger at chess without
                       forgetting what the clocks mean
    gate 1 ("match")   candidate vs best, fixed simulations, many games at
                       once; >= --promote-threshold makes it the new best
                       (what self-play uses)
    gate 2 ("bullet")  at most every --bullet-every-hours, the newest best vs
                       the bot's network in real clocked bullet games (self-
                       play and training hold meanwhile); >= --bullet-threshold
                       copies it to checkpoints/v2/bot.pt, which the Lichess bot
                       plays with

    python -m src.loop2                    # until logs/STOP

Resumable: checkpoints/v2/rl/state.json (counters), latest.pt (candidate,
optimizer, human-data position), gen/best_NNNNN.pt (every best), history.jsonl
(gate results), games2/ (self-play replay buffer). Version 0 is the
pre-trained network, which is gated against the old bot by hand before the
first run (`src.bullet_arena --vs`), so the first bullet gate is for v1.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import multiprocessing as mp
import os
import queue
import shutil
import signal
import time
from collections import Counter
from pathlib import Path

faulthandler.enable()

import numpy as np
import torch
import torch.nn.functional as F

from . import encoding
from ._log import setup_logging
from .evaluate import elo_gap
from .loop import _atomic_save, _copy_retry, _keep_awake, State
from .network2 import ChessNetV2, planes_u8_to_input
from .pretrain import ShardStream, find_shards, losses as human_losses
from .replay2 import ReplayBuffer2
from .selfplay2 import to_buffer_rec
from .workers import training_paused
from .workers2 import arena2_worker, best_path2, selfplay2_worker

log = logging.getLogger("loop2")



def selfplay_losses(net, b: dict, device):
    """Losses on a self-play batch (clocks unknown)."""
    n = len(b["result"])
    planes = torch.from_numpy(b["states"]).to(device, non_blocking=True)
    z = torch.zeros(n, device=device)
    x = planes_u8_to_input(planes, z, z, z, torch.zeros(n, dtype=torch.bool, device=device))
    idx = torch.from_numpy(b["pidx"].astype(np.int64)).to(device)
    prob = torch.from_numpy(b["pprob"].astype(np.float32)).to(device)
    target = torch.zeros(n, encoding.N_MOVES, device=device).scatter_add_(1, idx, prob)
    wdl_t = torch.from_numpy(1 - b["result"].astype(np.int64)).to(device)     # +1→0, 0→1, -1→2
    plies = torch.from_numpy(b["plies"].astype(np.float32)).to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        p, wdl, ml = net(x.to(memory_format=torch.channels_last))
    lp = -(target * F.log_softmax(p.float(), dim=1)).sum(1).mean()
    lv = F.cross_entropy(wdl.float(), wdl_t)
    known = plies >= 0
    lm = (F.smooth_l1_loss(ml.float()[known], plies[known] / 100.0) if known.any()
          else ml.float().sum() * 0.0)
    return lp, lv, lm


def _history(ckpt_dir: Path, event: dict) -> None:
    event = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **event}
    with open(ckpt_dir / "history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="checkpoints/v2/pretrain_best.pt", help="network for best v0")
    ap.add_argument("--ckpt-dir", default="checkpoints/v2/rl")
    ap.add_argument("--buffer", default="games2")
    ap.add_argument("--bot-path", default="checkpoints/v2/bot.pt",
                    help="the bot's network: gate 2 compares against it and replaces it")
    ap.add_argument("--max-steps", type=int, default=0, help="stop for good at this step (0 = never)")
    # self-play
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--parallel", type=int, default=128, help="concurrent games per worker")
    ap.add_argument("--n-full", type=int, default=128)
    ap.add_argument("--n-fast", type=int, default=24)
    ap.add_argument("--p-full", type=float, default=0.25)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--ml-coef", type=float, default=0.1, help="moves-left utility in search")
    # training
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--human-frac", type=float, default=0.25, help="share of each batch from human games")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--sample-ratio", type=float, default=4.0,
                    help="times each self-play position is sampled on average")
    ap.add_argument("--min-buffer", type=int, default=100_000)
    ap.add_argument("--buffer-capacity", type=int, default=2_000_000)
    # gate 1
    ap.add_argument("--eval-every", type=int, default=3000, help="train steps between candidates")
    ap.add_argument("--eval-games", type=int, default=160)
    ap.add_argument("--eval-sims", type=int, default=64)
    ap.add_argument("--arena-max-plies", type=int, default=300)
    ap.add_argument("--promote-threshold", type=float, default=0.55)
    # gate 2
    ap.add_argument("--bullet-every-hours", type=float, default=8.0)
    ap.add_argument("--bullet-games", type=int, default=40)
    ap.add_argument("--bullet-tc", default="1+0")
    ap.add_argument("--bullet-lag-ms", type=int, default=60)
    ap.add_argument("--bullet-threshold", type=float, default=0.55)
    ap.add_argument("--stats-every", type=float, default=120)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging("loop2", args.log_level)
    log.info(f"args = {vars(args)}")
    _keep_awake()
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ckpt_dir = Path(args.ckpt_dir)
    bot_path = Path(args.bot_path)
    (ckpt_dir / "gen").mkdir(parents=True, exist_ok=True)
    stop_file = Path("logs") / "STOP"
    State.DEFAULTS = {**State.DEFAULTS, "bot_checked": 0, "last_bullet": 0.0, "bot": "initial"}
    state = State(ckpt_dir / "state.json")

    init = torch.load(args.init, map_location="cpu")
    cfg = init.get("config", {"channels": 192, "blocks": 12})
    net = ChessNetV2(cfg["channels"], cfg["blocks"]).to(device).to(memory_format=torch.channels_last)
    log.info(f"ChessNetV2 {cfg['channels']}x{cfg['blocks']}")
    decay = [p for p in net.parameters() if p.ndim > 1]
    no_decay = [p for p in net.parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)

    train_shards, _ = find_shards()
    human_bs = int(round(args.batch_size * args.human_frac))
    sp_bs = args.batch_size - human_bs
    human = ShardStream(train_shards, human_bs, seed=7) if human_bs and train_shards else None
    if human_bs and human is None:
        log.warning("no human shards found; training on self-play only")
        sp_bs, human_bs = args.batch_size, 0

    latest = ckpt_dir / "latest.pt"
    best0 = best_path2(ckpt_dir, 0)
    if latest.exists():
        ck = torch.load(latest, map_location=device)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if human is not None and ck.get("human"):
            human.load_state(ck["human"])
        log.info(f"resumed candidate from {latest} (step {state['step']})")
    else:
        net.load_state_dict(init["model"] if "model" in init else init)
        log.info(f"initialised candidate from {args.init}")
    if not best0.exists():
        state["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _atomic_save({"model": net.state_dict(), "config": cfg}, best0)
        state.save()
        log.info(f"fresh run: best v0 = {args.init}")
    bot_path.parent.mkdir(parents=True, exist_ok=True)
    if not bot_path.exists():
        _copy_retry(Path("checkpoints") / "best.pt", bot_path)
        log.warning(f"{bot_path} missing; using the generation-1 best as the bot's network")
    del init
    log.info(f"state: {state.d}")

    buffer = ReplayBuffer2(args.buffer, capacity=args.buffer_capacity)
    rng = np.random.default_rng()

    # -- workers -----------------------------------------------------------
    ctx = mp.get_context("spawn")
    version = ctx.Value("i", state["version"])
    stop_evt, hold = ctx.Event(), ctx.Event()
    games_q, task_q, result_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    wcfg = {"device": args.device, "channels": cfg["channels"], "blocks": cfg["blocks"],
            "ckpt_dir": str(ckpt_dir), "parallel": args.parallel, "ml_coef": args.ml_coef,
            "eval_sims": args.eval_sims, "arena_max_plies": args.arena_max_plies,
            "log_level": args.log_level,
            "gumbel": {"n_full": args.n_full, "n_fast": args.n_fast, "p_full": args.p_full,
                       "max_plies": args.max_plies}}

    def spawn_sp(i):
        p = ctx.Process(target=selfplay2_worker, args=(i, wcfg, version, hold, games_q, stop_evt),
                        name=f"selfplay2-{i}", daemon=True)
        p.start()
        return p

    def spawn_arena():
        p = ctx.Process(target=arena2_worker, args=(wcfg, task_q, result_q, stop_evt),
                        name="arena2", daemon=True)
        p.start()
        return p

    sp_procs = [spawn_sp(i) for i in range(args.workers)]
    arena = spawn_arena()
    log.info(f"started {args.workers} self-play workers x {args.parallel} games + 1 arena worker")

    pending: list[dict] = []
    t_flush = t_stats = time.monotonic()
    t_pause_check, paused = 0.0, False
    busy = None                                     # kind of the running gate task
    window = {"games": 0, "plies": 0, "evals": 0, "steps": 0, "rec": 0,
              "lp": 0.0, "lv": 0.0, "lm": 0.0, "hp": 0.0, "hacc": 0.0, "outcomes": Counter()}
    evals_seen: dict[int, int] = {}

    def save_trainer():
        _atomic_save({"model": net.state_dict(), "optimizer": opt.state_dict(), "config": cfg,
                      "step": state["step"], "human": human.state() if human else None}, latest)
        state.save()

    def flush():
        nonlocal pending, t_flush
        recs = [to_buffer_rec(r) for r in pending]
        state["positions"] += buffer.save_games([r for r in recs if r])
        state["games"] += len(pending)
        pending, t_flush = [], time.monotonic()

    def handle_result(res):
        nonlocal busy
        busy = None
        if "error" in res:
            log.error(f"gate task failed ({res['kind']}); see worker2-arena log")
            if res["kind"] == "match":
                Path(res["a"]).unlink(missing_ok=True)
            else:
                hold.clear()
                state["last_bullet"] = time.time()          # don't retry immediately
            return
        w, d, l = res["wdl"]
        score = (w + 0.5 * d) / max(1, w + d + l)
        if res["kind"] == "match":
            v = state["version"]
            promoted = score >= args.promote_threshold
            log.info(f"[gate] candidate@{res['step']} vs best v{v}: {w}W-{d}D-{l}L score={score:.3f} "
                     f"Elo ≈ {elo_gap(score):+.0f} ({res['seconds']:.0f}s) "
                     f"{'→ PROMOTED v' + str(v + 1) if promoted else '→ rejected'}")
            _history(ckpt_dir, {"kind": "match", "step": res["step"], "best": v,
                                "wdl": [w, d, l], "score": score, "promoted": promoted})
            if promoted:
                os.replace(res["a"], best_path2(ckpt_dir, v + 1))
                state["version"] = v + 1
                state["promotions"] += 1
                version.value = v + 1
            else:
                Path(res["a"]).unlink(missing_ok=True)
        else:
            hold.clear()
            ok = score >= args.bullet_threshold
            log.info(f"[bullet] best v{res['version']} vs bot ({state['bot']}) at {args.bullet_tc}: "
                     f"{w}W-{d}D-{l}L score={score:.3f} ({res['seconds'] / 60:.0f} min) "
                     f"{'→ bot now plays v' + str(res['version']) if ok else '→ bot unchanged'}")
            _history(ckpt_dir, {"kind": "bullet", "best": res["version"], "bot": state["bot"],
                                "wdl": [w, d, l], "score": score, "promoted": ok})
            state["bot_checked"] = res["version"]
            state["last_bullet"] = time.time()
            if ok:
                _copy_retry(best_path2(ckpt_dir, res["version"]), bot_path)
                state["bot"] = f"rl v{res['version']}"
        state.save()

    reason = "?"
    try:
        while True:
            now = time.monotonic()
            if stop_file.exists():
                reason = "STOP file"
                break
            if args.max_steps and state["step"] >= args.max_steps and busy is None:
                reason = f"--max-steps {args.max_steps} reached (logs/STOP left in place)"
                stop_file.touch()
                break

            for i, p in enumerate(sp_procs):
                if not p.is_alive():
                    log.error(f"self-play worker {i} died (exit {p.exitcode}); restarting")
                    sp_procs[i] = spawn_sp(i)
                    evals_seen.pop(i, None)
            if not arena.is_alive():
                log.error(f"arena worker died (exit {arena.exitcode}); restarting")
                arena = spawn_arena()
                if busy == "bullet":
                    hold.clear()
                busy = None

            while True:
                try:
                    rec = games_q.get_nowait()
                except queue.Empty:
                    break
                pending.append(rec)
                window["games"] += 1
                window["plies"] += rec["plies"]
                window["rec"] += len(rec.get("pos_result", ()))
                wid = rec.get("worker", 0)
                window["evals"] += max(0, rec.get("evals_total", 0) - evals_seen.get(wid, 0))
                evals_seen[wid] = rec.get("evals_total", 0)
                window["outcomes"][rec["outcome"].split(" (")[0]] += 1
            if len(pending) >= 64 or (pending and now - t_flush > 120):
                flush()

            try:
                handle_result(result_q.get_nowait())
            except queue.Empty:
                pass
            if now - t_pause_check >= 1.0:
                was, paused = paused, training_paused()
                t_pause_check = now
                if paused != was:
                    log.info("paused: bot game in progress (logs/PAUSE)" if paused else "resumed")
            if paused or hold.is_set():
                time.sleep(0.25)
                continue

            # Gate 2 has priority: newest best vs the bot's network.
            if (busy is None and state["version"] > state["bot_checked"]
                    and time.time() - state["last_bullet"] >= args.bullet_every_hours * 3600):
                save_trainer()
                hold.set()
                task_q.put({"kind": "bullet", "a": str(best_path2(ckpt_dir, state["version"])),
                            "b": str(bot_path), "version": state["version"], "tc": args.bullet_tc,
                            "n_games": args.bullet_games, "lag_ms": args.bullet_lag_ms})
                busy = "bullet"
                log.info(f"[bullet] gating best v{state['version']} vs the bot's network; "
                         f"self-play and training hold meanwhile")
                continue
            if (busy is None and buffer.size >= args.min_buffer
                    and state["step"] - state["last_snapshot_step"] >= args.eval_every):
                snap = ckpt_dir / "gen" / f"cand_{state['step']:08d}.pt"
                _atomic_save({"model": net.state_dict(), "config": cfg}, snap)
                state["last_snapshot_step"] = state["step"]
                save_trainer()
                task_q.put({"kind": "match", "a": str(snap),
                            "b": str(best_path2(ckpt_dir, state["version"])),
                            "n_games": args.eval_games, "step": state["step"]})
                busy = "match"

            allowed = int(state["positions"] * args.sample_ratio / sp_bs) - state["step"]
            if buffer.size < args.min_buffer or allowed <= 0:
                time.sleep(0.25)
            else:
                net.train()
                for _ in range(min(allowed, 20)):
                    step = state["step"]
                    lr = args.lr * min(1.0, (step + 1) / args.warmup)
                    for g in opt.param_groups:
                        g["lr"] = lr
                    lp, lv, lm = selfplay_losses(net, buffer.sample(sp_bs, rng), device)
                    loss = (sp_bs / args.batch_size) * (lp + lv + 0.5 * lm)
                    if human is not None:
                        hp, hv, hm, hacc = human_losses(net, human.next(), device, 0.2, rng)
                        loss = loss + (human_bs / args.batch_size) * (hp + hv + 0.5 * hm)
                        window["hp"] += float(hp); window["hacc"] += float(hacc)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step()
                    state["step"] += 1
                    window["steps"] += 1
                    window["lp"] += float(lp); window["lv"] += float(lv); window["lm"] += float(lm)
                    if state["step"] % 2000 == 0:
                        save_trainer()

            if now - t_stats >= args.stats_every:
                dt, s, g = now - t_stats, max(1, window["steps"]), max(1, window["games"])
                oc = window["outcomes"]
                log.info(
                    f"[stats] games {state['games'] + len(pending)} (+{window['games'] / dt * 3600:.0f}/h)  "
                    f"positions {state['positions']} (+{window['rec'] / dt * 3600:.0f}/h)  "
                    f"evals/s {window['evals'] / dt:.0f}  avg plies {window['plies'] / g:.0f}  "
                    f"W/B/D {oc['white wins']}/{oc['black wins']}/"
                    f"{sum(v for k, v in oc.items() if k.startswith('draw'))}  "
                    f"| step {state['step']} (+{window['steps'] / dt:.1f}/s)  "
                    f"self-play p={window['lp'] / s:.3f} v={window['lv'] / s:.3f} m={window['lm'] / s:.4f}  "
                    f"human p={window['hp'] / s:.3f} top-1 {window['hacc'] / s:.1%}  "
                    f"| buffer {buffer.size}  best v{state['version']}  bot {state['bot']}")
                window = {k: (Counter() if k == "outcomes" else 0) for k in window}
                t_stats = now
                state.save()
    except KeyboardInterrupt:
        reason = "keyboard interrupt"
    finally:
        log.info(f"stopping ({reason}); saving state")
        stop_evt.set()
        hold.clear()
        flush()
        save_trainer()
        for p in sp_procs + [arena]:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        log.info(f"loop2 stopped: step {state['step']}, best v{state['version']}, "
                 f"{state['games']} games, bot {state['bot']}")


if __name__ == "__main__":
    main()
