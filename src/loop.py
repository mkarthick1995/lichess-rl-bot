"""Training orchestrator: parallel self-play → continuous training → gated
promotion, all running concurrently (AlphaGo Zero style).

    self-play workers (N processes)   play games with the current *best* net,
                                      stream them to the trainer
    trainer (this process)            ingests games into the replay buffer and
                                      trains the *candidate* net, paced so each
                                      generated position is sampled
                                      ~--sample-ratio times
    arena worker (1 process)          plays candidate snapshots against best;
                                      a snapshot scoring ≥ --promote-threshold
                                      becomes the new best (workers reload it),
                                      and every few promotions best is measured
                                      against a local Stockfish ladder

Run:
    python -m src.loop                      # forever, until logs/STOP exists
    python -m src.loop --hours 12

Fully resumable: checkpoints/state.json holds counters, checkpoints/latest.pt
the candidate + optimizer, checkpoints/gen/best_NNNNN.pt every promoted net
(checkpoints/best.pt is a copy of the newest, for the Lichess tools), and
games/ the replay buffer.
"""
from __future__ import annotations

import argparse
import ctypes
import faulthandler
import json
import logging
import multiprocessing as mp
import os
import queue
import shutil
import sys
import time
from collections import Counter, deque
from pathlib import Path

# Dump a native traceback on hard faults (segfault, CUDA abort) so a silent
# process death leaves evidence in stderr.
faulthandler.enable()

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD

from . import encoding
from ._log import setup_logging
from .evaluate import elo_gap
from .network import ChessNet
from .replay_buffer import ReplayBuffer
from .workers import arena_worker, best_path, selfplay_worker

log = logging.getLogger("loop")


# ---------------------------------------------------------------------------
# small helpers

def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:          # Windows: a reader has it open
            time.sleep(0.5)
    raise PermissionError(f"could not replace {path}")


def _copy_retry(src: Path, dst: Path) -> None:
    for attempt in range(20):
        try:
            shutil.copyfile(src, dst.with_suffix(".tmp"))
            os.replace(dst.with_suffix(".tmp"), dst)
            return
        except PermissionError:
            time.sleep(0.5)
    log.warning(f"could not update {dst} (file locked); will retry next promotion")


def _keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (no system change;
    released automatically when the process exits)."""
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        log.info("sleep inhibited while training runs")


def _lr_at(schedule: list[tuple[int, float]], step: int) -> float:
    lr = schedule[0][1]
    for s, v in schedule:
        if step >= s:
            lr = v
    return lr


def _parse_schedule(text: str) -> list[tuple[int, float]]:
    out = []
    for part in text.split(","):
        s, v = part.split(":")
        out.append((int(s), float(v)))
    return sorted(out)


class State:
    """Persistent counters in checkpoints/state.json."""

    DEFAULTS = {"version": 0, "step": 0, "positions": 0, "games": 0,
                "promotions": 0, "sf_level": 0, "last_snapshot_step": 0,
                "created": None}

    def __init__(self, path: Path):
        self.path = path
        self.d = dict(self.DEFAULTS)
        if path.exists():
            self.d.update(json.loads(path.read_text()))

    def __getitem__(self, k):
        return self.d[k]

    def __setitem__(self, k, v):
        self.d[k] = v

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=2))
        os.replace(tmp, self.path)


def _history(ckpt_dir: Path, event: dict) -> None:
    event = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **event}
    with open(ckpt_dir / "history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


# ---------------------------------------------------------------------------
# training

def train_step(net, opt, batch, device) -> tuple[float, float, float]:
    s, pidx, pprob, z = batch
    x = encoding.planes_to_float(torch.from_numpy(s).to(device, non_blocking=True))
    idx = torch.from_numpy(pidx.astype(np.int64)).to(device, non_blocking=True)
    prob = torch.from_numpy(pprob.astype(np.float32)).to(device, non_blocking=True)
    z = torch.from_numpy(z).to(device, non_blocking=True)
    target = torch.zeros(len(z), encoding.N_MOVES, device=device).scatter_add_(1, idx, prob)

    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, v = net(x)
    policy_loss = -(target * F.log_softmax(logits.float(), dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(v.float(), z)
    loss = policy_loss + value_loss

    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    opt.step()
    return float(policy_loss), float(value_loss), float(gn)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=0, help="stop after this long (0 = until logs/STOP)")
    ap.add_argument("--workers", type=int, default=6, help="self-play processes")
    ap.add_argument("--parallel", type=int, default=160, help="concurrent games per self-play worker")
    ap.add_argument("--gen-sims", type=int, default=100, help="MCTS visits per self-play move")
    ap.add_argument("--eval-sims", type=int, default=100, help="MCTS visits per arena move")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--temp-plies", type=int, default=30, help="plies sampled ∝ visits before playing greedily")
    ap.add_argument("--max-plies", type=int, default=512, help="self-play games longer than this are drawn")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr-schedule", type=str, default="0:0.02,300000:0.01,600000:0.002",
                    help="step:lr pairs")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--sample-ratio", type=float, default=4.0,
                    help="training samples drawn per generated position")
    ap.add_argument("--min-buffer", type=int, default=50_000, help="positions before training starts")
    ap.add_argument("--buffer-capacity", type=int, default=2_000_000, help="replay window (positions)")
    ap.add_argument("--eval-every", type=int, default=1500, help="min train steps between candidate snapshots")
    ap.add_argument("--eval-games", type=int, default=160)
    ap.add_argument("--arena-max-plies", type=int, default=300,
                    help="arena games longer than this are scored as draws")
    ap.add_argument("--promote-threshold", type=float, default=0.55)
    ap.add_argument("--sf-every", type=int, default=5, help="promotions between Stockfish checks (0 = never)")
    ap.add_argument("--sf-games", type=int, default=40)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--buffer", type=str, default="games")
    ap.add_argument("--ckpt-dir", type=str, default="checkpoints")
    ap.add_argument("--stats-every", type=float, default=120, help="seconds between progress lines")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    log_path = setup_logging("loop", args.log_level)
    log.info(f"args = {vars(args)}")
    if log_path:
        log.info(f"log file: {log_path}")
    _keep_awake()

    device = torch.device(args.device)
    log.info(f"device = {device}  cuda_available={torch.cuda.is_available()}")
    if device.type == "cuda":
        log.info(f"GPU = {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    ckpt_dir = Path(args.ckpt_dir)
    (ckpt_dir / "gen").mkdir(parents=True, exist_ok=True)
    stop_file = Path("logs") / "STOP"
    state = State(ckpt_dir / "state.json")
    schedule = _parse_schedule(args.lr_schedule)

    # -- networks ----------------------------------------------------------
    net = ChessNet(channels=args.channels, blocks=args.blocks).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    log.info(f"network: {args.channels}ch × {args.blocks} blocks = {n_params/1e6:.2f}M params")
    opt = SGD(net.parameters(), lr=_lr_at(schedule, state["step"]), momentum=0.9,
              weight_decay=args.weight_decay, nesterov=True)

    latest = ckpt_dir / "latest.pt"
    cur_best = best_path(ckpt_dir, state["version"])
    if latest.exists() and state.path.exists():
        ck = torch.load(latest, map_location=device)
        if isinstance(ck, dict) and "model" in ck:
            net.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            log.info(f"resumed candidate + optimizer from {latest} (step {ck.get('step')})")
        else:
            net.load_state_dict(ck)
            log.info(f"resumed candidate weights from {latest}")
    elif cur_best.exists():
        net.load_state_dict(torch.load(cur_best, map_location=device))
        log.info(f"initialised candidate from {cur_best}")
    if not cur_best.exists():
        state["created"] = state["created"] or time.strftime("%Y-%m-%d %H:%M:%S")
        _atomic_save(net.state_dict(), cur_best)
        _copy_retry(cur_best, ckpt_dir / "best.pt")
        state.save()
        log.warning(f"fresh run: initial best v{state['version']} written to {cur_best}")
    log.info(f"state: {state.d}")

    buffer = ReplayBuffer(args.buffer, capacity=args.buffer_capacity)
    log.info(f"replay buffer at {args.buffer}: {buffer.size} positions in memory")

    # -- workers -----------------------------------------------------------
    ctx = mp.get_context("spawn")
    version = ctx.Value("i", state["version"])
    stop_evt = ctx.Event()
    games_q, task_q, result_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    wcfg = {"device": args.device, "channels": args.channels, "blocks": args.blocks,
            "ckpt_dir": str(ckpt_dir), "gen_sims": args.gen_sims, "eval_sims": args.eval_sims,
            "c_puct": args.c_puct, "parallel": args.parallel, "temp_plies": args.temp_plies,
            "max_plies": args.max_plies, "arena_max_plies": args.arena_max_plies,
            "log_level": args.log_level}

    def spawn_sp(i):
        p = ctx.Process(target=selfplay_worker, args=(i, wcfg, version, games_q, stop_evt),
                        name=f"selfplay-{i}", daemon=True)
        p.start()
        return p

    sp_procs = [spawn_sp(i) for i in range(args.workers)]
    arena = ctx.Process(target=arena_worker, args=(wcfg, task_q, result_q, stop_evt),
                        name="arena", daemon=True)
    arena.start()
    log.info(f"started {args.workers} self-play workers × {args.parallel} games + 1 arena worker")

    # -- main loop ---------------------------------------------------------
    t_start = time.monotonic()
    pending_recs: list[dict] = []
    t_flush = t_stats = t_refresh = time.monotonic()
    arena_busy = False
    window = {"games": 0, "plies": 0, "sims": 0, "steps": 0,
              "pl": 0.0, "vl": 0.0, "gn": 0.0, "outcomes": Counter()}
    recent_lengths: deque = deque(maxlen=500)
    evals_seen: dict[int, int] = {}                # worker id → its cumulative net evals

    def save_trainer():
        _atomic_save({"model": net.state_dict(), "optimizer": opt.state_dict(),
                      "step": state["step"]}, latest)
        state.save()

    def flush():
        nonlocal pending_recs, t_flush
        if pending_recs:
            buffer.save_games(pending_recs)
            n = sum(len(r["values"]) for r in pending_recs if "values" in r)
            state["positions"] += n
            state["games"] += len(pending_recs)
            pending_recs = []
        t_flush = time.monotonic()

    def handle_result(res):
        nonlocal arena_busy
        arena_busy = False
        if "error" in res:
            log.error(f"arena task failed ({res['kind']}); see worker-arena log")
            if res["kind"] == "match":
                Path(res["a"]).unlink(missing_ok=True)
            return
        w, d, l = res["wdl"]
        n = max(1, w + d + l)
        score = (w + 0.5 * d) / n
        if res["kind"] == "match":
            v = state["version"]
            log.info(f"[arena] candidate@{res['step']} vs best v{v}: {w}W-{d}D-{l}L  "
                     f"score={score:.3f}  Elo ≈ {elo_gap(score):+.0f}  ({res['seconds']:.0f}s)")
            promoted = score >= args.promote_threshold
            _history(ckpt_dir, {"kind": "match", "step": res["step"], "best": v,
                                "wdl": [w, d, l], "score": score, "promoted": promoted})
            if promoted:
                nv = v + 1
                os.replace(res["a"], best_path(ckpt_dir, nv))
                _copy_retry(best_path(ckpt_dir, nv), ckpt_dir / "best.pt")
                state["version"] = nv
                state["promotions"] += 1
                version.value = nv
                state.save()
                log.info(f"[arena] ✓ PROMOTED step {res['step']} → best v{nv} "
                         f"(total promotions {state['promotions']})")
                _prune_generations(ckpt_dir, nv)
                if args.sf_every and state["promotions"] % args.sf_every == 0:
                    task_q.put({"kind": "stockfish", "a": str(best_path(ckpt_dir, nv)),
                                "level": state["sf_level"], "n_games": args.sf_games,
                                "version": nv})
                    arena_busy = True
            else:
                Path(res["a"]).unlink(missing_ok=True)
                log.info(f"[arena] ✗ candidate rejected; keeping best v{v}")
        else:
            lvl = res["level"]
            log.info(f"[stockfish] best v{res['version']} vs Stockfish skill {lvl}: "
                     f"{w}W-{d}D-{l}L  score={score:.3f}  ({res['seconds']:.0f}s)")
            _history(ckpt_dir, {"kind": "stockfish", "best": res["version"], "level": lvl,
                                "wdl": [w, d, l], "score": score})
            if score >= 0.6 and lvl < 20:
                state["sf_level"] = lvl + 1
                log.info(f"[stockfish] beat skill {lvl}; next check at skill {lvl + 1}")
            state.save()

    reason = "?"
    try:
        while True:
            now = time.monotonic()
            if stop_file.exists():
                reason = "STOP file"
                break
            if args.hours and now - t_start > args.hours * 3600:
                reason = f"--hours {args.hours} reached"
                break

            # Keep workers alive.
            for i, p in enumerate(sp_procs):
                if not p.is_alive():
                    log.error(f"self-play worker {i} died (exit {p.exitcode}); restarting")
                    sp_procs[i] = spawn_sp(i)
                    evals_seen.pop(i, None)
            if not arena.is_alive():
                log.error(f"arena worker died (exit {arena.exitcode}); restarting")
                arena = ctx.Process(target=arena_worker, args=(wcfg, task_q, result_q, stop_evt),
                                    name="arena", daemon=True)
                arena.start()
                arena_busy = False

            # Ingest finished games.
            while True:
                try:
                    rec = games_q.get_nowait()
                except queue.Empty:
                    break
                pending_recs.append(rec)
                window["games"] += 1
                window["plies"] += rec["plies"]
                w_id = rec.get("worker", 0)
                window["sims"] += max(0, rec.get("evals_total", 0) - evals_seen.get(w_id, 0))
                evals_seen[w_id] = rec.get("evals_total", 0)
                window["outcomes"][rec["outcome"].split(" (")[0]] += 1
                recent_lengths.append(rec["plies"])
            if len(pending_recs) >= 64 or (pending_recs and now - t_flush > 120):
                flush()
            if now - t_refresh > 60:
                added = buffer.refresh()            # e.g. Lichess farm chunks
                if added:
                    state["positions"] += added
                    log.info(f"ingested {added} external positions (farm)")
                t_refresh = now

            # Arena results / scheduling.
            try:
                handle_result(result_q.get_nowait())
            except queue.Empty:
                pass
            if (not arena_busy and buffer.size >= args.min_buffer
                    and state["step"] - state["last_snapshot_step"] >= args.eval_every):
                snap = ckpt_dir / "gen" / f"cand_{state['step']:08d}.pt"
                _atomic_save(net.state_dict(), snap)
                state["last_snapshot_step"] = state["step"]
                save_trainer()
                task_q.put({"kind": "match", "a": str(snap),
                            "b": str(best_path(ckpt_dir, state["version"])),
                            "n_games": args.eval_games, "step": state["step"]})
                arena_busy = True

            # Train, paced by data generated.
            allowed = int(state["positions"] * args.sample_ratio / args.batch_size) - state["step"]
            if buffer.size < args.min_buffer or allowed <= 0:
                time.sleep(0.25)
            else:
                net.train()
                for _ in range(min(allowed, 50)):
                    lr = _lr_at(schedule, state["step"])
                    for g in opt.param_groups:
                        g["lr"] = lr
                    pl, vl, gn = train_step(net, opt, buffer.sample(args.batch_size), device)
                    state["step"] += 1
                    window["steps"] += 1
                    window["pl"] += pl; window["vl"] += vl; window["gn"] += gn
                    if state["step"] % 2000 == 0:
                        save_trainer()

            # Progress line.
            if now - t_stats >= args.stats_every:
                dt = now - t_stats
                s = max(1, window["steps"])
                g = max(1, window["games"])
                oc = window["outcomes"]
                log.info(
                    f"[stats] games {state['games'] + len(pending_recs)} (+{window['games']/dt*3600:.0f}/h)  "
                    f"positions {state['positions']}  evals/s {window['sims']/dt:.0f}  "
                    f"avg plies {window['plies']/g:.0f}  "
                    f"W/B/D {oc['white wins']}/{oc['black wins']}/{sum(v for k, v in oc.items() if k.startswith('draw'))}  "
                    f"| step {state['step']} (+{window['steps']/dt:.1f}/s)  "
                    f"loss p={window['pl']/s:.3f} v={window['vl']/s:.3f} gn={window['gn']/s:.2f}  "
                    f"lr={_lr_at(schedule, state['step'])}  "
                    f"| buffer {buffer.size}  best v{state['version']}  "
                    f"sf_level {state['sf_level']}")
                window = {"games": 0, "plies": 0, "sims": 0, "steps": 0,
                          "pl": 0.0, "vl": 0.0, "gn": 0.0, "outcomes": Counter()}
                t_stats = now
                state.save()
    except KeyboardInterrupt:
        reason = "keyboard interrupt"
    finally:
        log.info(f"stopping ({reason}); saving state")
        stop_evt.set()
        flush()
        save_trainer()
        for p in sp_procs + [arena]:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        log.info(f"loop stopped: step {state['step']}, best v{state['version']}, "
                 f"{state['games']} games, {state['promotions']} promotions")


def _prune_generations(ckpt_dir: Path, newest: int, keep_last: int = 10, keep_every: int = 10) -> None:
    for p in (ckpt_dir / "gen").glob("best_*.pt"):
        try:
            v = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if v < newest - keep_last and v % keep_every != 0:
            try:
                p.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
