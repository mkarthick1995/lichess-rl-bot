"""Training orchestrator: self-play → train → evaluate → gate, repeated.

Keeps two networks in memory:
    best       — the strongest checkpoint so far; generates self-play games.
    candidate  — trained on the replay buffer; promoted to best only if it
                 beats best by a win-rate margin in the arena.

Run:
    python -m src.loop --iterations 100 \
        --games-per-iter 25 --train-steps 400 --eval-games 20

Everything is checkpointed under --ckpt-dir so the loop is resumable: on
restart it reloads best.pt and continues.
"""
from __future__ import annotations

import argparse
import copy
import faulthandler
import logging
import time
from pathlib import Path

# Dump a native traceback on hard faults (segfault, CUDA abort) so a silent
# process death leaves evidence in stderr.
faulthandler.enable()

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD

from ._log import setup_logging
from .network import ChessNet
from .replay_buffer import ReplayBuffer
from .self_play import _play_one_game
from .evaluate import _play_game

log = logging.getLogger("loop")


def _build_net(channels: int, blocks: int, device) -> ChessNet:
    return ChessNet(channels=channels, blocks=blocks).to(device)


def _generation_phase(best: ChessNet, device, buffer: ReplayBuffer,
                      n_games: int, n_sims: int, it: int) -> dict:
    best.eval()
    t0 = time.monotonic()
    plies_total = 0
    outcomes = {"white": 0, "black": 0, "draw/other": 0}
    for g in range(n_games):
        states, policies, values, stats = _play_one_game(
            best, device, n_sims, game_id=f"it{it}-g{g+1}/{n_games}")
        buffer.save_game(states, policies, values)
        plies_total += stats["plies"]
        oc = stats["outcome"]
        if oc.startswith("white"):   outcomes["white"] += 1
        elif oc.startswith("black"): outcomes["black"] += 1
        else:                        outcomes["draw/other"] += 1
    dt = time.monotonic() - t0
    log.info(f"[it {it}] generation: {n_games} games, {plies_total} plies, "
             f"{dt:.1f}s ({dt/max(1,n_games):.1f}s/game)  outcomes={outcomes}  "
             f"buffer={buffer.n_games()} games")
    return {"games": n_games, "plies": plies_total, "seconds": dt}


def _train_phase(candidate: ChessNet, device, buffer: ReplayBuffer,
                 steps: int, batch_size: int, lr: float, weight_decay: float,
                 log_every: int, it: int) -> dict:
    candidate.train()
    optimizer = SGD(candidate.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    running = {"total": 0.0, "policy": 0.0, "value": 0.0}
    last_gn = 0.0
    first_total = last_total = None

    for step in range(steps):
        s_np, p_np, v_np = buffer.sample_batch(batch_size)
        states   = torch.from_numpy(s_np).to(device)
        policies = torch.from_numpy(p_np).to(device)
        values   = torch.from_numpy(v_np).to(device)

        logits, value_pred = candidate(states)
        log_probs   = F.log_softmax(logits, dim=1)
        policy_loss = -(policies * log_probs).sum(dim=1).mean()
        value_loss  = F.mse_loss(value_pred, values)
        total = policy_loss + value_loss

        optimizer.zero_grad()
        total.backward()
        last_gn = float(torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0))
        optimizer.step()

        running["total"]  += total.item()
        running["policy"] += policy_loss.item()
        running["value"]  += value_loss.item()
        if first_total is None:
            first_total = total.item()
        last_total = total.item()

        if (step + 1) % log_every == 0:
            n = log_every
            log.info(f"[it {it}] train step {step+1}/{steps}  "
                     f"loss={running['total']/n:.3f}  policy={running['policy']/n:.3f}  "
                     f"value={running['value']/n:.3f}  grad_norm={last_gn:.2f}")
            running = {"total": 0.0, "policy": 0.0, "value": 0.0}

    log.info(f"[it {it}] training done: loss {first_total:.3f} → {last_total:.3f}")
    return {"loss_first": first_total, "loss_last": last_total}


def _eval_phase(candidate: ChessNet, best: ChessNet, device,
                n_games: int, n_sims: int, it: int) -> float:
    candidate.eval(); best.eval()
    wins = draws = losses = 0
    for i in range(n_games):
        r = _play_game(candidate, best, device, n_sims,
                       a_plays_white=(i % 2 == 0), game_id=f"it{it}-e{i+1}/{n_games}")
        if   r > 0: wins   += 1
        elif r < 0: losses += 1
        else:       draws  += 1
    score = wins + 0.5 * draws
    win_rate = score / n_games if n_games else 0.0
    log.info(f"[it {it}] eval: candidate {wins}W-{draws}D-{losses}L vs best  "
             f"win_rate={win_rate:.3f}")
    return win_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--games-per-iter", type=int, default=25)
    ap.add_argument("--train-steps", type=int, default=400)
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--gen-sims", type=int, default=100,
                    help="MCTS sims per move during self-play generation")
    ap.add_argument("--eval-sims", type=int, default=100,
                    help="MCTS sims per move during arena evaluation")
    ap.add_argument("--promote-threshold", type=float, default=0.55,
                    help="candidate win-rate needed to replace best")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--buffer", type=str, default="games")
    ap.add_argument("--ckpt-dir", type=str, default="checkpoints")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    log_path = setup_logging("loop", args.log_level)
    log.info(f"args = {vars(args)}")
    if log_path:
        log.info(f"log file: {log_path}")

    device = torch.device(args.device)
    log.info(f"device = {device}  cuda_available={torch.cuda.is_available()}")
    if device.type == "cuda":
        log.info(f"GPU = {torch.cuda.get_device_name(0)}")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path   = ckpt_dir / "best.pt"
    latest_path = ckpt_dir / "latest.pt"

    best = _build_net(args.channels, args.blocks, device)
    n_params = sum(p.numel() for p in best.parameters())
    log.info(f"network: {args.channels}ch × {args.blocks} blocks = {n_params/1e6:.2f}M params")
    if best_path.exists():
        best.load_state_dict(torch.load(best_path, map_location=device))
        log.info(f"resumed best from {best_path}")
    else:
        torch.save(best.state_dict(), best_path)
        log.warning(f"no best checkpoint; initialised random best at {best_path}")

    buffer = ReplayBuffer(args.buffer)
    log.info(f"replay buffer at {args.buffer} ({buffer.n_games()} games)")

    promotions = 0
    it = 1
    while it <= args.iterations:
        t_it = time.monotonic()
        log.info(f"════════ iteration {it}/{args.iterations} ════════")

        try:
            _generation_phase(best, device, buffer,
                               args.games_per_iter, args.gen_sims, it)

            candidate = _build_net(args.channels, args.blocks, device)
            candidate.load_state_dict(copy.deepcopy(best.state_dict()))
            _train_phase(candidate, device, buffer,
                         args.train_steps, args.batch_size, args.lr,
                         args.weight_decay, args.log_every, it)
            torch.save(candidate.state_dict(), latest_path)

            win_rate = _eval_phase(candidate, best, device,
                                   args.eval_games, args.eval_sims, it)

            if win_rate >= args.promote_threshold:
                best.load_state_dict(copy.deepcopy(candidate.state_dict()))
                torch.save(best.state_dict(), best_path)
                promotions += 1
                log.info(f"[it {it}] ✓ PROMOTED candidate to best "
                         f"(win_rate {win_rate:.3f} ≥ {args.promote_threshold})  "
                         f"total promotions={promotions}")
            else:
                log.info(f"[it {it}] ✗ candidate rejected "
                         f"(win_rate {win_rate:.3f} < {args.promote_threshold}); keeping best")

            log.info(f"[it {it}] iteration took {time.monotonic()-t_it:.1f}s")

        except KeyboardInterrupt:
            log.warning("interrupted by user; exiting cleanly")
            break
        except Exception:
            # A soft error (bad batch, transient CUDA error, etc.) must not kill
            # a multi-day run. Log the traceback, free the GPU cache, and retry
            # the same iteration. Generated games are already banked in the
            # buffer, so no self-play work is lost.
            log.exception(f"[it {it}] iteration failed; recovering and retrying")
            try:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass
            time.sleep(5)
            continue

        it += 1

    log.info(f"loop finished: {it - 1} iterations completed, {promotions} promotions")


if __name__ == "__main__":
    main()
