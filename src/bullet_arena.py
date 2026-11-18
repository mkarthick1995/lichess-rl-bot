"""Clocked bullet games against local Stockfish, to tune time management
without Lichess. Both sides' real thinking time is charged to their clocks
(plus a simulated network lag for us), and a side whose clock reaches zero
loses on time — or draws, if the other side cannot mate (Lichess rule).

    python -m src.bullet_arena --checkpoint checkpoints/best.pt --level 1 --tc 60+0 --games 10
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from pathlib import Path

import chess
import chess.engine
import torch

from ._log import setup_logging
from .bullet import BulletConfig, BulletEngine, Clock
from .evaluate import STOCKFISH
from .inference import BatchEvaluator
from .self_play import load_net

log = logging.getLogger("bullet_arena")


def parse_tc(tc: str) -> tuple[int, int]:
    """'60+0' / '1+0' (minutes if < 10 and no seconds suffix) → (base ms, inc ms)."""
    base, _, inc = tc.partition("+")
    b = float(base)
    base_s = b * 60 if b < 10 else b
    return int(base_s * 1000), int(float(inc or 0) * 1000)


def play_game(engine: BulletEngine, sf: chess.engine.SimpleEngine, net_white: bool,
              base_ms: int, inc_ms: int, lag_ms: int, max_plies: int = 600) -> dict:
    board = chess.Board()
    clocks = {chess.WHITE: base_ms, chess.BLACK: base_ms}
    net_color = chess.WHITE if net_white else chess.BLACK
    stats = Counter()
    modes = Counter()
    result, reason = None, None
    engine.close()
    while not board.is_game_over(claim_draw=False) and len(board.move_stack) < max_plies:
        mover = board.turn
        t0 = time.monotonic()
        if mover == net_color:
            info = engine.choose(board, Clock(clocks[mover], clocks[not mover], inc_ms, inc_ms))
            move = info.move
            spent = (time.monotonic() - t0) * 1000 + lag_ms
            stats["net_moves"] += 1
            stats["net_ms"] += spent
            stats["net_sims"] += info.sims
            modes[info.mode] += 1
        else:
            limit = chess.engine.Limit(
                white_clock=clocks[chess.WHITE] / 1000, black_clock=clocks[chess.BLACK] / 1000,
                white_inc=inc_ms / 1000, black_inc=inc_ms / 1000)
            move = sf.play(board, limit).move
            spent = (time.monotonic() - t0) * 1000   # (we keep pondering meanwhile)
        clocks[mover] -= spent
        if clocks[mover] <= 0:
            other = not mover
            if board.has_insufficient_material(other):
                result, reason = 0.0, "timeout vs insufficient material"
            else:
                result = 1.0 if other == chess.WHITE else -1.0
                reason = "flag"
            break
        clocks[mover] += inc_ms
        board.push(move)
    engine.close()
    if result is None:
        outcome = board.outcome(claim_draw=False)
        if outcome is None:
            result, reason = 0.0, "max plies"
        else:
            result = 0.0 if outcome.winner is None else (1.0 if outcome.winner else -1.0)
            reason = outcome.termination.name.lower()
    net_result = result if net_white else -result
    return {"net_result": net_result, "reason": reason, "plies": len(board.move_stack),
            "net_clock_ms": clocks[net_color], "sf_clock_ms": clocks[not net_color],
            "avg_move_ms": stats["net_ms"] / max(1, stats["net_moves"]),
            "avg_sims": stats["net_sims"] / max(1, stats["net_moves"]),
            "modes": dict(modes), "pgn_moves": " ".join(m.uci() for m in board.move_stack)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--level", type=int, default=1, help="Stockfish Skill Level 0-20")
    ap.add_argument("--tc", default="1+0", help="minutes+increment, e.g. 1+0, 0.5+0, 2+1")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--lag-ms", type=int, default=60, help="simulated network lag per move of ours")
    ap.add_argument("--no-ponder", action="store_true")
    ap.add_argument("--pause-training", action="store_true",
                    help="pause the training loop (logs/PAUSE) while these games run")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    setup_logging("bullet_arena", "INFO")
    torch.backends.cudnn.benchmark = True
    dev = torch.device(args.device)
    net = load_net(args.checkpoint, args.channels, args.blocks, dev)
    net.eval()
    engine = BulletEngine(BatchEvaluator(net, dev), BulletConfig(ponder=not args.no_ponder))
    base_ms, inc_ms = parse_tc(args.tc)

    sf = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
    sf.configure({"Skill Level": args.level, "Threads": 1, "Hash": 16})
    tot = Counter()
    pause = Path("logs") / "PAUSE"
    try:
        if args.pause_training:
            pause.parent.mkdir(exist_ok=True)
            pause.touch()
            time.sleep(3)                    # let the trainer and workers notice
        for g in range(args.games):
            if args.pause_training:
                pause.touch()                # heartbeat (stale after 10 min)
            r = play_game(engine, sf, g % 2 == 0, base_ms, inc_ms, args.lag_ms)
            key = "W" if r["net_result"] > 0 else "L" if r["net_result"] < 0 else "D"
            tot[key] += 1
            tot[f"{key}:{r['reason']}"] += 1
            log.info(f"game {g + 1}: net {'white' if g % 2 == 0 else 'black'} {key} ({r['reason']}) "
                     f"plies={r['plies']} clocks net {r['net_clock_ms'] / 1000:.1f}s / sf {r['sf_clock_ms'] / 1000:.1f}s "
                     f"avg {r['avg_move_ms']:.0f} ms/move, {r['avg_sims']:.0f} sims  modes={r['modes']}")
    finally:
        sf.quit()
        engine.close()
        if args.pause_training:
            pause.unlink(missing_ok=True)
    n = max(1, args.games)
    score = (tot["W"] + 0.5 * tot["D"]) / n
    detail = ", ".join(f"{k}={v}" for k, v in sorted(tot.items()) if ":" in k)
    log.info(f"final vs Stockfish skill {args.level} @ {args.tc}: "
             f"{tot['W']}W-{tot['D']}D-{tot['L']}L  score={score:.2f}  [{detail}]")


if __name__ == "__main__":
    main()
