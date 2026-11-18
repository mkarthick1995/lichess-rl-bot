"""Clocked bullet games, to test time management and to gate new networks.

Real thinking time is charged to each side's clock (plus a simulated network
lag for our engines); a side whose clock runs out loses — or draws if the
other side cannot mate (Lichess rule).

    # our engine vs local Stockfish at a skill level
    python -m src.bullet_arena --checkpoint checkpoints/best.pt --level 1 --tc 1+0 --games 10

    # network vs network (the promotion gate): paired games from fixed
    # openings, colours swapped, no pondering (one GPU, fair)
    python -m src.bullet_arena --checkpoint checkpoints/v2/pretrain_best.pt \
        --vs checkpoints/best.pt --tc 1+0 --games 40

Either network generation is accepted (auto-detected from the checkpoint).
"""
from __future__ import annotations

import argparse
import logging
import math
import time
from collections import Counter
from pathlib import Path

import chess
import chess.engine
import torch

from ._log import setup_logging
from .bullet import BulletConfig, BulletEngine, Clock
from .evaluate import STOCKFISH, elo_gap
from .inference2 import make_evaluator

log = logging.getLogger("bullet_arena")

# Common openings (UCI, 4-6 plies) so engine-vs-engine games differ.
OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6 f1b5", "e2e4 e7e5 g1f3 b8c6 f1c4", "e2e4 c7c5 g1f3 d7d6 d2d4",
    "e2e4 c7c5 b1c3 b8c6 g2g3", "e2e4 e7e6 d2d4 d7d5 b1c3", "e2e4 c7c6 d2d4 d7d5 e4e5",
    "e2e4 d7d5 e4d5 d8d5 b1c3", "e2e4 g7g6 d2d4 f8g7 b1c3", "d2d4 d7d5 c2c4 e7e6 b1c3",
    "d2d4 d7d5 c2c4 c7c6 g1f3", "d2d4 g8f6 c2c4 g7g6 b1c3", "d2d4 g8f6 c2c4 e7e6 g1f3",
    "d2d4 d7d5 c1f4 g8f6 e2e3", "c2c4 e7e5 b1c3 g8f6 g2g3", "g1f3 d7d5 g2g3 g8f6 f1g2",
    "e2e4 e7e5 f2f4 e5f4 g1f3", "d2d4 f7f5 g2g3 g8f6 f1g2", "e2e4 b7b6 d2d4 c8b7 f1d3",
    "b2b3 e7e5 c1b2 b8c6 e2e3", "e2e4 d7d6 d2d4 g8f6 b1c3",
]


def parse_tc(tc: str) -> tuple[int, int]:
    """'1+0' / '0.5+0' / '2+1' (minutes+seconds; >= 10 means seconds) → (base ms, inc ms)."""
    base, _, inc = tc.partition("+")
    b = float(base)
    base_s = b * 60 if b < 10 else b
    return int(base_s * 1000), int(float(inc or 0) * 1000)


class NetPlayer:
    def __init__(self, checkpoint: str, device, ponder: bool, name: str):
        self.engine = BulletEngine(make_evaluator(checkpoint, device), BulletConfig(ponder=ponder))
        self.name = name
        self.stats = Counter()

    def move(self, board, clocks, inc_ms, lag_ms):
        info = self.engine.choose(board, Clock(clocks[board.turn], clocks[not board.turn], inc_ms, inc_ms))
        self.stats["moves"] += 1
        self.stats["sims"] += info.sims
        self.stats[f"mode:{info.mode}"] += 1
        return info.move, lag_ms

    def new_game(self):
        self.engine.close()


class SfPlayer:
    def __init__(self, level: int):
        self.sf = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
        self.sf.configure({"Skill Level": level, "Threads": 1, "Hash": 16})
        self.name = f"stockfish-{level}"
        self.stats = Counter()

    def move(self, board, clocks, inc_ms, lag_ms):
        limit = chess.engine.Limit(white_clock=clocks[chess.WHITE] / 1000,
                                   black_clock=clocks[chess.BLACK] / 1000,
                                   white_inc=inc_ms / 1000, black_inc=inc_ms / 1000)
        return self.sf.play(board, limit).move, 0

    def new_game(self):
        pass

    def close(self):
        self.sf.quit()


def play_game(a, b, a_white: bool, base_ms: int, inc_ms: int, lag_ms: int,
              opening: str = "", max_plies: int = 600) -> dict:
    board = chess.Board()
    for u in opening.split():
        board.push_uci(u)
    clocks = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
    players = {chess.WHITE: a if a_white else b, chess.BLACK: b if a_white else a}
    a.new_game(); b.new_game()
    result = reason = None
    while not board.is_game_over(claim_draw=False) and len(board.move_stack) < max_plies:
        mover = board.turn
        t0 = time.monotonic()
        move, extra = players[mover].move(board, clocks, inc_ms, lag_ms)
        clocks[mover] -= (time.monotonic() - t0) * 1000 + extra
        if clocks[mover] <= 0:
            if board.has_insufficient_material(not mover):
                result, reason = 0.0, "timeout vs insufficient material"
            else:
                result, reason = (1.0 if not mover == chess.WHITE else -1.0), "flag"
            break
        clocks[mover] += inc_ms
        board.push(move)
    a.new_game(); b.new_game()
    if result is None:
        outcome = board.outcome(claim_draw=False)
        if outcome is None:
            result, reason = 0.0, "max plies"
        else:
            result = 0.0 if outcome.winner is None else (1.0 if outcome.winner else -1.0)
            reason = outcome.termination.name.lower()
    a_color = chess.WHITE if a_white else chess.BLACK
    return {"a_result": result if a_white else -result, "reason": reason,
            "plies": len(board.move_stack),
            "a_clock_ms": clocks[a_color], "b_clock_ms": clocks[not a_color]}


def run_match(a, b, games: int, base_ms: int, inc_ms: int, lag_ms: int,
              use_openings: bool, before_game=None) -> dict:
    tot = Counter()
    for g in range(games):
        if before_game is not None:
            before_game()
        opening = OPENINGS[(g // 2) % len(OPENINGS)] if use_openings else ""
        r = play_game(a, b, g % 2 == 0, base_ms, inc_ms, lag_ms, opening)
        key = "W" if r["a_result"] > 0 else "L" if r["a_result"] < 0 else "D"
        tot[key] += 1
        tot[f"{key}:{r['reason']}"] += 1
        log.info(f"game {g + 1}: {a.name} {'white' if g % 2 == 0 else 'black'} {key} ({r['reason']}) "
                 f"plies={r['plies']} clocks {a.name} {r['a_clock_ms'] / 1000:.1f}s / "
                 f"{b.name} {r['b_clock_ms'] / 1000:.1f}s")
    n = max(1, games)
    score = (tot["W"] + 0.5 * tot["D"]) / n
    # 95% interval of the score (normal approximation)
    se = math.sqrt(max(score * (1 - score), 1e-9) / n)
    detail = ", ".join(f"{k}={v}" for k, v in sorted(tot.items()) if ":" in k)
    log.info(f"final {a.name} vs {b.name}: {tot['W']}W-{tot['D']}D-{tot['L']}L  score={score:.3f} "
             f"(95% {max(0, score - 1.96 * se):.2f}-{min(1, score + 1.96 * se):.2f})  "
             f"Elo ≈ {elo_gap(score):+.0f}  [{detail}]")
    return {"w": tot["W"], "d": tot["D"], "l": tot["L"], "score": score, "se": se}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--vs", help="opponent network checkpoint (net-vs-net gate)")
    ap.add_argument("--level", type=int, default=1, help="Stockfish Skill Level 0-20 (without --vs)")
    ap.add_argument("--tc", default="1+0", help="minutes+increment, e.g. 1+0, 0.5+0, 2+1")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--lag-ms", type=int, default=60, help="simulated network lag per move of ours")
    ap.add_argument("--no-ponder", action="store_true")
    ap.add_argument("--pause-training", action="store_true",
                    help="pause the training loop (logs/PAUSE) while these games run")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    setup_logging("bullet_arena", "INFO")
    torch.backends.cudnn.benchmark = True
    dev = torch.device(args.device)
    base_ms, inc_ms = parse_tc(args.tc)
    pause = Path("logs") / "PAUSE"

    a_name = Path(args.checkpoint).stem
    if args.vs:
        a = NetPlayer(args.checkpoint, dev, ponder=False, name=a_name)
        b_name = Path(args.vs).stem
        b = NetPlayer(args.vs, dev, ponder=False, name=b_name if b_name != a_name else b_name + "'")
    else:
        a = NetPlayer(args.checkpoint, dev, ponder=not args.no_ponder, name=a_name)
        b = SfPlayer(args.level)
    try:
        if args.pause_training:
            pause.parent.mkdir(exist_ok=True)
            pause.touch()
            time.sleep(3)                    # let the trainer and workers notice
        # warm up cuDNN so the first move of game 1 is not slow
        a.move(chess.Board(), {chess.WHITE: 60000, chess.BLACK: 60000}, 0, 0)
        if isinstance(b, NetPlayer):
            b.move(chess.Board(), {chess.WHITE: 60000, chess.BLACK: 60000}, 0, 0)
        # the pause file is touched before every game (it goes stale after 10 min)
        heartbeat = pause.touch if args.pause_training else None
        run_match(a, b, args.games, base_ms, inc_ms, args.lag_ms,
                  use_openings=bool(args.vs), before_game=heartbeat)
    finally:
        if isinstance(b, SfPlayer):
            b.close()
        a.engine.close()
        if isinstance(b, NetPlayer):
            b.engine.close()
        if args.pause_training:
            pause.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
