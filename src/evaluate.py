"""Arena: pit two checkpoints against each other, or a checkpoint against
Stockfish at a fixed skill level, with all games played concurrently.

    python -m src.evaluate --a checkpoints/best.pt --b checkpoints/gen/best_00010.pt
    python -m src.evaluate --a checkpoints/best.pt --stockfish-level 3
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import chess
import chess.engine
import torch

from ._log import setup_logging
from .inference import BatchEvaluator
from .self_play import GameRunner, GameSpec, SearchConfig, load_net

log = logging.getLogger("evaluate")

STOCKFISH = Path(__file__).resolve().parent.parent / "tools" / "stockfish" / "stockfish.exe"


def elo_gap(score: float) -> float:
    score = min(max(score, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / score - 1)


def _run(runner: GameRunner, specs: list[GameSpec], parallel: int, on_game=None) -> tuple[int, int, int]:
    """Play specs (tag = True when A has white); returns A's (W, D, L)."""
    w = d = l = 0
    queue = list(specs)
    while queue or runner.games:
        while queue and len(runner.games) < parallel:
            runner.add(queue.pop())
        for rec in runner.step():
            a_white = rec["tag"]
            r = rec["result"] if a_white else -rec["result"]
            if r > 0:   w += 1
            elif r < 0: l += 1
            else:       d += 1
            if on_game:
                on_game(rec, (w, d, l))
    return w, d, l


def play_match(ev_a, ev_b, n_games: int, cfg: SearchConfig, parallel: int = 64,
               opening_plies: int = 6, max_plies: int = 512, on_game=None) -> tuple[int, int, int]:
    """A vs B, colours alternating. The first `opening_plies` plies are
    sampled ∝ visit counts so games differ (otherwise every game with the
    same colours would be identical)."""
    specs = []
    for i in range(n_games):
        a_white = i % 2 == 0
        specs.append(GameSpec(players={chess.WHITE: "a" if a_white else "b",
                                       chess.BLACK: "b" if a_white else "a"},
                              add_noise=False, temp_plies=opening_plies,
                              record=False, max_plies=max_plies, tag=a_white))
    return _run(GameRunner({"a": ev_a, "b": ev_b}, cfg), specs, parallel, on_game)


class StockfishPlayer:
    """Callable board → move, Stockfish at a fixed Skill Level (0–20)."""

    def __init__(self, level: int, movetime: float = 0.05, path: Path = STOCKFISH):
        self.engine = chess.engine.SimpleEngine.popen_uci(str(path))
        self.engine.configure({"Skill Level": level, "Threads": 1, "Hash": 16})
        self.limit = chess.engine.Limit(time=movetime)

    def __call__(self, board: chess.Board) -> chess.Move:
        return self.engine.play(board, self.limit).move

    def close(self) -> None:
        try:
            self.engine.quit()
        except Exception:
            pass


def play_stockfish(ev, level: int, n_games: int, cfg: SearchConfig, parallel: int = 32,
                   opening_plies: int = 4, on_game=None) -> tuple[int, int, int]:
    """Net (A) vs Stockfish skill `level`; returns the net's (W, D, L)."""
    sf = StockfishPlayer(level)
    try:
        specs = []
        for i in range(n_games):
            a_white = i % 2 == 0
            specs.append(GameSpec(players={chess.WHITE: "a" if a_white else sf,
                                           chess.BLACK: sf if a_white else "a"},
                                  add_noise=False, temp_plies=opening_plies,
                                  record=False, tag=a_white))
        return _run(GameRunner({"a": ev}, cfg), specs, parallel, on_game)
    finally:
        sf.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="challenger checkpoint")
    ap.add_argument("--b", help="incumbent checkpoint")
    ap.add_argument("--stockfish-level", type=int, default=None,
                    help="play A against Stockfish at this Skill Level (0-20) instead of B")
    ap.add_argument("--n-games", type=int, default=40)
    ap.add_argument("--n-simulations", type=int, default=100)
    ap.add_argument("--parallel", type=int, default=64)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()
    if args.b is None and args.stockfish_level is None:
        ap.error("give --b or --stockfish-level")

    setup_logging("evaluate", args.log_level)
    log.info(f"args = {vars(args)}")
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    cfg = SearchConfig(n_sims=args.n_simulations)
    ev_a = BatchEvaluator(load_net(args.a, args.channels, args.blocks, device), device)

    def on_game(rec, wdl):
        log.info(f"[game] {'A=W' if rec['tag'] else 'A=B'}  plies={rec['plies']}  "
                 f"{rec['outcome']}   running A: {wdl[0]}W-{wdl[1]}D-{wdl[2]}L")

    if args.stockfish_level is not None:
        log.info(f"A = {args.a}  vs  Stockfish skill {args.stockfish_level}")
        w, d, l = play_stockfish(ev_a, args.stockfish_level, args.n_games, cfg,
                                 min(args.parallel, 32), on_game=on_game)
    else:
        ev_b = BatchEvaluator(load_net(args.b, args.channels, args.blocks, device), device)
        log.info(f"A = {args.a}  vs  B = {args.b}")
        w, d, l = play_match(ev_a, ev_b, args.n_games, cfg, args.parallel, on_game=on_game)

    wr = (w + 0.5 * d) / max(1, w + d + l)
    log.info(f"final: {w}W-{d}D-{l}L  score={wr:.3f}  Elo gap ≈ {elo_gap(wr):+.0f}")


if __name__ == "__main__":
    main()
