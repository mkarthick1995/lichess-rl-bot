"""Play many games against Lichess's built-in Stockfish ("Play with the
computer") in a row, recording each one into the replay buffer as training
data, with the Stockfish level automatically ramped up or down based on the
bot's recent win rate.

USE AGAINST COMPUTER OPPONENTS ONLY. Playing against humans on Lichess
violates their Terms of Service — this farm only ever starts games via
lobby.start_stockfish_game, which is hard-coded to Lichess's AI setup screen.

Runs for a fixed number of games then exits (unlike src/loop.py's forever-
until-STOP-file convention) — invoke it repeatedly (e.g. from a scheduler)
for sustained data generation.

Usage:
    python -m src.lichess.farm --checkpoint checkpoints/best.pt --n-games 50
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import torch
from playwright.async_api import async_playwright

from .._log import setup_logging
from ..network import ChessNet
from ..replay_buffer import ReplayBuffer
from . import lobby
from .play import run_recorded_game
from ._debug import save_screenshot

log = logging.getLogger("lichess.farm")


class LevelRamp:
    """Tracks a rolling win/draw/loss record at the current Stockfish level
    and bumps the level up or down once enough recent games cross a
    threshold. Draws count as half a win, matching evaluate.py's scoring."""

    def __init__(self, start_level: int, *, window: int = 10,
                promote_score: float = 0.6, demote_score: float = 0.2,
                min_level: int = 1, max_level: int = 8):
        self.level = max(min_level, min(max_level, start_level))
        self.window = window
        self.promote_score = promote_score
        self.demote_score = demote_score
        self.min_level = min_level
        self.max_level = max_level
        self._recent: deque[float] = deque(maxlen=window)

    def record(self, result: str) -> None:
        score = {"win": 1.0, "draw": 0.5, "loss": 0.0}.get(result)
        if score is None:
            return  # "aborted" games don't count toward the ramp
        self._recent.append(score)

    def maybe_ramp(self) -> str | None:
        """Call after each recorded result. Returns 'up', 'down', or None."""
        if len(self._recent) < self.window:
            return None
        avg = sum(self._recent) / len(self._recent)
        if avg >= self.promote_score and self.level < self.max_level:
            self.level += 1
            self._recent.clear()
            log.info(f"ramp: score {avg:.2f} >= {self.promote_score} over "
                     f"{self.window} games -> level up to {self.level}")
            return "up"
        if avg <= self.demote_score and self.level > self.min_level:
            self.level -= 1
            self._recent.clear()
            log.info(f"ramp: score {avg:.2f} <= {self.demote_score} over "
                     f"{self.window} games -> level down to {self.level}")
            return "down"
        return None


async def _farm_async(args) -> None:
    device = torch.device(args.device)
    log.info(f"device = {device}  cuda_available={torch.cuda.is_available()}")

    net = ChessNet(channels=args.channels, blocks=args.blocks).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    log.info(f"network: {args.channels}ch x {args.blocks} blocks = {n_params/1e6:.2f}M params")
    if args.checkpoint and Path(args.checkpoint).exists():
        net.load_state_dict(torch.load(args.checkpoint, map_location=device))
        log.info(f"loaded checkpoint: {args.checkpoint}")
    else:
        log.warning(f"checkpoint missing ({args.checkpoint}); using random init — plays badly")
    net.eval()

    buffer = ReplayBuffer(args.buffer, load=False)   # write-only; the trainer ingests new chunks
    log.info(f"replay buffer at {args.buffer} (currently {buffer.n_games()} games)")

    ramp = LevelRamp(args.start_level, window=args.ramp_window,
                     promote_score=args.ramp_up_score, demote_score=args.ramp_down_score,
                     min_level=1, max_level=8)

    rng = np.random.default_rng()
    tally = {"win": 0, "draw": 0, "loss": 0, "aborted": 0}
    levels_played: dict[int, int] = {}

    with tempfile.TemporaryDirectory() as tmp_profile:
        async with async_playwright() as pw:
            log.info("launching browser…")
            ctx = await pw.chromium.launch_persistent_context(
                user_data_dir=tmp_profile,
                headless=args.headless,
                viewport={"width": 1280, "height": 900},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                args=["--disable-blink-features=AutomationControlled"],
            )
            page = ctx.pages[0] if ctx.pages else await ctx.new_page()

            for i in range(1, args.n_games + 1):
                level = ramp.level
                color = "random" if args.color == "random" else args.color
                log.info(f"════════ game {i}/{args.n_games}  level={level}  color={color} ════════")
                levels_played[level] = levels_played.get(level, 0) + 1

                started = await lobby.start_stockfish_game(page, level=level, color=color)
                if not started:
                    log.error(f"could not start game {i}; skipping")
                    tally["aborted"] += 1
                    continue

                try:
                    states, policies, values, stats = await run_recorded_game(
                        page, net, device,
                        n_simulations=args.n_simulations, poll=args.poll,
                        temp_moves=args.temp_moves, max_moves=args.max_moves,
                        rng=rng,
                    )
                except Exception:
                    log.exception(f"game {i} crashed mid-play")
                    await save_screenshot(page, tag="farm-crash")
                    tally["aborted"] += 1
                    continue

                result = stats.get("result", "aborted")
                tally[result] = tally.get(result, 0) + 1
                ramp.record(result)

                if states.shape[0] > 0:
                    path = buffer.save_game(states, policies, values)
                    log.info(f"game {i}: {stats['outcome']}  "
                             f"({states.shape[0]} plies recorded)  saved={path.name}  "
                             f"buffer={buffer.n_games()} games")
                else:
                    log.warning(f"game {i}: no plies recorded ({stats.get('outcome')}); "
                                f"not saved to buffer")

                ramp.maybe_ramp()

            log.info(f"farm run complete: {tally}  levels_played={levels_played}  "
                     f"final_level={ramp.level}  buffer={buffer.n_games()} games")

            await ctx.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    ap.add_argument("--buffer", type=str, default="games")
    ap.add_argument("--n-games", type=int, default=20)
    ap.add_argument("--start-level", type=int, default=1,
                    help="initial Stockfish level (1-8); ramps automatically after this")
    ap.add_argument("--color", type=str, default="random", choices=["white", "black", "random"])
    ap.add_argument("--n-simulations", type=int, default=200)
    ap.add_argument("--temp-moves", type=int, default=30,
                    help="plies of temperature=1.0 sampling before switching to greedy")
    ap.add_argument("--max-moves", type=int, default=512)
    ap.add_argument("--ramp-window", type=int, default=10,
                    help="games at a level before evaluating a ramp up/down")
    ap.add_argument("--ramp-up-score", type=float, default=0.6,
                    help="avg score (win=1,draw=0.5,loss=0) to level up")
    ap.add_argument("--ramp-down-score", type=float, default=0.2,
                    help="avg score to level down")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--poll", type=float, default=0.1)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    log_path = setup_logging("farm", args.log_level)
    log.info(f"args = {vars(args)}")
    if log_path:
        log.info(f"log file: {log_path}")
    log.info("Fair-play notice: this farm only plays the Lichess computer "
             "(Stockfish) via lobby.start_stockfish_game. Never point it at "
             "human games — that violates Lichess ToS.")

    asyncio.run(_farm_async(args))


if __name__ == "__main__":
    main()
