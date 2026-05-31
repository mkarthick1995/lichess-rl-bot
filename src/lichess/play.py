"""Lichess auto-play loop using the trained network + MCTS as the move source.

USE AGAINST COMPUTER OPPONENTS ONLY. Playing against humans on Lichess violates
their Terms of Service.

Usage:
    python -m src.lichess.play
    python -m src.lichess.play https://lichess.org/abc123
    python -m src.lichess.play --checkpoint checkpoints/latest.pt
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import chess
from playwright.async_api import async_playwright

from .. import mcts
from .._log import setup_logging
from ..network import ChessNet
from . import reader, executor
from ._debug import dump_lichess_state, save_screenshot

log = logging.getLogger("lichess.play")

_LICHESS_OVER = {
    "mate", "resign", "stalemate", "timeout", "draw",
    "outoftime", "cheat", "aborted", "noStart", "unknownFinish", "variantEnd",
}


async def _lichess_game_over(page) -> bool:
    try:
        status = await page.evaluate(
            """() => {
                try {
                    var g = window.lichess && window.lichess.round &&
                            window.lichess.round.data && window.lichess.round.data.game;
                    return g ? g.status && g.status.name : null;
                } catch(e) { return null; }
            }"""
        )
        if status and status in _LICHESS_OVER:
            log.info(f"lichess reports game over (status={status})")
            return True
    except Exception as e:
        log.debug(f"lichess game-over check raised: {e}")
    return False


async def _wait_for_our_turn(page, color: chess.Color, last_fen: str | None, poll: float):
    """Poll until the position changes AND it is our turn."""
    while True:
        try:
            board = await reader.parse_board(page)
        except Exception as e:
            log.warning(f"poll parse_board raised: {e}")
            await asyncio.sleep(poll)
            continue

        if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
            log.info(f"local board reports terminal: {board.result()}")
            return None, ""
        if await _lichess_game_over(page):
            return None, ""

        fen = board.fen()
        if board.turn == color and fen != last_fen:
            return board, fen

        await asyncio.sleep(poll)


async def _engine_move(net, device, board: chess.Board, n_simulations: int) -> chess.Move | None:
    """Pick a move via MCTS-guided search."""
    t0 = time.monotonic()
    root = mcts.run_mcts(board, net, device,
                         n_simulations=n_simulations,
                         add_noise=False)   # play strength mode, no Dirichlet
    if not root.children:
        log.error("MCTS produced no children — no legal moves?")
        return None
    move = max(root.children.items(), key=lambda kv: kv[1].visit_count)[0]
    dt = time.monotonic() - t0

    top = sorted(root.children.items(), key=lambda kv: kv[1].visit_count, reverse=True)[:5]
    top_str = ", ".join(f"{board.san(m)}:{c.visit_count}" for m, c in top)
    log.info(f"engine move = {board.san(move)}  uci={move.uci()}  "
             f"v={root.value:+.2f}  top-by-visit=[{top_str}]  mcts={dt:.2f}s")
    return move


async def run_loop(page, net, device, *, n_simulations: int, poll: float):
    # Give Lichess time to populate window.lichess.round.data.
    log.info("waiting 1.5s for Lichess to initialise…")
    await asyncio.sleep(1.5)

    color = await reader.our_color(page)
    side  = "white" if color == chess.WHITE else "black"
    log.info(f"playing as {side}  n_sims={n_simulations}")
    await dump_lichess_state(page)

    # When playing black, seed last_fen with the starting position so we don't
    # premove on the empty board before white has moved.
    if color == chess.BLACK:
        try:
            last_fen: str | None = (await reader.parse_board(page)).fen()
            log.info(f"black start: last_fen seeded with {last_fen}")
        except Exception:
            last_fen = None
    else:
        last_fen = None

    move_num = 0

    while True:
        board, fen = await _wait_for_our_turn(page, color, last_fen, poll)
        if board is None:
            break
        move_num += 1

        if board.move_stack:
            last_mv = board.peek()
            board.pop()
            try:
                opp_san  = board.san(last_mv)
                opp_side = "black" if color == chess.WHITE else "white"
                log.info(f"opponent ({opp_side}) played: {opp_san}  uci={last_mv.uci()}")
            except Exception as e:
                log.warning(f"opp SAN failed: {e}  raw={last_mv.uci()}")
            board.push(last_mv)
        elif move_num > 1:
            log.warning(f"move_stack empty at move {move_num} — UCI-replay path may have fallen back")

        log.info(f"── our move {move_num} ({side}) ──  fen={fen}")
        best = await _engine_move(net, device, board, n_simulations)
        if best is None:
            log.error("no engine move; stopping")
            break

        ok = await executor.execute_move(page, best)
        if not ok:
            log.warning("execute_move returned False — saving screenshot and retrying next poll")
            await save_screenshot(page, tag="execute-fail")
            await dump_lichess_state(page)
            continue

        last_fen = fen
        await asyncio.sleep(0.15)

    log.info("game loop exited")


async def main_async():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="https://lichess.org")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="trained network weights; uses random init if omitted")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks",   type=int, default=10)
    ap.add_argument("--n-simulations", type=int, default=200,
                    help="MCTS simulations per move at play time")
    ap.add_argument("--poll", type=float, default=0.05,
                    help="seconds between board-state polls")
    ap.add_argument("--headless", action="store_true",
                    help="run browser hidden (you won't see the moves)")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    log_path = setup_logging("play", args.log_level)
    log.info(f"args = {vars(args)}")
    if log_path:
        log.info(f"log file: {log_path}")
    log.info("Fair-play notice: use only against the Lichess computer / friend "
             "challenges. Playing vs humans violates Lichess ToS.")

    device = torch.device(args.device)
    log.info(f"device = {device}")
    if device.type == "cuda":
        log.info(f"GPU = {torch.cuda.get_device_name(0)}")

    net = ChessNet(channels=args.channels, blocks=args.blocks).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    log.info(f"network: {args.channels}ch × {args.blocks} blocks = {n_params/1e6:.2f}M params")
    if args.checkpoint and Path(args.checkpoint).exists():
        net.load_state_dict(torch.load(args.checkpoint, map_location=device))
        log.info(f"loaded checkpoint: {args.checkpoint}")
    else:
        if args.checkpoint:
            log.warning(f"checkpoint not found at {args.checkpoint}; random init")
        else:
            log.warning("no checkpoint provided; using random init (network will play badly)")
    net.eval()

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

            log.info(f"opening {args.url}")
            await page.goto(args.url, wait_until="domcontentloaded")

            on_homepage = ("/game/" not in args.url and "/play" not in args.url
                           and "@" not in args.url)
            if "lichess.org" in args.url and on_homepage:
                print("\n👉  Navigate to a game in the browser window, then press Enter here.")
                # Use a thread-blocking input on the asyncio loop so we don't burn CPU.
                await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)

            try:
                await page.wait_for_selector("cg-board", timeout=30_000)
                log.info("cg-board element found")
            except Exception:
                log.error("cg-board not found within 30s — is a game loaded?")
                await save_screenshot(page, tag="no-board")
                await ctx.close()
                return

            try:
                await run_loop(page, net, device,
                               n_simulations=args.n_simulations, poll=args.poll)
            except Exception as e:
                log.exception(f"run_loop crashed: {e}")
                await save_screenshot(page, tag="crash")
                await dump_lichess_state(page)

            log.info("done; browser stays open — Ctrl+C to exit")
            await asyncio.sleep(float("inf"))


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
