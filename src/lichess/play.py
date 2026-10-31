"""Lichess auto-play loop using the trained network + MCTS as the move source.

USE AGAINST COMPUTER OPPONENTS ONLY. Playing against humans on Lichess violates
their Terms of Service.

Design: the bot tracks the authoritative game itself as a chess.Board, starting
from the initial FEN in page-init-data. Only piece *placement* is scraped from
the DOM; the opponent's move is identified by finding the legal move whose
resulting placement matches what appeared on the board. This avoids depending on
any fragile turn / move-list heuristic.

Usage:
    python -m src.lichess.play --level 3              # auto-start a game vs Lichess Stockfish level 3
    python -m src.lichess.play https://lichess.org/<game id>
    python -m src.lichess.play                        # opens lichess.org; you start a game
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import chess
from playwright.async_api import async_playwright

from .. import encoding, mcts
from .._log import setup_logging
from ..network import ChessNet
from . import executor, lobby, reader
from ._debug import dump_lichess_state, save_screenshot

log = logging.getLogger("lichess.play")


async def _game_over(page) -> bool:
    """Lichess removes the `playing` class from <body> when the game ends."""
    try:
        playing = await page.evaluate("() => document.body.classList.contains('playing')")
        if playing is False:
            log.info("game over (body lost 'playing' class)")
            return True
    except Exception as e:
        log.debug(f"_game_over check raised: {e}")
    return False


async def _engine_move(net, device, board: chess.Board, n_simulations: int) -> chess.Move | None:
    t0 = time.monotonic()
    root = mcts.run_mcts(board, net, device, n_simulations=n_simulations, add_noise=False)
    if not root.children:
        log.error("MCTS produced no children — no legal moves?")
        return None
    move = max(root.children.items(), key=lambda kv: kv[1].visit_count)[0]
    dt = time.monotonic() - t0
    top = sorted(root.children.items(), key=lambda kv: kv[1].visit_count, reverse=True)[:5]
    top_str = ", ".join(f"{board.san(m)}:{c.visit_count}" for m, c in top)
    log.info(f"engine move = {board.san(move)}  uci={move.uci()}  v={root.value:+.2f}  "
             f"top=[{top_str}]  mcts={dt:.2f}s")
    return move


async def _resync(page, why: str) -> chess.Board | None:
    """Authoritative game state from the server (re-fetched page-init-data)."""
    data = await reader.read_init_data(page)
    if not data:
        log.warning(f"resync ({why}) failed: no game data")
        return None
    board = reader.board_from_init_data(data)
    log.info(f"resynced from server ({why}): ply {board.ply()}  fen={board.fen()}")
    return board


def _match_after(board: chess.Board, dom_set: frozenset,
                 last_sq: frozenset = frozenset()) -> chess.Move | None:
    """The legal move of `board` whose resulting placement is `dom_set` and
    which agrees with the last-move highlight squares (when known)."""
    for mv in board.legal_moves:
        if last_sq and not (mv.from_square in last_sq
                            and (mv.to_square in last_sq or board.is_castling(mv))):
            continue
        board.push(mv)
        match = reader.placement_set(board) == dom_set
        board.pop()
        if match:
            return mv
    return None


async def _observe(page, orientation) -> tuple[frozenset, frozenset]:
    dom = await reader.read_placement(page, orientation)
    return reader.placement_set(dom), await reader.read_last_move(page, orientation)


async def _await_opponent_move(page, board: chess.Board, orientation, poll: float,
                               timeout: float = 600.0, resync_after: float = 4.0,
                               ) -> chess.Board | None:
    """Poll the DOM until the placement shows the opponent's move. Returns the
    new board (move pushed), or None if the game ended. If the DOM matches
    neither the current position nor any single legal move for
    `resync_after` seconds (we lost track), rebuild the board from the server."""
    base = reader.placement_set(board)
    t0 = time.monotonic()
    t_mismatch = None
    candidate = None
    while True:
        if await _game_over(page):
            return None
        if time.monotonic() - t0 > timeout:
            log.warning(f"opponent move wait timed out after {timeout:.0f}s")
            return None

        dom_set, last_sq = await _observe(page, orientation)
        if dom_set == base:
            t_mismatch = None
            candidate = None
        else:
            mv = _match_after(board, dom_set, last_sq)
            if mv is not None and mv == candidate:      # same answer on two reads: settled
                log.info(f"opponent played: {board.san(mv)}  uci={mv.uci()}")
                nb = board.copy()
                nb.push(mv)
                return nb
            candidate = mv
            if mv is not None:
                await asyncio.sleep(poll)
                continue
            # Mid-animation frame, or we are out of sync with the game.
            t_mismatch = t_mismatch or time.monotonic()
            if time.monotonic() - t_mismatch > resync_after:
                nb = await _resync(page, "opponent move not recognised")
                if nb is not None and nb.move_stack != board.move_stack:
                    return nb
                t_mismatch = None
        await asyncio.sleep(poll)


async def _sync_after_our_move(page, board: chess.Board, move: chess.Move, orientation,
                               poll: float, timeout: float = 8.0) -> tuple[chess.Board, bool]:
    """After clicking `move`, wait for the DOM to show it — possibly with the
    opponent's reply already on top (Lichess's AI answers within a poll
    interval). Returns (new board, whether our move landed). Falls back to the
    server's move list if the DOM never matches."""
    expected = board.copy()
    expected.push(move)
    want = reader.placement_set(expected)
    t0 = time.monotonic()
    candidate = None
    while time.monotonic() - t0 < timeout:
        dom_set, last_sq = await _observe(page, orientation)
        if dom_set == want:
            return expected, True
        reply = _match_after(expected, dom_set, last_sq)
        if reply is not None and reply == candidate:    # settled on two reads
            log.info(f"opponent replied instantly: {expected.san(reply)}  uci={reply.uci()}")
            expected.push(reply)
            return expected, True
        candidate = reply
        if await _game_over(page):
            break
        await asyncio.sleep(poll)
    nb = await _resync(page, f"our move {move.uci()} not seen on board")
    if nb is None:
        return board, False
    n = len(board.move_stack)
    landed = len(nb.move_stack) > n and nb.move_stack[:n] == board.move_stack and nb.move_stack[n] == move
    return nb, landed


async def _final_result(page, board: chess.Board, our: chess.Color) -> tuple[str, str, chess.Board]:
    """(result win/draw/loss/aborted, description, final board) — from the
    server when possible, since the game can end on a move we never saw or
    by a rule our board can't know (resignation, timeout)."""
    data = await reader.read_init_data(page)
    if data:
        board = reader.board_from_init_data(data)
        game = data.get("game") or {}
        status = (game.get("status") or {}).get("name", "?")
        winner = game.get("winner")
        if status not in ("created", "started"):
            if winner in ("white", "black"):
                won = (winner == "white") == (our == chess.WHITE)
                return ("win" if won else "loss"), f"{'win' if won else 'loss'} ({status})", board
            return "draw", f"draw ({status})", board
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "aborted", "incomplete (move cap or disconnect)", board
    if outcome.winner is None:
        return "draw", f"draw ({outcome.termination.name})", board
    won = outcome.winner == our
    return ("win" if won else "loss"), f"{'win' if won else 'loss'} ({outcome.termination.name})", board


async def _engine_move_recorded(net, device, board: chess.Board, n_simulations: int,
                                temperature: float, rng: np.random.Generator,
                                c_puct: float = 1.5,
                                ) -> tuple[chess.Move | None, np.ndarray, np.ndarray]:
    """Like _engine_move, but also returns the encoded state and the MCTS
    visit-count policy target (temperature=1.0, matching self_play), for
    recording into the replay buffer. Uses add_noise=True for exploration,
    same as self-play generation."""
    t0 = time.monotonic()
    root = mcts.run_mcts(board, net, device, n_simulations=n_simulations,
                         c_puct=c_puct, add_noise=True, rng=rng)
    if not root.children:
        log.error("MCTS produced no children — no legal moves?")
        return None, np.zeros(encoding.N_PLANES * 64, dtype=np.float32), np.zeros(encoding.N_MOVES, dtype=np.float32)

    state = encoding.encode_board(board)
    policy = np.zeros(encoding.N_MOVES, dtype=np.float32)
    visit_probs = mcts.visit_count_policy(root, temperature=1.0)
    for m, p in visit_probs.items():
        policy[encoding.move_to_index(m, board)] = p

    sample_probs = mcts.visit_count_policy(root, temperature=temperature)
    moves_list = list(sample_probs.keys())
    probs = np.array(list(sample_probs.values()))
    move = moves_list[rng.choice(len(moves_list), p=probs)]

    dt = time.monotonic() - t0
    top = sorted(root.children.items(), key=lambda kv: kv[1].visit_count, reverse=True)[:5]
    top_str = ", ".join(f"{board.san(m)}:{c.visit_count}" for m, c in top)
    log.info(f"engine move = {board.san(move)}  uci={move.uci()}  v={root.value:+.2f}  "
             f"top=[{top_str}]  mcts={dt:.2f}s  temp={temperature:.1f}")
    return move, state, policy


async def run_recorded_game(page, net, device, *, n_simulations: int, poll: float,
                            temp_moves: int = 30, max_moves: int = 512,
                            c_puct: float = 1.5,
                            rng: np.random.Generator | None = None,
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Play one already-loaded Lichess game to completion, recording our own
    MCTS-chosen moves as training data (state, policy target, backfilled
    value) exactly like self_play._play_one_game. Opponent moves are read
    from the DOM and pushed to the board but are not used as training
    targets. Returns (states, policies, values, stats); stats includes the
    game outcome so the caller can drive a win-rate-based difficulty ramp."""
    if rng is None:
        rng = np.random.default_rng()

    log.info("waiting 1.5s for Lichess to initialise…")
    await asyncio.sleep(1.5)

    data = await reader.read_init_data(page)
    if not data:
        log.error("could not read page-init-data — is a game loaded?")
        await save_screenshot(page, tag="no-init-data")
        return (np.zeros((0, encoding.N_PLANES, 8, 8), dtype=np.float32),
                np.zeros((0, encoding.N_MOVES), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                {"plies": 0, "outcome": "no game loaded", "result": "aborted"})

    color = await reader.our_color(page)
    side = "white" if color == chess.WHITE else "black"

    board = reader.board_from_init_data(data)

    log.info(f"[recorded] playing as {side}  start_fen={board.fen()}  n_sims={n_simulations}")
    await dump_lichess_state(page)

    states, policies, players = [], [], []
    move_num = 0
    t_game = time.monotonic()

    while True:
        if board.is_game_over(claim_draw=False) or await _game_over(page):
            log.info(f"[recorded] game finished. final board: {board.fen()}")
            break
        if move_num >= max_moves:
            log.warning(f"[recorded] hit max_moves={max_moves}; stopping")
            break

        if board.turn != color:
            nb = await _await_opponent_move(page, board, color, poll)
            if nb is None:
                break
            board = nb
            continue

        move_num += 1
        temperature = 1.0 if move_num <= temp_moves else 0.0
        log.info(f"── [recorded] our move {move_num} ({side}) ──  fen={board.fen()}")

        best, state, policy = await _engine_move_recorded(
            net, device, board, n_simulations, temperature, rng, c_puct=c_puct)
        if best is None:
            log.error("no engine move; stopping")
            break

        if not await executor.execute_move(page, best, board):
            log.warning("execute_move returned False; screenshot + retry next loop")
            await save_screenshot(page, tag="execute-fail")
            move_num -= 1
            continue

        nb, landed = await _sync_after_our_move(page, board, best, color, poll)
        if landed:
            states.append(state)
            policies.append(policy)
            players.append(board.turn)
        else:
            log.warning(f"our move {best.uci()} did not land; re-planning from the synced position")
            await save_screenshot(page, tag="move-unconfirmed")
            move_num -= 1
        board = nb

    dt_game = time.monotonic() - t_game
    result, outcome_str, board = await _final_result(page, board, color)
    values = np.zeros(len(players), dtype=np.float32)
    if result in ("win", "loss"):
        winner = color if result == "win" else (not color)
        values[:] = [1.0 if p == winner else -1.0 for p in players]
    stats = {
        "plies": len(states),
        "duration_s": dt_game,
        "outcome": outcome_str,
        "result": result,
        "our_color": side,
    }
    log.info(f"[recorded] game over: {len(states)} recorded plies, {dt_game:.1f}s -- {outcome_str}")

    if not states:
        return (np.zeros((0, encoding.N_PLANES, 8, 8), dtype=np.float32),
                np.zeros((0, encoding.N_MOVES), dtype=np.float32),
                np.zeros(0, dtype=np.float32), stats)
    return np.stack(states), np.stack(policies), values, stats


async def run_loop(page, net, device, *, n_simulations: int, poll: float):
    log.info("waiting 1.5s for Lichess to initialise…")
    await asyncio.sleep(1.5)

    data = await reader.read_init_data(page)
    if not data:
        log.error("could not read page-init-data — is a game loaded?")
        await save_screenshot(page, tag="no-init-data")
        return

    color = await reader.our_color(page)
    side = "white" if color == chess.WHITE else "black"

    board = reader.board_from_init_data(data)

    log.info(f"playing as {side}  start_fen={board.fen()}  n_sims={n_simulations}")
    await dump_lichess_state(page)

    move_num = 0
    while True:
        if board.is_game_over(claim_draw=False) or await _game_over(page):
            log.info(f"game finished. final board: {board.fen()}")
            break

        if board.turn != color:
            # Opponent to move — watch the board for their move.
            nb = await _await_opponent_move(page, board, color, poll)
            if nb is None:
                break
            board = nb
            continue

        # Our move.
        move_num += 1
        log.info(f"── our move {move_num} ({side}) ──  fen={board.fen()}")
        best = await _engine_move(net, device, board, n_simulations)
        if best is None:
            log.error("no engine move; stopping")
            break

        if not await executor.execute_move(page, best, board):
            log.warning("execute_move returned False; screenshot + retry next loop")
            await save_screenshot(page, tag="execute-fail")
            continue

        board, landed = await _sync_after_our_move(page, board, best, color, poll)
        if not landed:
            log.warning(f"our move {best.uci()} did not land; re-planning from the synced position")
            await save_screenshot(page, tag="move-unconfirmed")

    result, outcome_str, board = await _final_result(page, board, color)
    log.info(f"game over: {outcome_str}  final fen={board.fen()}")
    log.info("game loop exited")


async def main_async():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="https://lichess.org")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    ap.add_argument("--level", type=int, default=None,
                    help="start a game vs Lichess's computer at this Stockfish level (1-8) "
                         "automatically, play it, and exit")
    ap.add_argument("--color", type=str, default="random", choices=["white", "black", "random"],
                    help="our side when --level is used")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks",   type=int, default=10)
    ap.add_argument("--n-simulations", type=int, default=800)
    ap.add_argument("--poll", type=float, default=0.1)
    ap.add_argument("--headless", action="store_true")
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

    net = ChessNet(channels=args.channels, blocks=args.blocks).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    log.info(f"network: {args.channels}ch × {args.blocks} blocks = {n_params/1e6:.2f}M params")
    if args.checkpoint and Path(args.checkpoint).exists():
        net.load_state_dict(torch.load(args.checkpoint, map_location=device))
        log.info(f"loaded checkpoint: {args.checkpoint}")
    else:
        log.warning(f"checkpoint missing ({args.checkpoint}); using random init — plays badly")
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

            if args.level is not None:
                if not await lobby.start_stockfish_game(page, level=args.level, color=args.color):
                    log.error("could not start a game vs the computer; see logs/screenshots/")
                    await ctx.close()
                    return
            else:
                log.info(f"opening {args.url}")
                await page.goto(args.url, wait_until="domcontentloaded")
                on_game = re.search(r"lichess\.org/[A-Za-z0-9]{8,12}(/|$)", args.url) is not None
                if "lichess.org" in args.url and not on_game:
                    print("\n👉  Navigate to a game in the browser window, then press Enter here.")
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

            if args.level is not None or args.headless:
                await ctx.close()
                return
            log.info("done; browser stays open — Ctrl+C to exit")
            await asyncio.sleep(float("inf"))


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
