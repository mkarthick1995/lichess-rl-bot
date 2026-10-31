"""Execute chess moves on the Lichess board via Playwright clicks."""
from __future__ import annotations

import asyncio
import logging

import chess
from playwright.async_api import Page

from .reader import get_orientation

log = logging.getLogger("lichess.executor")


async def _board_rect(page: Page) -> dict:
    try:
        rect = await page.eval_on_selector("cg-board", "el => el.getBoundingClientRect()")
        log.debug(f"board rect: x={rect['x']:.1f}  y={rect['y']:.1f}  "
                  f"w={rect['width']:.1f}  h={rect['height']:.1f}")
        return rect
    except Exception as e:
        log.warning(f"board rect failed: {e}; falling back to (0,0,512,512)")
        return {"x": 0, "y": 0, "width": 512, "height": 512}


def _square_center(square: chess.Square, rect: dict, sq: float,
                   orientation: chess.Color) -> tuple[float, float]:
    file_ = chess.square_file(square)
    rank_ = chess.square_rank(square)
    col = file_ if orientation == chess.WHITE else 7 - file_
    row = (7 - rank_) if orientation == chess.WHITE else rank_
    x = rect["x"] + col * sq + sq / 2
    y = rect["y"] + row * sq + sq / 2
    return x, y


def _castling_rook_square(move: chess.Move) -> chess.Square:
    rank = chess.square_rank(move.from_square)
    rook_file = 7 if chess.square_file(move.to_square) > chess.square_file(move.from_square) else 0
    return chess.square(rook_file, rank)


async def execute_move(page: Page, move: chess.Move, board: chess.Board | None = None) -> bool:
    """Drag the piece from its square to the target (the rook square for castling).

    Returns True if the drag was issued without raising; the caller confirms
    that Lichess accepted the move (play._sync_after_our_move).
    """
    uci = move.uci()
    try:
        rect = await _board_rect(page)
        sq   = rect["width"] / 8
        orientation = await get_orientation(page)
        log.debug(f"execute_move start  uci={uci}  orientation={'W' if orientation else 'B'}  sq={sq:.1f}")

        fx, fy = _square_center(move.from_square, rect, sq, orientation)

        if board is not None:
            is_castling = board.is_castling(move)
        else:   # heuristic without a board: a king-like two-file hop along the back rank
            is_castling = (
                abs(chess.square_file(move.to_square) - chess.square_file(move.from_square)) == 2
                and move.from_square in (chess.E1, chess.E8)
                and chess.square_rank(move.to_square) == chess.square_rank(move.from_square)
            )
        target_square = _castling_rook_square(move) if is_castling else move.to_square
        tx, ty = _square_center(target_square, rect, sq, orientation)

        from_name = chess.square_name(move.from_square)
        to_name   = chess.square_name(move.to_square)
        tgt_name  = chess.square_name(target_square)
        log.info(f"dragging {from_name} → {tgt_name}"
                 f"{' (castle: clicking rook)' if is_castling else ''}"
                 f"  uci={uci}  from=({fx:.0f},{fy:.0f})  to=({tx:.0f},{ty:.0f})")

        # Drag rather than click-click: a click pair toggles chessground's
        # selection, so one stray click (e.g. during the opponent's
        # animation) leaves a piece selected and every later pair misfires.
        # A drag moves the piece regardless of the current selection.
        await page.mouse.move(fx, fy)
        await page.mouse.down()
        await page.mouse.move((fx + tx) / 2, (fy + ty) / 2, steps=4)
        await page.mouse.move(tx, ty, steps=4)
        await page.mouse.up()
        await asyncio.sleep(0.08)

        if move.promotion:
            await _handle_promotion(page, move.promotion)

        log.debug(f"execute_move done  uci={uci}")
        return True
    except Exception as e:
        log.exception(f"execute_move failed for {uci}: {e}")
        return False


async def _handle_promotion(page: Page, piece: int):
    names = {chess.QUEEN: "queen", chess.ROOK: "rook",
             chess.BISHOP: "bishop", chess.KNIGHT: "knight"}
    name = names.get(piece, "queen")
    log.info(f"promotion dialog: selecting {name}")
    await asyncio.sleep(0.2)

    # Strategy 0 — Lichess's picker: <div id="promotion-choice"> holding
    # <square><piece class="queen white"></square> per option.
    try:
        sel = f"#promotion-choice piece.{name}"
        await page.wait_for_selector(sel, timeout=500)
        await page.click(sel, timeout=500)
        log.info(f"promotion: clicked {sel}")
        return
    except Exception as e:
        log.debug(f"promotion #promotion-choice strategy failed: {e}")

    # Strategy 1 — JS scan inside promotion-choice container.
    try:
        clicked = await page.evaluate(
            f"""() => {{
                var containers = document.querySelectorAll(
                    '#promotion-choice, .promotion-choice, promotion-choice, .promotion'
                );
                for (var c = 0; c < containers.length; c++) {{
                    var els = containers[c].querySelectorAll('*');
                    for (var i = 0; i < els.length; i++) {{
                        var cls = (els[i].className || '').toLowerCase();
                        if (cls.indexOf('{name}') !== -1) {{
                            els[i].click();
                            return 'found:{name}';
                        }}
                    }}
                }}
                return null;
            }}"""
        )
        if clicked:
            log.info(f"promotion JS-click result: {clicked}")
            return
    except Exception as e:
        log.debug(f"promotion JS-click raised: {e}")

    # Strategy 2 — CSS selector fallback.
    selectors = [
        f".promotion-choice square piece.{name}",
        f".promotion-choice piece.{name}",
        f".promotion-choice .{name}",
        f".promotion piece.{name}",
        f"promotion-choice .{name}",
    ]
    for sel in selectors:
        try:
            await page.wait_for_selector(sel, timeout=800)
            await page.click(sel)
            log.info(f"promotion CSS-click: {sel}")
            return
        except Exception:
            continue
    log.warning(f"promotion: could not find dialog for {name}")
