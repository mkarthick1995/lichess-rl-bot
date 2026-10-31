"""Read the Lichess board state from the DOM via Playwright.

Modern Lichess (2024+) no longer exposes ``window.lichess.round.data``. Instead
the initial game state is embedded as JSON in a ``<script id="page-init-data">``
tag, and board pieces are positioned with ``style="top:Y%;left:X%"`` (older
builds used ``transform: translate(Xpx, Ypx)`` — we support both).

This module provides primitives:
    read_init_data(page)        -> the load-time game data dict (color, fen, …)
    get_orientation(page)       -> which colour is at the bottom of the board
    read_placement(page, orient)-> a chess.Board holding only piece placement
    placement_set(board)        -> hashable set of (square, piece) for diffing

The play loop tracks the authoritative game itself (starting FEN + the moves it
plays and observes), so turn / castling / en-passant never depend on scraping —
only piece *placement* is read from the DOM, and the opponent's move is inferred
by matching that placement against the legal moves.
"""
from __future__ import annotations

import json
import logging
import re

import chess
from playwright.async_api import Page

log = logging.getLogger("lichess.reader")

_PIECE_TYPE = {
    "pawn":   chess.PAWN,   "knight": chess.KNIGHT, "bishop": chess.BISHOP,
    "rook":   chess.ROOK,   "queen":  chess.QUEEN,  "king":   chess.KING,
}
_COLOR = {"white": chess.WHITE, "black": chess.BLACK}


_INIT_RE = re.compile(r'<script[^>]*id="page-init-data"[^>]*>(.*?)</script>', re.S)


async def read_init_data(page: Page) -> dict | None:
    """Parse the <script id="page-init-data"> JSON. Returns the inner `data`
    object (game / player / opponent / steps / possibleMoves) or None.

    Lichess's round JS consumes and removes that element on load (as of
    2026-10), so when it is gone we re-fetch the page HTML with the same
    cookies and parse it from there — which also yields the *current* move
    list rather than the load-time one."""
    try:
        raw = await page.evaluate(
            """() => {
                const el = document.getElementById('page-init-data');
                return el ? el.textContent : null;
            }"""
        )
        if not raw:
            log.debug("read_init_data: no #page-init-data element; re-fetching page HTML")
            html = await page.evaluate(
                "fetch(location.href, {credentials: 'include'}).then(r => r.text())")
            m = _INIT_RE.search(html or "")
            if not m:
                log.debug("read_init_data: page-init-data not in re-fetched HTML either")
                return None
            raw = m.group(1)
        obj = json.loads(raw)
        data = obj.get("data")
        if data:
            g = data.get("game", {})
            log.debug(f"read_init_data: id={g.get('id')} fen={g.get('fen')!r} "
                      f"turns={g.get('turns')} player={data.get('player', {}).get('color')}")
        return data
    except Exception as e:
        log.debug(f"read_init_data raised: {e}")
        return None


def board_from_init_data(data: dict) -> chess.Board:
    """The game so far as a chess.Board with full move history (so repetition
    detection works), replayed from data["steps"]; falls back to game.fen."""
    game = data.get("game") or {}
    steps = data.get("steps") or []
    try:
        board = chess.Board(game.get("initialFen") or chess.STARTING_FEN)
        for st in steps[1:]:
            board.push_uci(st["uci"])
        return board
    except Exception as e:
        log.warning(f"could not replay steps ({e}); using game.fen")
        try:
            return chess.Board(game.get("fen") or chess.STARTING_FEN)
        except ValueError:
            return chess.Board()


async def get_orientation(page: Page) -> chess.Color:
    """Detect which colour is at the bottom (our colour when playing)."""

    # Strategy 1 — cg-wrap / cg-container orientation class (cheap; checked
    # first because the executor calls this on every move).
    try:
        for sel in ("cg-container", ".cg-wrap"):
            cls = await page.get_attribute(sel, "class") or ""
            if "orientation-black" in cls:
                log.debug(f"orientation: black (via {sel})")
                return chess.BLACK
            if "orientation-white" in cls:
                log.debug(f"orientation: white (via {sel})")
                return chess.WHITE
    except Exception as e:
        log.debug(f"orientation class strategy raised: {e}")

    # Strategy 0 — page-init-data player.color (may re-fetch the page).
    data = await read_init_data(page)
    if data:
        color = (data.get("player") or {}).get("color")
        if color in ("white", "black"):
            log.info(f"orientation: {color} (via page-init-data)")
            return _COLOR[color]

    # Strategy 2 — average vertical position of pieces (works for either
    # style format because we read the numeric top value).
    try:
        result = await page.evaluate(
            r"""() => {
                var pieces = document.querySelectorAll('cg-board piece');
                var whiteY = 0, blackY = 0, wN = 0, bN = 0;
                for (var i = 0; i < pieces.length; i++) {
                    var p = pieces[i];
                    var cls = p.className.toLowerCase();
                    var s = p.getAttribute('style') || '';
                    var y = null;
                    var mt = s.match(/top:\s*([\d.]+)%/);
                    if (mt) { y = parseFloat(mt[1]); }
                    else {
                        var tr = (p.style.transform || s).match(/translate\(\s*[\d.]+px,\s*([\d.]+)px/);
                        if (tr) y = parseFloat(tr[1]);
                    }
                    if (y === null) continue;
                    if (cls.indexOf('white') !== -1) { whiteY += y; wN++; }
                    if (cls.indexOf('black') !== -1) { blackY += y; bN++; }
                }
                if (wN === 0 || bN === 0) return null;
                return (whiteY / wN) > (blackY / bN) ? 'white' : 'black';
            }"""
        )
        if result in ("white", "black"):
            log.info(f"orientation: {result} (via piece Y heuristic)")
            return _COLOR[result]
    except Exception as e:
        log.debug(f"orientation Y strategy raised: {e}")

    log.warning("orientation: all strategies failed — defaulting to WHITE")
    return chess.WHITE


async def _read_pieces(page: Page) -> list[dict]:
    """Return [{cls, style}] for every piece element on the board."""
    return await page.evaluate(
        """() =>
            Array.from(document.querySelectorAll('cg-board piece')).map(function(p) {
                return { cls: p.className, style: p.getAttribute('style') || (p.style.cssText || '') };
            })
        """
    )


def _piece_col_row(style: str, sq_px: float | None) -> tuple[int, int] | None:
    """Extract (col, row) board indices (0..7, from the top-left of the board as
    displayed) from a piece's style, supporting both % and px formats."""
    ml = re.search(r"left:\s*([\d.]+)%", style)
    mt = re.search(r"top:\s*([\d.]+)%", style)
    if ml and mt:
        col = round(float(ml.group(1)) / 12.5)
        row = round(float(mt.group(1)) / 12.5)
        return col, row
    m = re.search(r"translate\(([0-9.]+)px,\s*([0-9.]+)px\)", style)
    if m and sq_px:
        col = round(float(m.group(1)) / sq_px)
        row = round(float(m.group(2)) / sq_px)
        return col, row
    return None


async def read_placement(page: Page, orientation: chess.Color) -> chess.Board:
    """Build a chess.Board holding ONLY the piece placement currently shown on
    the board (turn/castling/ep are not meaningful here — the caller tracks
    those). Supports both the percentage and pixel style formats."""
    sq_px = None
    try:
        rect = await page.eval_on_selector("cg-board", "el => el.getBoundingClientRect()")
        sq_px = rect["width"] / 8.0
    except Exception:
        pass

    pieces = await _read_pieces(page)

    board = chess.Board(fen=None)
    board.clear()
    parsed = 0
    for p in pieces:
        classes    = p["cls"].lower().split()
        if "ghost" in classes or "fading" in classes:   # drag shadow / captured piece fading out
            continue
        color      = next((_COLOR[c]      for c in classes if c in _COLOR),      None)
        piece_type = next((_PIECE_TYPE[c] for c in classes if c in _PIECE_TYPE), None)
        if color is None or piece_type is None:
            continue
        cr = _piece_col_row(p["style"], sq_px)
        if cr is None:
            continue
        col, row = cr
        if not (0 <= col <= 7 and 0 <= row <= 7):
            continue
        if orientation == chess.WHITE:
            file_, rank_ = col, 7 - row
        else:
            file_, rank_ = 7 - col, row
        board.set_piece_at(chess.square(file_, rank_), chess.Piece(piece_type, color))
        parsed += 1

    log.debug(f"read_placement: {len(pieces)} elements, {parsed} placed, "
              f"orientation={'W' if orientation == chess.WHITE else 'B'}")
    return board


async def read_last_move(page: Page, orientation: chess.Color) -> frozenset:
    """Squares chessground highlights as the last move (`square.last-move`).
    They are set to the true from/to squares as soon as a move is made, while
    piece positions are still animating — so they disambiguate mid-animation
    frames (e.g. a pawn passing d3 on its way to d4)."""
    try:
        styles = await page.evaluate(
            """() => Array.from(document.querySelectorAll('cg-board square.last-move'))
                         .map(s => s.getAttribute('style') || s.style.cssText || '')""")
        rect = await page.eval_on_selector("cg-board", "el => el.getBoundingClientRect()")
        sq_px = rect["width"] / 8.0
    except Exception as e:
        log.debug(f"read_last_move raised: {e}")
        return frozenset()
    out = set()
    for st in styles:
        cr = _piece_col_row(st, sq_px)
        if cr and 0 <= cr[0] <= 7 and 0 <= cr[1] <= 7:
            col, row = cr
            out.add(chess.square(col, 7 - row) if orientation == chess.WHITE
                    else chess.square(7 - col, row))
    return frozenset(out)


def placement_set(board: chess.Board) -> frozenset:
    """A hashable snapshot of piece placement, for comparing two positions."""
    return frozenset(
        (sq, board.piece_at(sq).piece_type, board.piece_at(sq).color)
        for sq in chess.SQUARES if board.piece_at(sq) is not None
    )


async def our_color(page: Page) -> chess.Color:
    return await get_orientation(page)
