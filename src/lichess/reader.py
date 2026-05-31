"""Read the Lichess board state from the DOM via Playwright.

Strategy stack (most → least authoritative):
    Path 0: replay UCI moves from window.lichess.round.data.steps onto the
            initial FEN. Preserves castling rights / ep / move counters.
    Path 1: full FEN from window.lichess.round.data.steps[-1].fen
    Path 2: piece-placement FEN + inferred turn + inferred castling
    Path 3: DOM pixel-parsing of <piece> elements
"""
from __future__ import annotations

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


async def get_orientation(page: Page) -> chess.Color:
    """Detect which colour we are playing (board orientation)."""

    # Strategy 0 — Lichess JS API (player.color is always set and correct).
    try:
        result = await page.evaluate(
            """() => {
                try {
                    var r = window.lichess && window.lichess.round;
                    if (r && r.data && r.data.player && r.data.player.color)
                        return r.data.player.color;
                } catch(e) {}
                return null;
            }"""
        )
        if result in ("black", "white"):
            log.info(f"orientation: {result} (via window.lichess.round.data.player.color)")
            return chess.BLACK if result == "black" else chess.WHITE
        log.debug(f"orientation strategy 0 returned: {result!r}")
    except Exception as e:
        log.debug(f"orientation strategy 0 raised: {e}")

    # Strategy 1 — cg-container class attribute.
    try:
        cls = await page.get_attribute("cg-container", "class") or ""
        log.debug(f"orientation cg-container class = {cls!r}")
        if "orientation-black" in cls:
            log.info("orientation: black (via cg-container)")
            return chess.BLACK
        if "orientation-white" in cls:
            log.info("orientation: white (via cg-container)")
            return chess.WHITE
    except Exception as e:
        log.debug(f"orientation strategy 1 raised: {e}")

    # Strategy 2 — average Y of pieces.
    try:
        result = await page.evaluate(
            r"""() => {
                var pieces = document.querySelectorAll('cg-board piece');
                var whiteY = 0, blackY = 0, wN = 0, bN = 0;
                for (var i = 0; i < pieces.length; i++) {
                    var p = pieces[i];
                    var cls = p.className.toLowerCase();
                    var raw = p.style.transform || p.style.cssText || '';
                    var m = raw.match(/translate\(\s*[\d.]+px,\s*([\d.]+)px\s*\)/);
                    if (!m) continue;
                    var y = parseFloat(m[1]);
                    if (cls.indexOf('white') !== -1) { whiteY += y; wN++; }
                    if (cls.indexOf('black') !== -1) { blackY += y; bN++; }
                }
                if (wN === 0 || bN === 0) return null;
                return (whiteY / wN) > (blackY / bN) ? 'white' : 'black';
            }"""
        )
        if result in ("black", "white"):
            log.info(f"orientation: {result} (via piece Y heuristic)")
            return chess.BLACK if result == "black" else chess.WHITE
    except Exception as e:
        log.debug(f"orientation strategy 2 raised: {e}")

    log.warning("orientation: all strategies failed — defaulting to WHITE")
    return chess.WHITE


async def _detect_turn_fallback(page: Page) -> chess.Color:
    try:
        result = await page.evaluate(
            """() => {
                var clocks = document.querySelectorAll('.rclock');
                for (var i = 0; i < clocks.length; i++) {
                    var c = clocks[i];
                    if (c.classList.contains('running') || c.classList.contains('rclock-turn')) {
                        if (c.classList.contains('rclock-bottom')) return 'bottom';
                        if (c.classList.contains('rclock-top'))    return 'top';
                    }
                }
                var txt = document.querySelector('.rclock-turn__text');
                if (txt) return txt.textContent.toLowerCase().indexOf('your') !== -1 ? 'ours' : 'theirs';
                return null;
            }"""
        )
        orientation = await get_orientation(page)
        if result in ("bottom", "ours"):
            log.debug(f"turn fallback: {result} → our colour")
            return orientation
        if result in ("top", "theirs"):
            log.debug(f"turn fallback: {result} → opponent colour")
            return chess.BLACK if orientation == chess.WHITE else chess.WHITE
        log.debug(f"turn fallback returned {result!r}")
    except Exception as e:
        log.debug(f"turn fallback raised: {e}")
    return chess.WHITE


async def _get_game_state(page: Page):
    try:
        state = await page.evaluate(
            """() => {
                try {
                    var r = window.lichess && window.lichess.round;
                    if (!r || !r.data) return null;
                    var game = r.data.game, steps = r.data.steps;
                    var turns = (game && game.turns != null) ? game.turns : null;
                    var uci_moves = [];
                    if (steps) {
                        for (var i = 1; i < steps.length; i++) {
                            if (steps[i].uci) uci_moves.push(steps[i].uci);
                        }
                    }
                    var initial_fen = (game && game.initialFen) ? game.initialFen : null;
                    var raw_fen = null;
                    if (steps && steps.length > 0) {
                        var last = steps[steps.length - 1];
                        if (last && last.fen) raw_fen = last.fen;
                    }
                    if (!raw_fen && game && game.fen) raw_fen = game.fen;
                    return {
                        initial_fen: initial_fen,
                        uci_moves: uci_moves,
                        turns: turns,
                        raw_fen: raw_fen
                    };
                } catch(e) { return null; }
            }"""
        )
        if state:
            log.debug(f"game state: initial_fen={state.get('initial_fen')!r}, "
                      f"n_uci={len(state.get('uci_moves') or [])}, turns={state.get('turns')}, "
                      f"raw_fen={state.get('raw_fen')!r}")
        else:
            log.debug("game state: window.lichess.round.data unavailable")
        return state
    except Exception as e:
        log.debug(f"game state raised: {e}")
        return None


async def _parse_from_dom(page: Page) -> chess.Board:
    orientation = await get_orientation(page)
    try:
        rect = await page.eval_on_selector("cg-board", "el => el.getBoundingClientRect()")
        sq_size = rect["width"] / 8.0
        log.debug(f"DOM parse: board rect={rect}, sq={sq_size:.1f}")
    except Exception as e:
        log.debug(f"DOM parse rect failed: {e}; using 64px squares")
        sq_size = 64.0

    pieces_data = await page.evaluate(
        """() =>
            Array.from(document.querySelectorAll('cg-board piece')).map(function(p) {
                return { cls: p.className, style: p.style.transform || p.getAttribute('style') || '' };
            })
        """
    )
    log.debug(f"DOM parse: {len(pieces_data)} piece elements found")

    board = chess.Board(fen=None)
    board.clear()
    for p in pieces_data:
        classes    = p["cls"].lower().split()
        color      = next((_COLOR[c]      for c in classes if c in _COLOR),      None)
        piece_type = next((_PIECE_TYPE[c] for c in classes if c in _PIECE_TYPE), None)
        if color is None or piece_type is None:
            continue
        m = re.search(r"translate\(([0-9.]+)px,\s*([0-9.]+)px\)", p["style"])
        if not m:
            continue
        col_idx = round(float(m.group(1)) / sq_size)
        row_idx = round(float(m.group(2)) / sq_size)
        if not (0 <= col_idx <= 7 and 0 <= row_idx <= 7):
            continue
        if orientation == chess.WHITE:
            file_ = col_idx; rank_ = 7 - row_idx
        else:
            file_ = 7 - col_idx; rank_ = row_idx
        board.set_piece_at(chess.square(file_, rank_), chess.Piece(piece_type, color))

    castling = ""
    if board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE):
        if board.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE): castling += "K"
        if board.piece_at(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE): castling += "Q"
    if board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK):
        if board.piece_at(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK): castling += "k"
        if board.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK): castling += "q"
    if castling:
        board.set_castling_fen(castling)

    return board


async def parse_board(page: Page) -> chess.Board:
    """Return a chess.Board reflecting the current Lichess position."""
    state = await _get_game_state(page)

    # Path 0 — UCI replay (most accurate; preserves castling/ep/halfmove).
    if state and state.get("uci_moves") is not None:
        uci_moves   = state["uci_moves"]
        initial_fen = state.get("initial_fen")
        try:
            if initial_fen and initial_fen not in ("startpos", ""):
                board = chess.Board(initial_fen) if " " in initial_fen \
                        else chess.Board(f"{initial_fen} w KQkq - 0 1")
            else:
                board = chess.Board()
            for uci in uci_moves:
                mv = chess.Move.from_uci(uci)
                if mv in board.legal_moves:
                    board.push(mv)
                else:
                    raise ValueError(f"illegal uci {uci} during replay at fen={board.fen()}")
            log.info(f"parse_board: UCI-replay path  plies={len(uci_moves)}  "
                     f"turn={'W' if board.turn else 'B'}  fen={board.fen()}")
            return board
        except Exception as e:
            log.warning(f"parse_board: UCI-replay failed ({e}); falling back")

    raw_fen = state.get("raw_fen") if state else None
    turns   = state.get("turns")   if state else None

    # Path 1 — full FEN.
    if raw_fen and " " in raw_fen:
        try:
            board = chess.Board(raw_fen)
            log.info(f"parse_board: full-FEN path  fen={board.fen()}")
            return board
        except Exception as e:
            log.debug(f"parse_board: full FEN parse failed: {e}")

    # Path 2 — piece-placement FEN + inferred turn / castling.
    if raw_fen:
        try:
            turn_char = "w" if (turns is None or turns % 2 == 0) else "b"
            board = chess.Board(f"{raw_fen} {turn_char} KQkq - 0 1")
            log.info(f"parse_board: piece-placement path  fen={board.fen()}")
            return board
        except Exception as e:
            log.debug(f"parse_board: piece-placement parse failed: {e}")

    # Path 3 — DOM pixel parsing.
    log.info("parse_board: DOM-pixel path")
    board = await _parse_from_dom(page)
    if turns is not None:
        board.turn = chess.WHITE if (turns % 2 == 0) else chess.BLACK
    else:
        board.turn = await _detect_turn_fallback(page)
    log.info(f"parse_board: DOM result  fen={board.fen()}")
    return board


async def our_color(page: Page) -> chess.Color:
    return await get_orientation(page)
