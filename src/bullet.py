"""Clock-aware move selection for bullet chess (1+0, 30+0, 2+1 ...).

The network and tree search are clock-agnostic; this layer decides *how long*
to think and *which* of the searched moves to play given both clocks:

* TimeManager   — per-move thinking budget from our clock, the increment and
                  the move number; instant moves when in time trouble
                  ("panic") or when the opponent is about to flag ("flag").
* safety filter — cheap checks that a short search can miss and that matter
                  a lot in bullet: never allow mate-in-1, never stalemate the
                  opponent or walk into a repetition draw while we are better
                  or ahead on the clock, and keep mating material when we are
                  trying to win on time (a flag against a bare king is a draw).
* BulletEngine  — owns the search tree across moves (subtree reuse) and
                  ponders on the opponent's time, so the GPU is never idle.

The engine never resigns: in bullet a lost position with more time is often
a win on the clock.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import chess
import numpy as np

from .mcts import Node, search


@dataclass
class Clock:
    """Remaining time in milliseconds, from the side to move's point of view."""
    our_ms: int
    opp_ms: int
    inc_ms: int = 0
    opp_inc_ms: int = 0


@dataclass
class BulletConfig:
    max_sims: int = 6000          # hard cap per move (memory / sanity)
    ponder_sims: int = 20000      # cap while pondering
    leaf_batch: int = 16
    c_puct: float = 1.5
    fpu_reduction: float = 0.2
    overhead_ms: int = 120        # per-move cost outside the search (network, API)
    opening_moves: int = 8        # our first N moves are played quickly...
    opening_factor: float = 0.35  # ...at this fraction of the normal budget
    max_fraction: float = 0.12    # never spend more than this share of our clock
    low_ms: int = 15000           # below this, budget only 1/40 of the clock per move
    panic_ms: int = 2500          # below this (and no increment): instant moves
    flag_opp_ms: int = 8000       # opponent below this...
    flag_margin_ms: int = 2000    # ...and we have 2x theirs + this → flag mode
    flag_think_ms: int = 40       # think budget in flag mode
    ponder: bool = True


@dataclass
class MoveInfo:
    move: chess.Move
    mode: str                     # normal / opening / panic / flag / forced
    think_ms: int                 # budget given to the search
    spent_ms: float               # wall time actually spent in choose()
    sims: int                     # root visits (incl. reused + pondered)
    value: float                  # root value, our POV, in [-1, 1]
    reused: int = 0               # visits inherited from the previous tree
    filtered: list = field(default_factory=list)   # moves rejected by safety


class TimeManager:
    def __init__(self, cfg: BulletConfig):
        self.cfg = cfg

    def budget(self, clock: Clock, our_moves_played: int) -> tuple[int, str]:
        """(milliseconds to search, mode)."""
        c = self.cfg
        if clock.our_ms < c.panic_ms and clock.inc_ms < 1000:
            return 0, "panic"
        if (clock.opp_ms < c.flag_opp_ms
                and clock.our_ms > 2 * clock.opp_ms + c.flag_margin_ms):
            return c.flag_think_ms, "flag"
        # Bullet games often run long (scrambles, engine-like opponents that
        # never flag), so assume plenty of moves remain, and more so as the
        # clock runs down: below `low_ms` only 1/40 of the clock per move.
        moves_left = max(20.0, 45.0 - 0.5 * our_moves_played)
        if clock.our_ms < c.low_ms:
            moves_left = max(moves_left, 40.0)
        ms = clock.our_ms / moves_left + 0.75 * clock.inc_ms
        mode = "normal"
        if our_moves_played < c.opening_moves:
            ms *= c.opening_factor
            mode = "opening"
        ms = min(ms, c.max_fraction * clock.our_ms)
        return max(0, int(ms - c.overhead_ms)), mode


# ---------------------------------------------------------------------------
# Safety checks (all cheap: a handful of legal-move generations).

def allows_mate_in_one(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    try:
        for reply in board.legal_moves:
            board.push(reply)
            mate = board.is_checkmate()
            board.pop()
            if mate:
                return True
        return False
    finally:
        board.pop()


def find_mate_in_one(board: chess.Board) -> Optional[chess.Move]:
    """A move that checkmates right now, if any. The value head only says
    *that* we win, not how fast, so a short search happily shuffles in a won
    position while the clock runs down; this check costs a few ms."""
    for move in board.legal_moves:
        board.push(move)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return move
    return None


def _draw_after(board: chess.Board, move: chess.Move) -> bool:
    """Would this move end the game in a draw (stalemate, dead position,
    threefold — Lichess draws a threefold automatically)?"""
    board.push(move)
    try:
        return (board.is_stalemate() or board.is_insufficient_material()
                or board.is_repetition(3))
    finally:
        board.pop()


def _loses_mating_material(board: chess.Board, move: chess.Move) -> bool:
    us = board.turn
    board.push(move)
    try:
        return board.has_insufficient_material(us)
    finally:
        board.pop()


def pick_move(board: chess.Board, root: Node, want_decisive: bool,
              flagging: bool, near_best: float = 0.5) -> tuple[chess.Move, list]:
    """Choose among the searched root moves.

    Moves are ranked by visits (or by prior when the search was too short to
    visit anything). Hard rule: skip moves that allow mate-in-1 if any move
    avoids it. Soft rules, only among moves with at least `near_best` of the
    best move's score: when we want a decisive result avoid immediate draws,
    and when trying to flag keep our own mating material."""
    moves = list(root.moves)
    score = root.N.astype(np.float64) if root.N.sum() > 0 else root.priors.astype(np.float64)
    order = [int(i) for i in np.argsort(-score, kind="stable")]
    rejected = []

    # Checks run lazily in rank order, so the usual case (the top move passes)
    # costs one mate-in-1 scan.
    first_safe = None
    for i in order:
        m = moves[i]
        if first_safe is not None and score[i] < near_best * score[first_safe]:
            break                        # only near-best moves are worth trading for
        if allows_mate_in_one(board, m):
            rejected.append((m.uci(), "mate-in-1"))
            continue
        if first_safe is None:
            first_safe = i
        if want_decisive and _draw_after(board, m):
            rejected.append((m.uci(), "draw"))
            continue
        if flagging and _loses_mating_material(board, m):
            rejected.append((m.uci(), "no mating material"))
            continue
        return m, rejected
    if first_safe is not None:
        return moves[first_safe], rejected
    return moves[order[0]], rejected     # every move allows mate: play the best


# ---------------------------------------------------------------------------

class BulletEngine:
    """Stateful per-game player. Call choose() when it is our turn with the
    current board and clocks; between our moves it ponders automatically."""

    def __init__(self, evaluate, cfg: Optional[BulletConfig] = None):
        self.evaluate = evaluate
        self.cfg = cfg or BulletConfig()
        self.tm = TimeManager(self.cfg)
        self._root: Optional[Node] = None
        self._root_stack: list[chess.Move] = []   # move stack of _root's position
        self._ponder_thread: Optional[threading.Thread] = None
        self._ponder_stop = threading.Event()

    # -- tree reuse ---------------------------------------------------------
    def _reuse(self, board: chess.Board) -> Node:
        """Root for `board`: descend the stored tree by the moves played since
        (normally just the opponent's reply), else start fresh."""
        root = self._root
        stack = board.move_stack
        n = len(self._root_stack)
        if root is None or len(stack) < n or stack[:n] != self._root_stack:
            return Node()
        for mv in stack[n:]:
            if not root.expanded or root.terminal is not None:
                return Node()
            try:
                i = root.moves.index(mv)
            except ValueError:
                return Node()
            root = root.kids[i]
            if root is None:
                return Node()
        return root

    def _set_root(self, node: Optional[Node], board: Optional[chess.Board]) -> None:
        self._root = node
        self._root_stack = list(board.move_stack) if board is not None else []

    # -- pondering ------------------------------------------------------------
    def _ponder_loop(self, root: Node, board: chess.Board) -> None:
        try:
            search(root, board, self.evaluate, self.cfg.ponder_sims,
                   c_puct=self.cfg.c_puct, fpu_reduction=self.cfg.fpu_reduction,
                   leaf_batch=self.cfg.leaf_batch, should_stop=self._ponder_stop.is_set)
        except Exception:
            pass                          # pondering is best-effort

    def stop_ponder(self) -> None:
        if self._ponder_thread is not None:
            self._ponder_stop.set()
            self._ponder_thread.join()
            self._ponder_thread = None

    def _start_ponder(self, board_after: chess.Board) -> None:
        if not self.cfg.ponder or board_after.is_game_over(claim_draw=False):
            return
        root = self._reuse(board_after)
        self._set_root(root, board_after)
        self._ponder_stop = threading.Event()
        self._ponder_thread = threading.Thread(
            target=self._ponder_loop, args=(root, board_after.copy()), daemon=True)
        self._ponder_thread.start()

    # -- main entry point -----------------------------------------------------
    def choose(self, board: chess.Board, clock: Clock) -> MoveInfo:
        t0 = time.monotonic()
        self.stop_ponder()
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("no legal moves")

        mate = find_mate_in_one(board)
        if mate is not None:
            self._set_root(None, None)
            return MoveInfo(move=mate, mode="mate", think_ms=0,
                            spent_ms=(time.monotonic() - t0) * 1000.0, sims=0, value=1.0)

        root = self._reuse(board)
        reused = root.visit_count
        our_moves = len(board.move_stack) // 2
        think_ms, mode = self.tm.budget(clock, our_moves)
        if len(legal) == 1:
            think_ms, mode = 0, "forced"
        deadline = t0 + think_ms / 1000.0
        # With no budget the root is still expanded (one network call), so the
        # move comes from the policy head ("instant move").
        search(root, board, self.evaluate, self.cfg.max_sims if think_ms else 0,
               c_puct=self.cfg.c_puct, fpu_reduction=self.cfg.fpu_reduction,
               leaf_batch=self.cfg.leaf_batch, deadline=deadline)

        value = root.value
        flagging = mode == "flag" or clock.our_ms > clock.opp_ms + 3000
        want_decisive = value > 0.05 or flagging
        if len(legal) == 1:
            move, rejected = legal[0], []
        else:
            move, rejected = pick_move(board, root, want_decisive, flagging)

        info = MoveInfo(move=move, mode=mode, think_ms=think_ms,
                        spent_ms=(time.monotonic() - t0) * 1000.0,
                        sims=root.visit_count, value=value, reused=reused,
                        filtered=rejected)

        # Keep the subtree of our move and ponder on the opponent's time.
        after = board.copy()
        after.push(move)
        self._set_root(root, board)
        self._start_ponder(after)
        return info

    def close(self) -> None:
        self.stop_ponder()
        self._set_root(None, None)
