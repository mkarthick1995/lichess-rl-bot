"""Generation-2 self-play: many concurrent games, Gumbel root search, and
playout-cap randomisation.

Each move is either a *full* search (probability p_full, n_full simulations,
recorded as a training position) or a *fast* search (n_fast simulations,
not recorded) — KataGo's playout-cap randomisation: most moves only need to
be good enough to continue the game, so the compute goes into the positions
we actually learn from. Every recorded position gets three targets:
    policy   the Gumbel improved policy (gumbel.improved_policy)
    result   the game result from the side to move
    plies    plies until the game ended (moves-left head)
Clocks are off in self-play (the network's "clock known" flag is 0); clock
knowledge comes from the human games mixed into training (loop2.py).

Games are never resigned (the bot never resigns either); games reaching
max_plies are drawn, with unknown moves-left.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import chess
import numpy as np

from . import encoding, gumbel, mcts
from .replay2 import POLICY_K
from .self_play import outcome_str


@dataclass
class GumbelConfig:
    n_full: int = 128
    m_full: int = 16
    n_fast: int = 24
    m_fast: int = 8
    p_full: float = 0.25
    noise_plies: int = 30          # Gumbel noise (move sampling) for this many plies
    c_puct: float = 1.5            # below the root
    fpu_reduction: float = 0.2
    max_plies: int = 400


@dataclass
class _Game:
    board: chess.Board = field(default_factory=chess.Board)
    root: mcts.Node = field(default_factory=mcts.Node)
    g: Optional[np.ndarray] = None          # Gumbel noise for this move (None = not set up)
    logits: Optional[np.ndarray] = None
    n: int = 0
    m: int = 0
    full: bool = False
    states: list = field(default_factory=list)
    pidx: list = field(default_factory=list)
    pprob: list = field(default_factory=list)
    stm: list = field(default_factory=list)
    ply: list = field(default_factory=list)
    t0: float = field(default_factory=time.monotonic)
    sims: int = 0


def sparse(idx: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(idx) > POLICY_K:
        keep = np.argsort(-probs)[:POLICY_K]
        idx, probs = idx[keep], probs[keep] / probs[keep].sum()
    out_i = np.zeros(POLICY_K, dtype=np.int16)
    out_p = np.zeros(POLICY_K, dtype=np.float16)
    out_i[:len(idx)] = idx
    out_p[:len(idx)] = probs
    return out_i, out_p


class GumbelSelfPlay:
    """Plays `parallel` games at a time with one network evaluator."""

    def __init__(self, evaluator, cfg: GumbelConfig, parallel: int,
                 rng: Optional[np.random.Generator] = None):
        self.ev, self.cfg, self.parallel = evaluator, cfg, parallel
        self.rng = rng or np.random.default_rng()
        self.games: list[_Game] = []
        self.evals = 0

    def step(self) -> list[dict]:
        """One batched round (≤ 1 leaf per game). Returns finished game records."""
        while len(self.games) < self.parallel:
            self.games.append(_Game())
        cfg, finished, batch = self.cfg, [], []
        for g in list(self.games):
            root = g.root
            if root.expanded:
                if g.g is None:
                    self._setup_move(g)
                if root.N.sum() >= g.n or len(root.moves) == 1:
                    rec = self._play(g)
                    if rec is not None:
                        finished.append(rec)
                        self.games.remove(g)
                    continue
                a = gumbel.root_action(root, g.g, g.logits, g.m, g.n)
                p = mcts.gather_leaf(root, g.board, cfg.c_puct, cfg.fpu_reduction, 0.0, root_action=a)
            else:
                p = mcts.gather_leaf(root, g.board, cfg.c_puct, cfg.fpu_reduction)
            g.sims += 1
            if isinstance(p, mcts.Pending):
                batch.append(p)
        if batch:
            priors, values = self.ev([p.planes for p in batch], [p.idx for p in batch])
            for p, pr, v in zip(batch, priors, values):
                mcts.complete_leaf(p, pr, float(v))
            self.evals += len(batch)
        return finished

    def _setup_move(self, g: _Game) -> None:
        cfg, k = self.cfg, len(g.root.moves)
        g.full = self.rng.random() < cfg.p_full
        g.n, g.m = (cfg.n_full, cfg.m_full) if g.full else (cfg.n_fast, cfg.m_fast)
        g.logits = gumbel.logits_of(g.root)
        g.g = (self.rng.gumbel(size=k) if g.board.ply() < cfg.noise_plies else np.zeros(k))

    def _play(self, g: _Game) -> Optional[dict]:
        root, board = g.root, g.board
        if len(root.moves) == 1:
            a = 0
        else:
            a = gumbel.final_action(root, g.g, g.logits)
            if g.full:
                _, idx = encoding.legal_moves_and_indices(board)     # same order as root.moves
                i, p = sparse(idx, gumbel.improved_policy(root, g.logits))
                g.states.append(encoding.encode_board_u8(board))
                g.pidx.append(i); g.pprob.append(p)
                g.stm.append(board.turn); g.ply.append(board.ply())
        board.push(root.moves[a])
        g.root, g.g, g.logits = mcts.Node(), None, None     # fresh tree: the visit table assumes it
        capped = board.ply() >= self.cfg.max_plies
        if not capped and mcts.terminal_value(board, list(board.legal_moves)) is None:
            return None
        return self._finish(g, capped)

    def _finish(self, g: _Game, capped: bool) -> dict:
        desc, result = outcome_str(g.board, capped)
        plies = g.board.ply()
        rec = {"outcome": desc, "result": result, "plies": plies,
               "seconds": time.monotonic() - g.t0, "sims": g.sims}
        if g.states:
            stm = np.array(g.stm, dtype=bool)
            res = np.zeros(len(stm), dtype=np.int8)
            if result != 0:
                res[:] = np.where(stm == (result > 0), 1, -1)
            left = (plies - np.array(g.ply)).astype(np.int16)
            if capped:
                left[:] = -1
            rec.update(states=np.stack(g.states), pidx=np.stack(g.pidx), pprob=np.stack(g.pprob),
                       pos_result=res, plies_left=left)
        return rec


def to_buffer_rec(rec: dict) -> dict:
    """Game record → ReplayBuffer2 fields (or {} when nothing was recorded)."""
    if "states" not in rec:
        return {}
    return {"states": rec["states"], "pidx": rec["pidx"], "pprob": rec["pprob"],
            "result": rec["pos_result"], "plies": rec["plies_left"]}


def run(evaluator, cfg: GumbelConfig, parallel: int, on_game: Callable[[dict], None],
        should_stop: Callable[[], bool] = lambda: False,
        between_steps: Callable[[], None] = lambda: None, n_games: int = 0) -> None:
    sp = GumbelSelfPlay(evaluator, cfg, parallel)
    done = 0
    while not should_stop() and (n_games == 0 or done < n_games):
        for rec in sp.step():
            on_game(rec)
            done += 1
        between_steps()
