"""Batched game generation: self-play and arena matches.

`GameRunner` plays many games at once. Every step it pulls one leaf (or a
few, with virtual loss) from each game's search tree, scores all of them in
one network batch per player, then advances any game whose root has enough
visits. With ~100+ concurrent games the GPU sees large batches and Python
tree work dominates, which is what makes this ~50x faster than searching one
position at a time.

Standalone (writes to the replay buffer):
    python -m src.self_play --checkpoint checkpoints/best.pt --n-games 64
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import chess
import numpy as np
import torch

from . import encoding, mcts
from ._log import setup_logging
from .inference import BatchEvaluator
from .network import ChessNet
from .replay_buffer import POLICY_K, ReplayBuffer

log = logging.getLogger("self_play")


@dataclass
class SearchConfig:
    n_sims: int = 100
    c_puct: float = 1.5
    fpu_reduction: float = 0.2
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    leaves_per_game: int = 1          # >1 uses virtual loss within one tree


@dataclass
class GameSpec:
    """How one game is played. players maps colour → evaluator key."""
    players: dict
    add_noise: bool = True
    temp_plies: int = 30               # sample ∝ visits for this many plies, then argmax
    record: bool = True                # keep (state, policy) per ply for training
    max_plies: int = 512
    tag: object = None                 # caller data (e.g. which side is the candidate)


@dataclass
class _Game:
    spec: GameSpec
    board: chess.Board = field(default_factory=chess.Board)
    root: mcts.Node = field(default_factory=mcts.Node)
    noised: bool = False
    t0: float = field(default_factory=time.monotonic)
    states: list = field(default_factory=list)
    pidx: list = field(default_factory=list)
    pprob: list = field(default_factory=list)
    stm: list = field(default_factory=list)
    sims: int = 0


def outcome_str(board: chess.Board, capped: bool) -> tuple[str, int]:
    """(human description, result from white's POV: +1 / 0 / -1)."""
    if capped:
        return "draw (move cap)", 0
    moves = list(board.legal_moves)
    if not moves:
        if board.is_check():
            return ("white" if board.turn == chess.BLACK else "black") + " wins (checkmate)", \
                   (1 if board.turn == chess.BLACK else -1)
        return "draw (stalemate)", 0
    if board.is_insufficient_material():
        return "draw (insufficient material)", 0
    if board.halfmove_clock >= 100:
        return "draw (50-move rule)", 0
    return "draw (repetition)", 0


def sparse_policy(root: mcts.Node, board: chess.Board) -> tuple[np.ndarray, np.ndarray]:
    """Root visit distribution as fixed-width (POLICY_K) index/prob arrays."""
    _, idx = encoding.legal_moves_and_indices(board)   # same order as root.moves
    probs = mcts.visit_policy(root)
    if len(idx) > POLICY_K:
        keep = np.argsort(-probs)[:POLICY_K]
        idx, probs = idx[keep], probs[keep]
        probs = probs / probs.sum()
    out_i = np.zeros(POLICY_K, dtype=np.int16)
    out_p = np.zeros(POLICY_K, dtype=np.float16)
    out_i[:len(idx)] = idx
    out_p[:len(idx)] = probs
    return out_i, out_p


class GameRunner:
    """Drives many concurrent games against a dict of evaluators."""

    def __init__(self, evaluators: dict, cfg: SearchConfig,
                 rng: Optional[np.random.Generator] = None):
        self.evaluators = evaluators
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self.games: list[_Game] = []
        self.sims_total = 0

    def add(self, spec: GameSpec) -> None:
        self.games.append(_Game(spec))

    def step(self) -> list[dict]:
        """One batched round. Returns records of games that finished."""
        cfg, finished = self.cfg, []
        vl = 1.0 if cfg.leaves_per_game > 1 else 0.0
        batches: dict = {}

        for g in list(self.games):
            root = g.root
            if root.expanded and g.spec.add_noise and not g.noised:
                mcts.add_dirichlet(root, cfg.dirichlet_alpha, cfg.dirichlet_eps, self.rng)
                g.noised = True
            if root.expanded and root.n - 1 >= cfg.n_sims:
                rec = self._play_move(g)
                if rec is not None:
                    finished.append(rec)
                    self.games.remove(g)
                    continue
                root = g.root
            key = g.spec.players[g.board.turn]
            if callable(key):                  # external engine (e.g. Stockfish) to move
                rec = self._advance(g, key(g.board), None)
                if rec is not None:
                    finished.append(rec)
                    self.games.remove(g)
                continue
            n_leaves = cfg.leaves_per_game if root.expanded else 1
            for _ in range(n_leaves):
                p = mcts.gather_leaf(root, g.board, cfg.c_puct, cfg.fpu_reduction, vl)
                if p is mcts.COLLISION:
                    break
                g.sims += 1
                if isinstance(p, mcts.Pending):
                    batches.setdefault(key, []).append(p)

        for key, batch in batches.items():
            priors, values = self.evaluators[key]([p.planes for p in batch], [p.idx for p in batch])
            for p, pr, v in zip(batch, priors, values):
                mcts.complete_leaf(p, pr, float(v), vl)
            self.sims_total += len(batch)
        return finished

    def _play_move(self, g: _Game) -> Optional[dict]:
        root, board, spec = g.root, g.board, g.spec
        if spec.record:
            g.states.append(encoding.encode_board_u8(board))
            i, p = sparse_policy(root, board)
            g.pidx.append(i); g.pprob.append(p)
            g.stm.append(board.turn)

        N = root.N
        if board.ply() < spec.temp_plies:
            k = int(self.rng.choice(len(N), p=N / N.sum()))
        else:
            k = int(np.argmax(N))
        return self._advance(g, root.moves[k], k)

    def _advance(self, g: _Game, move: chess.Move, k: Optional[int]) -> Optional[dict]:
        """Play `move`; returns the game record if that ended the game."""
        board, spec = g.board, g.spec
        board.push(move)
        same_player = spec.players[chess.WHITE] == spec.players[chess.BLACK]
        nxt = g.root.kids[k] if (same_player and k is not None) else None  # reuse subtree only within one net
        g.root = nxt if nxt is not None and nxt.terminal is None and not nxt.pending else mcts.Node()
        g.noised = False

        capped = board.ply() >= spec.max_plies
        if not capped and mcts.terminal_value(board, list(board.legal_moves)) is None:
            return None
        return self._finish(g, capped)

    def _finish(self, g: _Game, capped: bool) -> dict:
        desc, result = outcome_str(g.board, capped)
        rec = {"outcome": desc, "result": result, "plies": g.board.ply(),
               "seconds": time.monotonic() - g.t0, "sims": g.sims, "tag": g.spec.tag}
        if g.spec.record and g.states:
            stm = np.array(g.stm)
            z = np.zeros(len(stm), dtype=np.float32)
            if result != 0:
                z[:] = np.where(stm == (result > 0), 1.0, -1.0)   # chess.WHITE is True
            rec.update(states=np.stack(g.states), pidx=np.stack(g.pidx),
                       pprob=np.stack(g.pprob), values=z)
        return rec


def selfplay_specs(temp_plies: int = 30, max_plies: int = 512):
    return GameSpec(players={chess.WHITE: "net", chess.BLACK: "net"},
                    temp_plies=temp_plies, max_plies=max_plies)


def run_selfplay(evaluator, cfg: SearchConfig, n_games: int, parallel: int,
                 on_game: Callable[[dict], None], temp_plies: int = 30,
                 max_plies: int = 512, should_stop: Callable[[], bool] = lambda: False,
                 between_steps: Callable[[], None] = lambda: None) -> None:
    """Play `n_games` (0 = forever) self-play games, `parallel` at a time."""
    runner = GameRunner({"net": evaluator}, cfg)
    started = 0
    while not should_stop():
        while len(runner.games) < parallel and (n_games == 0 or started < n_games):
            runner.add(selfplay_specs(temp_plies, max_plies))
            started += 1
        if not runner.games:
            break
        for rec in runner.step():
            on_game(rec)
        between_steps()


def load_net(path: str | Path | None, channels: int, blocks: int, device) -> ChessNet:
    net = ChessNet(channels=channels, blocks=blocks).to(device)
    if path and Path(path).exists():
        sd = torch.load(path, map_location=device)
        net.load_state_dict(sd["model"] if "model" in sd else sd)
    elif path:
        log.warning(f"checkpoint not found at {path}; using random init")
    return net.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="model weights; uses random init if missing")
    ap.add_argument("--buffer", type=str, default="games")
    ap.add_argument("--n-games", type=int, default=64)
    ap.add_argument("--parallel", type=int, default=128)
    ap.add_argument("--n-simulations", type=int, default=100)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-level", type=str, default="INFO",
                    help="DEBUG / INFO / WARNING / ERROR")
    args = ap.parse_args()

    log_path = setup_logging("self-play", args.log_level)
    log.info(f"args = {vars(args)}")
    if log_path:
        log.info(f"log file: {log_path}")
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    net = load_net(args.checkpoint, args.channels, args.blocks, device)
    ev = BatchEvaluator(net, device)
    buffer = ReplayBuffer(args.buffer, load=False)

    t0 = time.monotonic()
    done = []

    def on_game(rec):
        done.append(rec)
        buffer.save_games([rec])
        log.info(f"[game {len(done)}/{args.n_games}] plies={rec['plies']}  "
                 f"{rec['seconds']:.1f}s  -- {rec['outcome']}")

    run_selfplay(ev, SearchConfig(n_sims=args.n_simulations), args.n_games,
                 args.parallel, on_game)
    dt = time.monotonic() - t0
    plies = sum(r["plies"] for r in done)
    log.info(f"generated {len(done)} games / {plies} positions in {dt:.1f}s  "
             f"({ev.positions/dt:.0f} evals/s)")


if __name__ == "__main__":
    main()
