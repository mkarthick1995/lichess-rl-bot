"""Monte Carlo Tree Search (PUCT) guided by a policy/value network — batched.

The search is split into two halves so that many leaves (from one tree with
virtual loss, or from many trees — one per concurrent game) can share a single
network forward pass:

    pending = gather_leaf(root, board, ...)   # select → leaf; encodes it
    ...collect many pendings, evaluate them in one batch (inference.py)...
    complete_leaf(pending, priors, value)     # expand + back up

Per-edge statistics live in numpy arrays on the parent node (N = visits,
W = summed value from the parent's side-to-move POV), so selection is a few
vectorised ops instead of a Python loop over child objects.

`run_mcts` / `visit_count_policy` keep the original single-position API used
by the Lichess wiring.
"""
from __future__ import annotations

import math
from typing import Callable, NamedTuple, Optional

import numpy as np
import chess

from . import encoding


class Node:
    __slots__ = ("moves", "priors", "N", "W", "kids", "n", "q_sum", "terminal", "pending")

    def __init__(self):
        self.moves: list[chess.Move] | None = None   # set on expansion
        self.priors: np.ndarray | None = None
        self.N: np.ndarray | None = None              # per-edge visits
        self.W: np.ndarray | None = None              # per-edge value sum (this node's STM POV)
        self.kids: list[Node | None] | None = None
        self.n: float = 0                             # visits through this node
        self.q_sum: float = 0.0                       # value sum, this node's STM POV
        self.terminal: float | None = None            # game-over value, STM POV
        self.pending: bool = False                    # awaiting network eval

    @property
    def expanded(self) -> bool:
        return self.moves is not None

    @property
    def value(self) -> float:
        """Mean value from this node's side-to-move perspective."""
        return self.q_sum / self.n if self.n else 0.0

    @property
    def visit_count(self) -> int:
        return int(self.n)

    @property
    def children(self) -> dict[chess.Move, "EdgeView"]:
        """move → EdgeView (visit_count, prior, value from the child's STM POV).
        Convenience view for logging / move choice; not used on the hot path."""
        if not self.moves:
            return {}
        out = {}
        for i, m in enumerate(self.moves):
            n = int(self.N[i])
            out[m] = EdgeView(n, float(self.priors[i]),
                              -float(self.W[i]) / n if n else 0.0)
        return out


class EdgeView(NamedTuple):
    visit_count: int
    prior: float
    value: float


class Pending(NamedTuple):
    node: Node
    path: list                 # [(parent Node, edge index), ...] root → leaf
    planes: np.ndarray         # uint8 (19, 8, 8)
    moves: list
    idx: np.ndarray            # policy indices of `moves`


# gather_leaf outcomes other than a Pending:
DONE = 1        # hit a terminal node; already backed up
COLLISION = 2   # leaf already awaiting evaluation (virtual-loss batch); retry later


def terminal_value(board: chess.Board, moves: list) -> Optional[float]:
    """Game-over value from the STM's POV, or None if the game continues.
    Draw rules match self-play adjudication: threefold repetition and the
    50-move rule end the game (as if claimed)."""
    if not moves:
        return -1.0 if board.is_check() else 0.0
    if (board.halfmove_clock >= 100 or board.is_insufficient_material()
            or encoding.is_repetition(board, 3)):
        return 0.0
    return None


def _select(node: Node, c_puct: float, fpu_reduction: float) -> int:
    N = node.N
    fpu = node.q_sum / node.n - fpu_reduction
    q = np.where(N > 0, node.W / np.maximum(N, 1.0), fpu)
    u = (c_puct * math.sqrt(node.n)) * node.priors / (1.0 + N)
    return int(np.argmax(q + u))


def _backup(node: Node, path: list, value: float, vl: float) -> None:
    node.n += 1
    node.q_sum += value
    v = -value
    for parent, i in reversed(path):
        parent.N[i] += 1 - vl
        parent.W[i] += v + vl
        parent.n += 1 - vl
        parent.q_sum += v
        v = -v


def _revert_virtual_loss(path: list, vl: float) -> None:
    for parent, i in path:
        parent.N[i] -= vl
        parent.W[i] += vl
        parent.n -= vl


def gather_leaf(root: Node, board: chess.Board, c_puct: float = 1.5,
                fpu_reduction: float = 0.2, vl: float = 0.0):
    """Walk from `root` (whose position is `board`) to a leaf.

    Returns a Pending to evaluate, DONE (terminal leaf, backed up) or
    COLLISION. `board` is mutated during the walk and restored before return.
    With vl > 0, virtual loss is applied along the path so further calls pick
    different leaves before the batch is evaluated."""
    node, path = root, []
    while node.moves is not None and node.terminal is None:
        i = _select(node, c_puct, fpu_reduction)
        path.append((node, i))
        if vl:
            node.N[i] += vl
            node.W[i] -= vl
            node.n += vl
        board.push(node.moves[i])
        kid = node.kids[i]
        if kid is None:
            kid = node.kids[i] = Node()
        node = kid

    result = DONE
    try:
        if node.terminal is None:
            if node.pending:
                _revert_virtual_loss(path, vl)
                return COLLISION
            moves, idx = encoding.legal_moves_and_indices(board)
            term = terminal_value(board, moves)
            if term is None:
                node.pending = True
                return Pending(node, path, encoding.encode_board_u8(board), moves, idx)
            node.terminal = term
            if node is root:           # searching a finished game: nothing to do
                root.moves, root.kids = [], []
                root.priors = root.N = root.W = np.zeros(0, np.float32)
        _backup(node, path, node.terminal, vl)
        return result
    finally:
        for _ in path:
            board.pop()


def complete_leaf(p: Pending, priors: np.ndarray, value: float, vl: float = 0.0) -> None:
    """Expand the pending leaf with network `priors` (over p.moves) and back
    up `value` (leaf STM POV)."""
    node = p.node
    k = len(p.moves)
    node.moves = p.moves
    node.priors = priors.astype(np.float32, copy=False)
    node.N = np.zeros(k, dtype=np.float32)
    node.W = np.zeros(k, dtype=np.float32)
    node.kids = [None] * k
    node.pending = False
    _backup(node, p.path, value, vl)


def add_dirichlet(node: Node, alpha: float, eps: float, rng: np.random.Generator) -> None:
    if node.moves:
        noise = rng.dirichlet([alpha] * len(node.moves)).astype(np.float32)
        node.priors = (1 - eps) * node.priors + eps * noise


def visit_policy(root: Node) -> np.ndarray:
    """Visit distribution over root.moves (temperature 1)."""
    total = root.N.sum()
    return root.N / total if total > 0 else np.full(len(root.N), 1.0 / max(1, len(root.N)), np.float32)


# ---------------------------------------------------------------------------
# Single-position API (Lichess wiring, standalone tools).

Evaluator = Callable[[list, list], tuple[list, np.ndarray]]


def search(root: Node, board: chess.Board, evaluate: Evaluator, n_simulations: int,
           c_puct: float = 1.5, fpu_reduction: float = 0.2, leaf_batch: int = 16,
           on_root_expanded: Optional[Callable[[Node], None]] = None) -> Node:
    """Run simulations on one tree until root has `n_simulations` visits
    beyond its own expansion, batching up to `leaf_batch` leaves per network
    call via virtual loss."""
    vl = 1.0 if leaf_batch > 1 else 0.0
    if not root.expanded:
        p = gather_leaf(root, board, c_puct, fpu_reduction)
        if isinstance(p, Pending):
            priors, values = evaluate([p.planes], [p.idx])
            complete_leaf(p, priors[0], float(values[0]))
    if root.terminal is not None and root.terminal == 0.0 and any(board.legal_moves):
        # A draw is *claimable* here (threefold / 50-move) but nobody claimed
        # it — e.g. on Lichess the game simply continues — so we still need a
        # move. Expand the root normally; deeper nodes keep the draw rule.
        moves, idx = encoding.legal_moves_and_indices(board)
        priors, values = evaluate([encoding.encode_board_u8(board)], [idx])
        root.terminal, root.n, root.q_sum = None, 0, 0.0
        complete_leaf(Pending(root, [], None, moves, idx), priors[0], float(values[0]))
    if on_root_expanded is not None:
        on_root_expanded(root)
    if root.terminal is not None or not root.moves:
        return root

    while root.n - 1 < n_simulations:
        want = min(leaf_batch, int(n_simulations - (root.n - 1)))
        batch = []
        for _ in range(want):
            p = gather_leaf(root, board, c_puct, fpu_reduction, vl)
            if p is COLLISION:
                break
            if isinstance(p, Pending):
                batch.append(p)
        if batch:
            priors, values = evaluate([p.planes for p in batch], [p.idx for p in batch])
            for p, pr, v in zip(batch, priors, values):
                complete_leaf(p, pr, float(v), vl)
    return root


def run_mcts(
    board: chess.Board,
    net,
    device,
    n_simulations: int = 200,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    add_noise: bool = True,
    rng: Optional[np.random.Generator] = None,
    leaf_batch: int = 16,
) -> Node:
    """Search `board` from scratch with `net`; returns the root Node.
    `board` is not modified."""
    from .inference import evaluator_for

    if rng is None:
        rng = np.random.default_rng()
    noise = (lambda r: add_dirichlet(r, dirichlet_alpha, dirichlet_eps, rng)) if add_noise else None
    return search(Node(), board.copy(), evaluator_for(net, device), n_simulations,
                  c_puct=c_puct, leaf_batch=leaf_batch, on_root_expanded=noise)


def visit_count_policy(root: Node, temperature: float = 1.0) -> dict[chess.Move, float]:
    """Move → probability proportional to visit counts (with temperature)."""
    moves = list(root.moves or [])
    if not moves:
        return {}
    visits = root.N.astype(np.float64)
    if temperature == 0 or visits.max() == 0:
        probs = np.zeros_like(visits)
        probs[int(np.argmax(visits))] = 1.0
    else:
        visits = visits ** (1.0 / temperature)
        probs = visits / visits.sum()
    return dict(zip(moves, probs))
