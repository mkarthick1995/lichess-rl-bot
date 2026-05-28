"""Monte Carlo Tree Search guided by a policy/value network (PUCT)."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import chess

from . import encoding


class Node:
    __slots__ = ("prior", "visit_count", "value_sum", "children")

    def __init__(self, prior: float):
        self.prior:       float = prior
        self.visit_count: int   = 0
        self.value_sum:   float = 0.0
        self.children:    dict[chess.Move, "Node"] = {}

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def is_expanded(self) -> bool:
        return bool(self.children)


def _ucb(parent: Node, child: Node, c_puct: float) -> float:
    # child.value is from the child's STM POV; negate for the parent.
    prior_term = c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    return -child.value + prior_term


def _select_child(node: Node, c_puct: float) -> tuple[chess.Move, Node]:
    best_score = -float("inf")
    best: tuple[chess.Move, Node] | None = None
    for move, child in node.children.items():
        s = _ucb(node, child, c_puct)
        if s > best_score:
            best_score = s
            best = (move, child)
    assert best is not None
    return best


@torch.inference_mode()
def _evaluate(board: chess.Board, net, device) -> tuple[np.ndarray, float]:
    x = torch.from_numpy(encoding.encode_board(board)).unsqueeze(0).to(device)
    logits, value = net(x)
    logits = logits[0].detach().cpu().numpy()

    mask = encoding.legal_move_mask(board)
    logits = logits - logits.max()                  # numerical stability
    exp = np.exp(logits, dtype=np.float64) * mask
    total = exp.sum()
    if total > 0:
        policy = (exp / total).astype(np.float32)
    else:
        # No legal moves matched (shouldn't happen in non-terminal positions),
        # fall back to uniform over the mask.
        n = int(mask.sum())
        policy = (mask.astype(np.float32) / n) if n else mask.astype(np.float32)
    return policy, float(value.item())


def _expand(node: Node, board: chess.Board, net, device) -> float:
    policy, value = _evaluate(board, net, device)
    for move in board.legal_moves:
        try:
            idx = encoding.move_to_index(move, board)
        except ValueError:
            continue
        node.children[move] = Node(prior=float(policy[idx]))
    return value


def _terminal_value(board: chess.Board) -> float:
    """Value of a terminal position from the STM's perspective."""
    outcome = board.outcome(claim_draw=False)
    if outcome is None or outcome.winner is None:
        return 0.0
    # If STM equals the winner this is impossible (winner just moved). For
    # checkmate the STM has lost → -1; stalemate is handled above.
    return 1.0 if outcome.winner == board.turn else -1.0


def _add_dirichlet(node: Node, alpha: float, eps: float, rng: np.random.Generator) -> None:
    if not node.children:
        return
    noise = rng.dirichlet([alpha] * len(node.children))
    for n, child in zip(noise, node.children.values()):
        child.prior = (1 - eps) * child.prior + eps * float(n)


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
) -> Node:
    if rng is None:
        rng = np.random.default_rng()

    root = Node(prior=0.0)
    _expand(root, board, net, device)
    root.visit_count = 1   # so sqrt(N) is non-zero for the first child selection
    if add_noise:
        _add_dirichlet(root, dirichlet_alpha, dirichlet_eps, rng)

    for _ in range(n_simulations):
        node = root
        sim_board = board.copy(stack=False)
        path = [node]

        # SELECTION ---------------------------------------------------------
        while node.is_expanded():
            move, node = _select_child(node, c_puct)
            sim_board.push(move)
            path.append(node)

        # EXPANSION / EVALUATION -------------------------------------------
        if sim_board.is_game_over(claim_draw=False):
            value = _terminal_value(sim_board)
        else:
            value = _expand(node, sim_board, net, device)

        # BACKUP — sign flips at each ply ----------------------------------
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum   += value
            value = -value

    return root


def visit_count_policy(root: Node, temperature: float = 1.0) -> dict[chess.Move, float]:
    """Move → probability proportional to visit counts (with temperature)."""
    moves  = list(root.children.keys())
    visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)
    if temperature == 0 or visits.max() == 0:
        probs = np.zeros_like(visits)
        probs[int(np.argmax(visits))] = 1.0
    else:
        visits = visits ** (1.0 / temperature)
        probs = visits / visits.sum()
    return dict(zip(moves, probs))
