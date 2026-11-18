"""Gumbel root search (Danihelka et al., "Policy improvement by planning with
Gumbel", ICLR 2022), on top of the PUCT tree in mcts.py.

At the root, instead of PUCT + Dirichlet noise:
  * sample Gumbel noise g(a) and consider the top-m moves by g(a) + logit(a);
  * spend the simulations by *sequential halving*: visit every considered
    move equally, drop the worse half by g + logit + sigma(q), repeat;
  * play the surviving move (with noise this samples from the improved
    policy; without noise it is the best move);
  * train the policy towards the *improved policy*
        softmax(logit(a) + sigma(completed_q(a))),
    which is a policy improvement even with very few simulations — the
    reason for using it: it learns well at 16-128 simulations per move,
    where AlphaZero's visit-count targets are poor.
Below the root, selection stays PUCT (mcts.gather_leaf with root_action).

Stateless, like mctx: which move to visit next follows from the root's
current visit counts and the precomputed visit table, so the batched
self-play runner only needs one call per simulation.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .mcts import Node

C_VISIT = 50.0
C_SCALE = 1.0


@lru_cache(maxsize=256)
def considered_visits(m: int, n: int) -> tuple[int, ...]:
    """For simulation k (0-based), the visit count a considered move must
    currently have to be eligible (mctx get_sequence_of_considered_visits)."""
    if m <= 1:
        return tuple(range(n))
    log2max = int(math.ceil(math.log2(m)))
    seq: list[int] = []
    visits = [0] * m
    num = m
    while len(seq) < n:
        extra = max(1, int(n / (log2max * num)))
        for _ in range(extra):
            seq.extend(visits[:num])
            for i in range(num):
                visits[i] += 1
        num = max(2, num // 2)
    return tuple(seq[:n])


def completed_q(root: Node) -> np.ndarray:
    """Q per root move from the root player's view, in [0, 1] after min-max
    rescaling; unvisited moves get the root's mean value."""
    N = root.N
    v_root = root.value                       # root STM POV, [-1, 1]
    q = np.where(N > 0, root.W / np.maximum(N, 1.0), v_root).astype(np.float64)
    lo, hi = q.min(), q.max()
    if hi - lo > 1e-6:
        return (q - lo) / (hi - lo)
    return np.full_like(q, 0.5)


def sigma(root: Node, q: np.ndarray) -> np.ndarray:
    return (C_VISIT + float(root.N.max(initial=0.0))) * C_SCALE * q


def logits_of(root: Node) -> np.ndarray:
    return np.log(np.maximum(root.priors.astype(np.float64), 1e-12))


def root_action(root: Node, gumbel: np.ndarray, logits: np.ndarray, m: int, n_sims: int) -> int:
    """Root move to visit for the next simulation."""
    k = int(root.N.sum())
    table = considered_visits(min(m, len(root.moves)), n_sims)
    target = table[min(k, len(table) - 1)]
    score = gumbel + logits + sigma(root, completed_q(root))
    score = np.where(root.N == target, score, -np.inf)
    if not np.isfinite(score).any():          # can't happen with a fresh root; be safe
        score = gumbel + logits
    return int(np.argmax(score))


def final_action(root: Node, gumbel: np.ndarray, logits: np.ndarray) -> int:
    """Move to play: among the most-visited moves, the best by g + logit + sigma(q)."""
    score = gumbel + logits + sigma(root, completed_q(root))
    score = np.where(root.N == root.N.max(), score, -np.inf)
    return int(np.argmax(score))


def improved_policy(root: Node, logits: np.ndarray) -> np.ndarray:
    """Training target over root.moves: softmax(logit + sigma(completed q))."""
    z = logits + sigma(root, completed_q(root))
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()
