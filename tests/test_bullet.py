import time

import chess
import numpy as np

from src.bullet import (BulletConfig, BulletEngine, Clock, TimeManager,
                        allows_mate_in_one, pick_move)
from src.mcts import Node, search


def uniform_eval(planes, idx):
    """Dummy network: uniform priors, value 0."""
    return [np.full(len(i), 1.0 / len(i), np.float32) for i in idx], np.zeros(len(idx), np.float32)


def searched_root(board, sims=0):
    root = Node()
    search(root, board, uniform_eval, sims)
    return root


def force_rank(root, best_uci):
    """Pretend the search strongly preferred `best_uci`."""
    for i, m in enumerate(root.moves):
        root.N[i] = 100.0 if m.uci() == best_uci else 1.0


# -- time manager ------------------------------------------------------------

def test_time_manager_modes():
    tm = TimeManager(BulletConfig())
    ms, mode = tm.budget(Clock(60_000, 60_000), 0)
    assert mode == "opening" and 0 < ms < 1000
    ms, mode = tm.budget(Clock(40_000, 40_000), 20)
    assert mode == "normal" and 500 < ms < 3000
    assert tm.budget(Clock(2_000, 30_000), 30) == (0, "panic")
    ms, mode = tm.budget(Clock(20_000, 4_000), 30)
    assert mode == "flag" and ms <= 100
    # with a real increment we never go fully instant
    assert tm.budget(Clock(2_000, 30_000, inc_ms=2_000), 30)[1] != "panic"


def test_budget_never_exceeds_fraction_of_clock():
    cfg = BulletConfig()
    tm = TimeManager(cfg)
    for our in (5_000, 10_000, 30_000, 60_000, 120_000):
        for n in (0, 10, 40, 80):
            ms, _ = tm.budget(Clock(our, our), n)
            assert ms <= cfg.max_fraction * our


# -- safety filter -------------------------------------------------------------

def test_avoids_move_allowing_mate_in_one():
    # White to move; Qd1-h5?? is fine, but moving the f2 pawn... use a classic:
    # after 1.f3 e5 2.g4?? Qh4# — here White must not play g4.
    b = chess.Board()
    for u in ("f2f3", "e7e5"):
        b.push_uci(u)
    assert allows_mate_in_one(b, chess.Move.from_uci("g2g4"))
    root = searched_root(b)
    force_rank(root, "g2g4")
    move, rejected = pick_move(b, root, want_decisive=False, flagging=False)
    assert move.uci() != "g2g4"
    assert ("g2g4", "mate-in-1") in rejected


def test_avoids_stalemate_when_winning():
    # White K+Q vs K: Qc7?? stalemates (black king a8, white king b6).
    b = chess.Board("k7/8/1K6/8/8/8/8/2Q5 w - - 0 1")
    root = searched_root(b)
    force_rank(root, "c1c7")
    move, rejected = pick_move(b, root, want_decisive=True, flagging=False, near_best=0.0)
    assert move.uci() != "c1c7"
    b.push(move)
    assert not b.is_stalemate()


def test_keeps_mating_material_when_flagging():
    # White K+R+P vs K+R: Rxh7?? is fine; trading rooks with Rxd8 would leave
    # K+P — still mating material. Use a position where the only capture leaves
    # a bare king: K+N vs K+P, Nxp leaves K+N (insufficient) for White.
    b = chess.Board("4k3/8/8/8/8/8/3p4/3NK3 w - - 0 1")
    root = searched_root(b)
    force_rank(root, "e1d2")             # Kxd2 leaves K+N vs K
    move, rejected = pick_move(b, root, want_decisive=True, flagging=True, near_best=0.0)
    assert move.uci() != "e1d2"


# -- engine ----------------------------------------------------------------------

def test_engine_plays_legal_moves_and_reuses_tree():
    eng = BulletEngine(uniform_eval, BulletConfig(overhead_ms=0, ponder=True))
    b = chess.Board()
    clock = Clock(60_000, 60_000)
    try:
        for _ in range(6):
            info = eng.choose(b, clock)
            assert info.move in b.legal_moves
            b.push(info.move)
            time.sleep(0.05)              # let the ponder thread run a little
            reply = next(iter(b.legal_moves))
            b.push(reply)
        info = eng.choose(b, clock)
        assert info.reused > 0            # pondered / previous tree was reused
    finally:
        eng.close()


def test_engine_plays_mate_in_one_instantly():
    # Scholar's mate available: Qxf7#
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    eng = BulletEngine(uniform_eval, BulletConfig(ponder=False))
    info = eng.choose(b, Clock(60_000, 60_000))
    assert info.move.uci() == "h5f7" and info.mode == "mate"
    eng.close()


def test_engine_instant_move_in_panic():
    eng = BulletEngine(uniform_eval, BulletConfig(ponder=False))
    b = chess.Board()
    info = eng.choose(b, Clock(1_000, 30_000))
    assert info.mode == "panic" and info.think_ms == 0
    assert info.move in b.legal_moves
    eng.close()
