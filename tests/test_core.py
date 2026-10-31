"""Core correctness tests: encoding, search, game generation, replay buffer.

    .venv\\Scripts\\python.exe -m pytest -q
"""
from __future__ import annotations

import random

import chess
import numpy as np
import pytest
import torch

from src import encoding, mcts
from src.inference import BatchEvaluator
from src.network import ChessNet
from src.replay_buffer import POLICY_K, ReplayBuffer
from src.self_play import GameRunner, GameSpec, SearchConfig, selfplay_specs


def _random_positions(n_games=40, seed=0, shuffle_bias=0.0):
    rng = random.Random(seed)
    out = []
    for _ in range(n_games):
        b = chess.Board()
        while not b.is_game_over(claim_draw=False) and b.ply() < 250:
            moves = list(b.legal_moves)
            quiet = [m for m in moves if not b.is_capture(m)
                     and b.piece_type_at(m.from_square) != chess.PAWN]
            b.push(rng.choice(quiet if quiet and rng.random() < shuffle_bias else moves))
            out.append(b.copy())
    return out


def _slow_encode(board: chess.Board) -> np.ndarray:
    """Reference per-square encoder (the original implementation)."""
    us, flip = board.turn, board.turn == chess.BLACK
    planes = np.zeros((encoding.N_PLANES, 8, 8), dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        plane = (0 if piece.color == us else 6) + encoding.PIECE_TYPES.index(piece.piece_type)
        s = encoding._flip_square(sq) if flip else sq
        planes[plane, chess.square_rank(s), chess.square_file(s)] = 1.0
    planes[12] = 1.0
    planes[13] = board.has_kingside_castling_rights(us)
    planes[14] = board.has_queenside_castling_rights(us)
    planes[15] = board.has_kingside_castling_rights(not us)
    planes[16] = board.has_queenside_castling_rights(not us)
    planes[17] = min(board.halfmove_clock / 100.0, 1.0)
    planes[18] = 1.0 if board.is_repetition(2) else 0.0
    return planes


@pytest.fixture(scope="module")
def positions():
    return _random_positions()


@pytest.fixture(scope="module")
def tiny_eval():
    torch.manual_seed(0)
    net = ChessNet(channels=16, blocks=1).eval()
    return BatchEvaluator(net, "cpu")


# -- encoding ---------------------------------------------------------------

def test_encode_matches_reference(positions):
    for b in positions[::7]:
        assert np.array_equal(encoding.encode_board(b), _slow_encode(b)), b.fen()


def test_move_indices_roundtrip(positions):
    for b in positions[::5]:
        moves, idx = encoding.legal_moves_and_indices(b)
        assert len(set(idx.tolist())) == len(moves)
        for m, i in zip(moves, idx):
            assert i == encoding.move_to_index(m, b)
            assert encoding.index_to_move(int(i), b) == m


def test_fast_repetition_matches_python_chess():
    for b in _random_positions(n_games=60, seed=3, shuffle_bias=0.9)[::3]:
        for c in (2, 3):
            assert encoding.is_repetition(b, c) == b.is_repetition(c), b.fen()


# -- search -----------------------------------------------------------------

def test_mcts_finds_mate_in_one(tiny_eval):
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 2 3")
    fen = b.fen()
    root = mcts.search(mcts.Node(), b, tiny_eval, n_simulations=300)
    assert b.fen() == fen                                      # board restored
    best = max(root.children.items(), key=lambda kv: kv[1].visit_count)[0]
    assert b.san(best) == "Qxf7#"
    assert root.value > 0.5


def test_mcts_visit_accounting(tiny_eval):
    for leaf_batch in (1, 8):
        root = mcts.search(mcts.Node(), chess.Board(), tiny_eval, 64, leaf_batch=leaf_batch)
        assert root.n == 65                     # expansion + 64 simulations
        assert root.N.sum() == 64               # virtual loss fully reverted
        assert abs(sum(mcts.visit_count_policy(root).values()) - 1) < 1e-6


def test_terminal_root(tiny_eval):
    mated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    root = mcts.search(mcts.Node(), mated, tiny_eval, 10)
    assert root.terminal == -1.0 and root.children == {}


# -- games ------------------------------------------------------------------

def test_selfplay_records(tiny_eval):
    runner = GameRunner({"net": tiny_eval}, SearchConfig(n_sims=8))
    for _ in range(3):
        runner.add(selfplay_specs(temp_plies=10, max_plies=40))
    recs = []
    while runner.games:
        recs += runner.step()
    assert len(recs) == 3
    for r in recs:
        n = r["plies"]
        assert r["states"].shape == (n, encoding.N_PLANES, 8, 8) and r["states"].dtype == np.uint8
        assert r["pidx"].shape == (n, POLICY_K) and r["pprob"].shape == (n, POLICY_K)
        assert np.allclose(r["pprob"].astype(np.float32).sum(1), 1, atol=1e-2)
        if r["result"] == 0:
            assert not r["values"].any()
        else:   # value alternates sign with side to move; first ply is white's
            assert r["values"][0] == r["result"] and r["values"][1] == -r["result"]


def test_value_signs_for_checkmate(tiny_eval):
    """Fool's mate played out by an external 'player' scripted move list."""
    script = iter(["f3", "e5", "g4", "Qh4#"])
    play = lambda board: board.parse_san(next(script))
    runner = GameRunner({}, SearchConfig(n_sims=4))
    runner.add(GameSpec(players={chess.WHITE: play, chess.BLACK: play}, record=False))
    while runner.games:
        recs = runner.step()
    assert recs[0]["result"] == -1 and "black wins" in recs[0]["outcome"]


# -- replay buffer ----------------------------------------------------------

def _fake_rec(n, val):
    return {"states": np.full((n, encoding.N_PLANES, 8, 8), val % 2, np.uint8),
            "pidx": np.zeros((n, POLICY_K), np.int16),
            "pprob": np.ones((n, POLICY_K), np.float16) / POLICY_K,
            "values": np.full(n, val, np.float32)}


def test_buffer_ring_and_reload(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity=100)
    for i in range(5):
        buf.save_games([_fake_rec(30, float(i))])
    assert buf.size == 100
    assert set(np.unique(buf.values)) <= {1.0, 2.0, 3.0, 4.0}   # oldest overwritten
    s, pi, pp, z = buf.sample(16)
    assert s.shape == (16, encoding.N_PLANES, 8, 8)

    again = ReplayBuffer(tmp_path, capacity=100)                 # reload from disk
    assert again.size == 100 and 4.0 in again.values

    other = ReplayBuffer(tmp_path, load=False)                   # e.g. the Lichess farm
    other.save_games([_fake_rec(10, 9.0)])
    assert again.refresh() == 10 and 9.0 in again.values


def test_buffer_dense_compat(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity=50)
    b = chess.Board()
    b.push_san("e4")
    states = np.stack([encoding.encode_board(b)])
    pol = np.zeros((1, encoding.N_MOVES), np.float32)
    _, idx = encoding.legal_moves_and_indices(b)
    pol[0, idx[:3]] = [0.5, 0.25, 0.25]
    buf.save_game(states, pol, np.array([1.0], np.float32))
    assert np.array_equal(buf.states[0], encoding.encode_board_u8(b))
    assert sorted(buf.pidx[0][:3].tolist()) == sorted(idx[:3].tolist())


def test_claimable_draw_root_still_searched(tiny_eval):
    """50-move rule reached but not claimed (Lichess plays on): we still need a move."""
    b = chess.Board("8/8/4k3/8/8/3K4/8/R7 w - - 100 120")
    root = mcts.search(mcts.Node(), b, tiny_eval, 20)
    assert root.terminal is None and len(root.moves) > 0 and root.N.sum() == 20
