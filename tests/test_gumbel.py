import chess
import numpy as np
import torch

from src import encoding, gumbel, mcts
from src.inference2 import BatchEvaluatorV2
from src.network2 import ChessNetV2
from src.replay2 import ReplayBuffer2
from src.selfplay2 import GumbelConfig, GumbelSelfPlay, to_buffer_rec


def test_sequential_halving_table():
    t = gumbel.considered_visits(16, 128)
    assert len(t) == 128
    # 4 phases of 32 sims: 16 moves x2, 8 x4, 4 x8, 2 x16
    assert t[:16] == tuple([0] * 16) and t[16:32] == tuple([1] * 16)
    assert t[32:40] == tuple([2] * 8)
    assert t[-1] == 29                           # 2 + 4 + 8 + 16 visits, 0-based
    assert gumbel.considered_visits(1, 5) == (0, 1, 2, 3, 4)


def _uniform_ev(planes, idx, parity=None):
    return [np.full(len(i), 1.0 / len(i), np.float32) for i in idx], np.zeros(len(idx), np.float32)


def _search(board, n, m, ev, g=None):
    root = mcts.Node()
    p = mcts.gather_leaf(root, board)
    pr, v = ev([p.planes], [p.idx])
    mcts.complete_leaf(p, pr[0], float(v[0]))
    g = np.zeros(len(root.moves)) if g is None else g
    logits = gumbel.logits_of(root)
    while root.N.sum() < n:
        a = gumbel.root_action(root, g, logits, m, n)
        p = mcts.gather_leaf(root, board, root_action=a)
        if isinstance(p, mcts.Pending):
            pr, v = ev([p.planes], [p.idx])
            mcts.complete_leaf(p, pr[0], float(v[0]))
    return root, g, logits


def test_gumbel_finds_mate_in_one_and_targets_it():
    # White to move: Qh5xf7# (scholar's mate)
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 4")
    mate = chess.Move.from_uci("h5f7")
    rng = np.random.default_rng(0)
    root, g, logits = _search(b, 64, 64, _uniform_ev)   # consider every move
    assert int(root.N.sum()) == 64
    a = gumbel.final_action(root, g, logits)
    assert root.moves[a] == mate
    pi = gumbel.improved_policy(root, logits)
    assert abs(pi.sum() - 1) < 1e-6 and root.moves[int(np.argmax(pi))] == mate
    # board untouched
    assert b.fen().startswith("r1bqkbnr/pppp1ppp/2n5/4p2Q")


def test_forced_root_action_is_used():
    b = chess.Board()
    root = mcts.Node()
    p = mcts.gather_leaf(root, b)
    pr, v = _uniform_ev([p.planes], [p.idx])
    mcts.complete_leaf(p, pr[0], 0.0)
    for k in (3, 3, 7):
        p = mcts.gather_leaf(root, b, root_action=k)
        assert p.path[0] == (root, k)
        mcts.complete_leaf(p, *[x[0] for x in _uniform_ev([p.planes], [p.idx])])
    assert root.N[3] == 2 and root.N[7] == 1


def test_selfplay_records_targets(tmp_path):
    torch.manual_seed(0)
    ev = BatchEvaluatorV2(ChessNetV2(16, 1).eval(), "cpu", pad_to=1)
    cfg = GumbelConfig(n_full=8, m_full=4, n_fast=4, m_fast=2, p_full=0.5, max_plies=40)
    sp = GumbelSelfPlay(ev, cfg, parallel=3, rng=np.random.default_rng(1))
    recs = []
    while len(recs) < 3:
        recs += sp.step()
    buf = ReplayBuffer2(tmp_path / "g", capacity=10_000)
    for r in recs:
        assert r["plies"] <= 40
        if "states" not in r:
            continue
        n = len(r["states"])
        assert r["pidx"].shape == (n, 96) and r["pos_result"].shape == (n,)
        assert np.allclose(r["pprob"].astype(np.float32).sum(1), 1, atol=1e-2)
        if r["outcome"] == "draw (move cap)":
            assert (r["plies_left"] == -1).all() and (r["pos_result"] == 0).all()
        else:
            assert (r["plies_left"] >= 1).all()
    n_saved = buf.save_games([to_buffer_rec(r) for r in recs if "states" in r])
    assert buf.size == n_saved > 0
    again = ReplayBuffer2(tmp_path / "g", capacity=10_000)
    assert again.size == n_saved
    s = again.sample(5, np.random.default_rng(0))
    assert s["states"].shape == (5, encoding.N_PLANES, 8, 8)


def test_selfplay_loss_runs():
    from src.loop2 import selfplay_losses

    net = ChessNetV2(16, 1)
    n = 4
    b = {"states": np.stack([encoding.encode_board_u8(chess.Board())] * n),
         "pidx": np.zeros((n, 96), np.int16), "pprob": np.zeros((n, 96), np.float16),
         "result": np.array([1, 0, -1, 1], np.int8), "plies": np.array([10, -1, 5, 3], np.int16)}
    _, idx = encoding.legal_moves_and_indices(chess.Board())
    b["pidx"][:, :len(idx)] = idx
    b["pprob"][:, :len(idx)] = 1.0 / len(idx)
    lp, lv, lm = selfplay_losses(net, b, torch.device("cpu"))
    (lp + lv + lm).backward()
    assert all(torch.isfinite(x) for x in (lp, lv, lm))
