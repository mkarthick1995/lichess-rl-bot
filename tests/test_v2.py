import numpy as np
import torch
import chess

from src import encoding
from src.lichess_db import SAMPLE_DTYPE, game_samples, keep_game
from src.network2 import N_INPUT, ChessNetV2, planes_u8_to_input, samples_to_input
from src.pretrain import ShardStream

# A real-looking Lichess export: castling both sides, a capture, a promotion
# race is too long to script, so add an en-passant-free middlegame and checks.
PGN = """[Event "Rated Bullet game"]
[Site "https://lichess.org/abcdefgh"]
[White "a"]
[Black "b"]
[Result "0-1"]
[WhiteElo "2210"]
[BlackElo "2305"]
[TimeControl "60+0"]
[Termination "Time forfeit"]

1. e4 { [%clk 0:01:00] } 1... e5 { [%clk 0:01:00] } 2. Nf3 { [%clk 0:00:59.5] } 2... Nc6 { [%clk 0:00:59] } 3. Bc4 { [%clk 0:00:58] } 3... Nf6 { [%clk 0:00:58.2] } 4. O-O { [%clk 0:00:57] } 4... Bc5 { [%clk 0:00:57] } 5. d3 { [%clk 0:00:55] } 5... O-O { [%clk 0:00:56] } 6. Bg5 { [%clk 0:00:52] } 6... h6 { [%clk 0:00:54] } 7. Bxf6 { [%clk 0:00:50] } 7... Qxf6 { [%clk 0:00:53] } 8. Nc3 { [%clk 0:00:47] } 8... d6 { [%clk 0:00:51] } 9. Nd5 { [%clk 0:00:44] } 9... Qd8 { [%clk 0:00:50] } 10. c3 { [%clk 0:00:40] } 10... Be6 { [%clk 0:00:48] } 0-1

"""


def replay():
    b = chess.Board()
    out = []
    for san in ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "O-O", "Bc5", "d3", "O-O",
                "Bg5", "h6", "Bxf6", "Qxf6", "Nc3", "d6", "Nd5", "Qd8", "c3", "Be6"]:
        m = b.parse_san(san)
        out.append((b.copy(), m))
        b.push(m)
    return out


def test_filter():
    h = {"Event": "Rated Bullet game", "Termination": "Normal", "Result": "1-0",
         "WhiteElo": "2100", "BlackElo": "2050", "TimeControl": "60+0"}
    assert keep_game(h, 2000)
    assert not keep_game({**h, "BlackElo": "1990"}, 2000)
    assert not keep_game({**h, "Event": "Rated Blitz game"}, 2000)
    assert not keep_game({**h, "Termination": "Abandoned"}, 2000)
    assert not keep_game({**h, "TimeControl": "15+0"}, 2000)       # UltraBullet


def test_samples_match_play_time_encoding():
    rows, val = game_samples(PGN, 2000)
    positions = replay()
    assert len(rows) == len(positions) == 20
    x = samples_to_input(rows, torch.device("cpu"))
    assert x.shape == (20, N_INPUT, 8, 8)
    for i, (board, move) in enumerate(positions):
        ref = encoding.planes_to_float(encoding.encode_board_u8(board))
        assert np.allclose(x[i, :encoding.N_PLANES].numpy(), ref), f"ply {i}"
        assert rows["move"][i] == encoding.move_to_index(move, board)
        assert encoding.index_to_move(int(rows["move"][i]), board) == move
    # targets: black won; side to move alternates starting with white
    assert list(rows["result"][:4]) == [-1, 1, -1, 1]
    assert list(rows["plies_left"][:2]) == [20, 19] and rows["plies_left"][-1] == 1
    # clocks: before White's 3rd move White has 59.5 s, Black 59.0 s
    assert rows["our_ds"][4] == 595 and rows["opp_ds"][4] == 590
    assert rows["our_ds"][0] == 600 and rows["opp_ds"][0] == 600
    assert rows["timeout"].all() and rows["base_s"][0] == 60


def test_clock_planes_same_from_rows_and_board():
    rows, _ = game_samples(PGN, 2000)
    board, _ = replay()[4]
    planes = torch.from_numpy(encoding.encode_board_u8(board))[None]
    x_play = planes_u8_to_input(planes, torch.tensor([59.5]), torch.tensor([59.0]),
                                torch.tensor([0.0]), torch.tensor([True]))
    x_rows = samples_to_input(rows[4:5], torch.device("cpu"))
    assert torch.allclose(x_play, x_rows)
    x_drop = samples_to_input(rows[4:5], torch.device("cpu"), clock_dropout=1.0)
    assert float(x_drop[0, encoding.N_PLANES:].abs().sum()) == 0.0


def test_network_shapes():
    net = ChessNetV2(32, 2).eval()
    rows, _ = game_samples(PGN, 2000)
    with torch.no_grad():
        p, wdl, ml = net(samples_to_input(rows, torch.device("cpu")))
    assert p.shape == (20, encoding.N_MOVES) and wdl.shape == (20, 3) and ml.shape == (20,)
    v = ChessNetV2.scalar_value(wdl)
    assert (v.abs() <= 1).all() and (ml >= 0).all()


def test_v2_evaluator_orients_clocks_by_parity():
    from src.bullet import Clock
    from src.inference2 import BatchEvaluatorV2

    seen = []

    class Spy(ChessNetV2):
        def forward(self, x):
            seen.append(x[:, encoding.N_PLANES:encoding.N_PLANES + 2, 0, 0].clone())
            return super().forward(x)

    ev = BatchEvaluatorV2(Spy(16, 1).eval(), "cpu", pad_to=1)
    ev.set_clock(Clock(our_ms=30_000, opp_ms=6_000))
    b = chess.Board()
    mv, idx = encoding.legal_moves_and_indices(b)
    pl = encoding.encode_board_u8(b)
    ev([pl, pl], [idx, idx], [0, 1])
    our_min, opp_min = 30 / 60, 6 / 60
    assert torch.allclose(seen[-1][0], torch.tensor([our_min, opp_min]))
    assert torch.allclose(seen[-1][1], torch.tensor([opp_min, our_min]))   # opponent to move


def test_bullet_engine_with_v2_network():
    from src.bullet import BulletConfig, BulletEngine, Clock
    from src.inference2 import BatchEvaluatorV2

    eng = BulletEngine(BatchEvaluatorV2(ChessNetV2(16, 1).eval(), "cpu"),
                       BulletConfig(ponder=True, overhead_ms=0))
    b = chess.Board()
    try:
        for _ in range(3):
            info = eng.choose(b, Clock(20_000, 20_000))
            assert info.move in b.legal_moves and info.sims > 1
            b.push(info.move)
            b.push(next(iter(b.legal_moves)))
    finally:
        eng.close()


def test_shard_stream_resumes_exactly(tmp_path):
    shards = []
    for k in range(3):
        a = np.zeros(1000, SAMPLE_DTYPE)
        a["move"] = np.arange(1000) + 1000 * k
        np.save(tmp_path / f"s{k}.npy", a)
        shards.append((tmp_path / f"s{k}.npy", 1000))
    s1 = ShardStream(shards, 64, group=2)
    for _ in range(40):                      # crosses a group boundary
        s1.next()
    st = dict(s1.state())
    expect = [s1.next()["move"].copy() for _ in range(5)]
    s2 = ShardStream(shards, 64, group=2)
    s2.load_state(st)
    got = [s2.next()["move"] for _ in range(5)]
    for a, b in zip(expect, got):
        assert (a == b).all()
