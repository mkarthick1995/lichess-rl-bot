import time

import chess
import numpy as np

from src.bullet import BulletConfig, BulletEngine
from src.lichess.bot import (Bot, ChallengePolicy, LichessGame, board_from,
                             challenge_decision, clock_for)


def uniform_eval(planes, idx):
    return [np.full(len(i), 1.0 / len(i), np.float32) for i in idx], np.zeros(len(idx), np.float32)


def make_engine():
    return BulletEngine(uniform_eval, BulletConfig(ponder=False, overhead_ms=0))


def ch(limit=60, inc=0, rated=True, variant="standard", title=None, kind="clock"):
    return {"id": "c1", "rated": rated, "variant": {"key": variant},
            "timeControl": {"type": kind, "limit": limit, "increment": inc, "show": f"{limit}+{inc}"},
            "challenger": {"id": "someone", "name": "Someone", "title": title}}


def test_challenge_decision():
    p = ChallengePolicy()
    assert challenge_decision(ch(60, 0), p, busy=False) is None          # 1+0
    assert challenge_decision(ch(30, 0), p, busy=False) is None          # 30+0 (est 30 s)
    assert challenge_decision(ch(120, 1), p, busy=False) is None         # 2+1 (est 160 s)
    assert challenge_decision(ch(15, 0), p, busy=False) == "tooFast"     # UltraBullet
    assert challenge_decision(ch(180, 2), p, busy=False) == "tooSlow"    # 3+2 blitz
    assert challenge_decision(ch(variant="chess960"), p, busy=False) == "standard"
    assert challenge_decision(ch(kind="unlimited"), p, busy=False) == "timeControl"
    assert challenge_decision(ch(), p, busy=True) == "later"
    assert challenge_decision(ch(title="BOT"), ChallengePolicy(allow_bots=False), False) == "noBot"
    assert challenge_decision(ch(), ChallengePolicy(allow_humans=False), False) == "onlyBot"
    assert challenge_decision(ch(rated=True), ChallengePolicy(allow_rated=False), False) == "casual"


def test_board_and_clock_parsing():
    b = board_from("startpos", "e2e4 e7e5 g1f3")
    assert b.turn == chess.BLACK and len(b.move_stack) == 3
    st = {"wtime": 50_000, "btime": 40_000, "winc": 0, "binc": 1000}
    c = clock_for(st, chess.BLACK)
    assert (c.our_ms, c.opp_ms, c.inc_ms, c.opp_inc_ms) == (40_000, 50_000, 1000, 0)


class FakeAPI:
    """Plays the opponent's side from a script; records our moves. The stream
    runs on the bot's reader thread, so it waits for each of our moves."""

    def __init__(self, my_color, opp_moves, end_after):
        self.my_color = my_color
        self.opp_moves = list(opp_moves)
        self.end_after = end_after           # stop after this many of our moves
        self.sent = []
        self.rejected = []

    def _wait_sent(self, n, timeout=10.0):
        t0 = time.monotonic()
        while len(self.sent) < n:
            if time.monotonic() - t0 > timeout:
                raise TimeoutError("bot did not move")
            time.sleep(0.002)

    def _players(self):
        white = {"id": "me" if self.my_color == "white" else "opp", "name": "x"}
        black = {"id": "opp" if self.my_color == "white" else "me", "name": "y"}
        return white, black

    def stream_game(self, gid):
        white, black = self._players()
        self.moves = moves = []
        state = lambda status="started": {"type": "gameState", "moves": " ".join(moves),
                                          "wtime": 60_000, "btime": 60_000, "winc": 0, "binc": 0,
                                          "status": status}
        if self.my_color == "black":
            moves.append(self.opp_moves.pop(0))
        yield {"type": "gameFull", "id": gid, "white": white, "black": black,
               "initialFen": "startpos", "clock": {"initial": 60000, "increment": 0},
               "state": state()}
        n = 0
        while n < self.end_after:
            self._wait_sent(n + 1)
            n += 1
            moves.append(self.sent[-1])
            yield state()                     # state after our move (opponent to move)
            moves.append(self.opp_moves.pop(0))
            yield {"type": "chatLine", "text": "hi"}
            yield state()
        final = state("mate")
        final["winner"] = self.my_color
        yield final

    def make_move(self, gid, uci):
        board = board_from("startpos", " ".join(self.moves))
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            self.rejected.append(uci)
            return False
        self.sent.append(uci)
        return True


def _legal_replies():
    # a sequence of opponent moves that stays legal whatever we play is hard to
    # script, so the opponent shuffles a knight: Nf6/Ng8 (black) or Nf3/Ng1 (white)
    return ["g8f6", "f6g8"] * 5, ["g1f3", "f3g1"] * 5


def test_game_loop_plays_one_move_per_turn_as_white(tmp_path, monkeypatch):
    monkeypatch.setattr("src.lichess.bot.PGN_FILE", tmp_path / "games.pgn")
    black_moves, _ = _legal_replies()
    api = FakeAPI("white", black_moves, end_after=4)
    beats = []
    LichessGame(api, "g1", "me", make_engine, heartbeat=lambda: beats.append(1)).run()
    assert len(api.sent) == 4 and len(beats) == 4
    board = chess.Board()
    for ours, theirs in zip(api.sent, black_moves):
        assert chess.Move.from_uci(ours) in board.legal_moves
        board.push_uci(ours)
        board.push_uci(theirs)
    pgn = (tmp_path / "games.pgn").read_text()
    assert '[Result "1-0"]' in pgn and "lichess.org/g1" in pgn


def test_game_loop_as_black(tmp_path, monkeypatch):
    monkeypatch.setattr("src.lichess.bot.PGN_FILE", tmp_path / "games.pgn")
    _, white_moves = _legal_replies()
    api = FakeAPI("black", white_moves, end_after=3)
    LichessGame(api, "g2", "me", make_engine).run()
    assert len(api.sent) == 3
    assert '[Result "0-1"]' in (tmp_path / "games.pgn").read_text()


class BrowserMoveAPI(FakeAPI):
    """While the bot thinks about move 2, 'someone in a browser' plays a move
    for it and the opponent replies — the bot's move must be discarded, not
    sent stale, and it must warn."""

    def stream_game(self, gid):
        white, black = self._players()
        self.moves = moves = []
        st = lambda: {"type": "gameState", "moves": " ".join(moves), "wtime": 60_000,
                      "btime": 60_000, "winc": 0, "binc": 0, "status": "started"}
        yield {"type": "gameFull", "id": gid, "white": white, "black": black,
               "initialFen": "startpos", "clock": {"initial": 60000, "increment": 0}, "state": st()}
        self._wait_sent(1)
        moves.append(self.sent[0])
        moves.append("e7e5")
        yield st()                            # our turn again (move 2)...
        time.sleep(0.05)                      # ...the bot is thinking...
        legal = lambda: [m.uci() for m in board_from("startpos", " ".join(moves)).legal_moves]
        moves.append(legal()[-1])             # ...a browser plays for us
        moves.append(legal()[-1])             # and the opponent replies
        yield st()
        self._wait_sent(2)                    # the bot answers the NEW position
        moves.append(self.sent[1])
        final = st()
        final["status"], final["winner"] = "resign", "white"
        yield final


class SlowEngine:
    def __init__(self):
        self.e = make_engine()

    def choose(self, board, clock):
        time.sleep(0.2)
        return self.e.choose(board, clock)

    def close(self):
        self.e.close()


def test_stale_move_is_discarded_and_foreign_move_warned(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("src.lichess.bot.PGN_FILE", tmp_path / "games.pgn")
    api = BrowserMoveAPI("white", [], end_after=0)
    g = LichessGame(api, "g3", "me", SlowEngine)
    with caplog.at_level("WARNING"):
        g.run()
    assert api.rejected == []                # nothing stale was sent
    assert len(api.sent) == 2
    assert g.stats["discarded"] == 1
    assert "NOT sent by this bot" in caplog.text


class EventAPI:
    def __init__(self):
        self.accepted, self.declined = [], []

    def accept(self, cid):
        self.accepted.append(cid)

    def decline(self, cid, reason):
        self.declined.append((cid, reason))


def test_bot_accepts_and_declines():
    api = EventAPI()
    bot = Bot(api, "me", make_engine, ChallengePolicy(), pause_training=False)
    bot.handle_event({"type": "challenge", "challenge": ch(60, 0)})
    bot.handle_event({"type": "challenge", "challenge": {**ch(15, 0), "id": "c2"}})
    own = {**ch(60, 0), "id": "c3", "challenger": {"id": "me", "name": "Me"}}
    bot.handle_event({"type": "challenge", "challenge": own})
    assert api.accepted == ["c1"]
    assert api.declined == [("c2", "tooFast")]
