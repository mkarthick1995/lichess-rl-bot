"""Play bullet on Lichess as a BOT account, through the official Bot API.

    # one-time: turn a NEW account (no games played) into a BOT — permanent
    python -m src.lichess.bot --upgrade-account --yes

    # accept bullet challenges (humans and bots), one game at a time
    python -m src.lichess.bot

    # also challenge random online bots while idle
    python -m src.lichess.bot --matchmaking --tc 1+0

    # challenge one player / bot a few times
    python -m src.lichess.bot --challenge SomeBot --tc 1+0 --games 3

While a game is running the bot creates logs/PAUSE, which the training loop
honours (self-play and training pause), so the GPU is ours and moves are fast.
Finished games are appended to logs/bot-games.pgn.

The bot never resigns: in bullet a lost position with more time on the clock
is often a win on time.
"""
from __future__ import annotations

import argparse
import logging
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chess
import chess.pgn
import torch

from .._log import setup_logging
from ..bullet import BulletConfig, BulletEngine, Clock
from ..inference2 import make_evaluator
from .bot_api import LichessAPI, load_token

log = logging.getLogger("lichess.bot")

PAUSE_FILE = Path("logs") / "PAUSE"
PGN_FILE = Path("logs") / "bot-games.pgn"


# ---------------------------------------------------------------------------
# Challenges

@dataclass
class ChallengePolicy:
    min_est_s: int = 30          # Lichess estimate = limit + 40 * increment;
    max_est_s: int = 179         # 30..179 s is "bullet" (bots can't play UltraBullet)
    allow_rated: bool = True
    allow_casual: bool = True
    allow_bots: bool = True
    allow_humans: bool = True


def challenge_decision(ch: dict, policy: ChallengePolicy, busy: bool) -> Optional[str]:
    """None to accept, else a Lichess decline reason."""
    if (ch.get("variant") or {}).get("key", "standard") != "standard":
        return "standard"
    tc = ch.get("timeControl") or {}
    if tc.get("type") != "clock":
        return "timeControl"
    est = int(tc.get("limit", 0)) + 40 * int(tc.get("increment", 0))
    if est < policy.min_est_s:
        return "tooFast"
    if est > policy.max_est_s:
        return "tooSlow"
    if ch.get("rated") and not policy.allow_rated:
        return "casual"
    if not ch.get("rated") and not policy.allow_casual:
        return "rated"
    is_bot = (ch.get("challenger") or {}).get("title") == "BOT"
    if is_bot and not policy.allow_bots:
        return "noBot"
    if not is_bot and not policy.allow_humans:
        return "onlyBot"
    if busy:
        return "later"
    return None


# ---------------------------------------------------------------------------
# One game

def board_from(initial_fen: str, moves: str) -> chess.Board:
    board = chess.Board() if initial_fen in ("", "startpos", None) else chess.Board(initial_fen)
    for u in moves.split():
        board.push_uci(u)
    return board


def clock_for(state: dict, color: chess.Color) -> Clock:
    w, b = int(state.get("wtime", 0)), int(state.get("btime", 0))
    wi, bi = int(state.get("winc", 0)), int(state.get("binc", 0))
    if color == chess.WHITE:
        return Clock(w, b, wi, bi)
    return Clock(b, w, bi, wi)


class LichessGame:
    def __init__(self, api: LichessAPI, game_id: str, my_id: str, make_engine,
                 heartbeat=lambda: None):
        self.api = api
        self.heartbeat = heartbeat
        self.id = game_id
        self.my_id = my_id.lower()
        self.make_engine = make_engine
        self.full: dict = {}
        self.color: Optional[chess.Color] = None
        self.initial_fen = "startpos"
        self.moved_at = -1           # ply count at which we last moved
        self.final_state: dict = {}
        self.stats = {"moves": 0, "think_ms": 0.0, "sims": 0, "discarded": 0}
        self.sent: dict[int, str] = {}   # ply → uci we sent
        self.foreign_warned = False
        self._first_our_ply_seen: Optional[int] = None
        # The stream is read on its own thread; the game loop always works on
        # the NEWEST state, never on a backlog of stale ones.
        self._cv = threading.Condition()
        self._latest: Optional[dict] = None
        self._seq = 0
        self._stream_done = False

    def _reader(self) -> None:
        try:
            for ev in self.api.stream_game(self.id):
                t = ev.get("type")
                if t == "gameFull":
                    self.full = ev
                    white_id = (ev.get("white") or {}).get("id", "")
                    self.color = chess.WHITE if white_id.lower() == self.my_id else chess.BLACK
                    self.initial_fen = ev.get("initialFen", "startpos")
                    state = ev.get("state", {})
                elif t == "gameState":
                    state = ev
                else:
                    continue                  # chatLine, opponentGone, ...
                with self._cv:
                    self._latest, self._seq = state, self._seq + 1
                    self._cv.notify_all()
                if state.get("status", "started") not in ("created", "started"):
                    break
        except Exception:
            log.exception(f"[{self.id}] game stream failed")
        finally:
            with self._cv:
                self._stream_done = True
                self._cv.notify_all()

    def run(self) -> None:
        engine = self.make_engine()
        log.info(f"[{self.id}] game start  https://lichess.org/{self.id}")
        reader = threading.Thread(target=self._reader, name=f"stream-{self.id}", daemon=True)
        reader.start()
        seen = 0
        try:
            while True:
                with self._cv:
                    while self._seq == seen and not self._stream_done:
                        self._cv.wait(timeout=1.0)
                    if self._seq == seen:
                        break                 # stream ended, nothing new
                    st, seen = self._latest, self._seq
                if not self._on_state(st, engine):
                    break
        except Exception:
            log.exception(f"[{self.id}] game loop failed")
        finally:
            engine.close()
            self._save_pgn()

    def _check_foreign_moves(self, moves: list[str]) -> None:
        """Warn if a move for our side appears that this bot did not send —
        e.g. someone moving pieces in a browser logged in as the bot."""
        first = 0 if self.color == chess.WHITE else 1
        for ply in range(first, len(moves), 2):
            if self.foreign_warned:
                return
            if ply not in self.sent and ply >= self._first_our_ply_seen:
                log.warning(f"[{self.id}] move {moves[ply]} at ply {ply + 1} was played for us but "
                            f"NOT sent by this bot — is someone moving in a browser logged in "
                            f"as the bot, or another bot instance running?")
                self.foreign_warned = True
            elif ply in self.sent and self.sent[ply] != moves[ply]:
                log.warning(f"[{self.id}] ply {ply + 1}: server has {moves[ply]}, we sent {self.sent[ply]}")
                self.foreign_warned = True

    def _on_state(self, st: dict, engine: BulletEngine) -> bool:
        status = st.get("status", "started")
        if status not in ("created", "started"):
            self.final_state = st
            return False
        moves = st.get("moves", "").split()
        if self._first_our_ply_seen is None:
            # moves already on the board when we joined are not "foreign"
            self._first_our_ply_seen = len(moves)
        self._check_foreign_moves(moves)
        board = board_from(self.initial_fen, st.get("moves", ""))
        if board.turn != self.color or board.is_game_over(claim_draw=False):
            return True
        ply = len(board.move_stack)
        if ply == self.moved_at:
            return True                       # same position again (draw offer etc.)
        clock = clock_for(st, self.color)
        info = engine.choose(board, clock)
        with self._cv:
            newest = self._latest
        if newest is not None and newest.get("moves", "") != st.get("moves", ""):
            # The position changed while we were thinking (a move we didn't
            # make, or the game moved on): sending now would be stale.
            self.stats["discarded"] += 1
            log.warning(f"[{self.id}] position changed while thinking; discarded {info.move.uci()}")
            return True
        ok = self.api.make_move(self.id, info.move.uci())
        self.moved_at = ply
        if ok:
            self.sent[ply] = info.move.uci()
        self.heartbeat()
        self.stats["moves"] += 1
        self.stats["think_ms"] += info.spent_ms
        self.stats["sims"] += info.sims
        log.info(f"[{self.id}] {board.fullmove_number}{'.' if board.turn else '...'} {board.san(info.move)}"
                 f"  {info.mode:7s} think {info.think_ms:4d} ms, spent {info.spent_ms:5.0f} ms, "
                 f"sims {info.sims:5d} (reused {info.reused}), value {info.value:+.2f}  "
                 f"clock {clock.our_ms / 1000:.1f}s vs {clock.opp_ms / 1000:.1f}s"
                 + (f"  filtered {info.filtered}" if info.filtered else "")
                 + ("" if ok else "  [MOVE REJECTED]"))
        return True

    def _save_pgn(self) -> None:
        st = self.final_state or (self.full.get("state") or {})
        try:
            board = board_from(self.initial_fen, st.get("moves", ""))
        except Exception:
            return
        game = chess.pgn.Game.from_board(board)
        w, b = self.full.get("white") or {}, self.full.get("black") or {}
        clk = self.full.get("clock") or {}
        winner = st.get("winner")
        result = "1-0" if winner == "white" else "0-1" if winner == "black" else (
            "1/2-1/2" if st.get("status") in ("draw", "stalemate", "outoftime", "timeout") and not winner else "*")
        game.headers.update({
            "Event": f"Lichess {'rated' if self.full.get('rated') else 'casual'} {self.full.get('speed', '')}",
            "Site": f"https://lichess.org/{self.id}",
            "Date": time.strftime("%Y.%m.%d"),
            "White": w.get("name") or w.get("id", "?"), "Black": b.get("name") or b.get("id", "?"),
            "WhiteElo": str(w.get("rating", "?")), "BlackElo": str(b.get("rating", "?")),
            "TimeControl": f"{clk.get('initial', 0) // 1000}+{clk.get('increment', 0) // 1000}",
            "Result": result, "Termination": st.get("status", "?"),
        })
        PGN_FILE.parent.mkdir(exist_ok=True)
        with open(PGN_FILE, "a", encoding="utf-8") as f:
            print(game, file=f, end="\n\n")
        n = max(1, self.stats["moves"])
        mine = "white" if self.color == chess.WHITE else "black"
        verdict = "won" if winner == mine else "lost" if winner else "drew/ended"
        log.info(f"[{self.id}] game over: {st.get('status')} winner={winner} → we {verdict}; "
                 f"{self.stats['moves']} moves, avg {self.stats['think_ms'] / n:.0f} ms/move, "
                 f"avg {self.stats['sims'] / n:.0f} sims")


# ---------------------------------------------------------------------------
# Bot

class Bot:
    def __init__(self, api: LichessAPI, my_id: str, make_engine, policy: ChallengePolicy,
                 pause_training: bool = True):
        self.api = api
        self.my_id = my_id
        self.make_engine = make_engine
        self.policy = policy
        self.pause_training = pause_training
        self.games: dict[str, threading.Thread] = {}
        self.lock = threading.Lock()
        self.pending_challenge: Optional[str] = None   # our outgoing challenge id
        self.games_played = 0

    def busy(self) -> bool:
        with self.lock:
            self.games = {k: t for k, t in self.games.items() if t.is_alive()}
            return bool(self.games)

    def _set_pause(self, on: bool) -> None:
        if not self.pause_training:
            return
        try:
            if on:
                PAUSE_FILE.parent.mkdir(exist_ok=True)
                PAUSE_FILE.touch()
            else:
                PAUSE_FILE.unlink(missing_ok=True)
        except OSError as e:
            log.warning(f"could not {'create' if on else 'remove'} {PAUSE_FILE}: {e}")

    def _play(self, game_id: str) -> None:
        self._set_pause(True)
        try:
            LichessGame(self.api, game_id, self.my_id, self.make_engine,
                        heartbeat=lambda: self._set_pause(True)).run()
        finally:
            self.games_played += 1
            if not self.busy_except(game_id):
                self._set_pause(False)

    def busy_except(self, game_id: str) -> bool:
        with self.lock:
            return any(t.is_alive() for k, t in self.games.items() if k != game_id)

    def handle_event(self, ev: dict) -> None:
        t = ev.get("type")
        if t == "challenge":
            ch = ev["challenge"]
            if (ch.get("challenger") or {}).get("id", "").lower() == self.my_id.lower():
                return                        # our own outgoing challenge
            reason = challenge_decision(ch, self.policy, self.busy())
            who = (ch.get("challenger") or {}).get("name", "?")
            tc = (ch.get("timeControl") or {}).get("show", "?")
            if reason is None:
                log.info(f"accepting challenge {ch['id']} from {who} ({tc}, "
                         f"{'rated' if ch.get('rated') else 'casual'})")
                self.api.accept(ch["id"])
            else:
                log.info(f"declining challenge {ch['id']} from {who} ({tc}): {reason}")
                self.api.decline(ch["id"], reason)
        elif t == "gameStart":
            gid = ev["game"].get("gameId") or ev["game"].get("id")
            self.pending_challenge = None
            with self.lock:
                if gid in self.games and self.games[gid].is_alive():
                    return
                th = threading.Thread(target=self._play, args=(gid,), name=f"game-{gid}", daemon=True)
                self.games[gid] = th
            th.start()
        elif t in ("challengeDeclined", "challengeCanceled"):
            cid = (ev.get("challenge") or {}).get("id")
            if cid and cid == self.pending_challenge:
                log.info(f"our challenge {cid} was {t[9:].lower()}")
                self.pending_challenge = None

    def run_events(self, stop: threading.Event) -> None:
        backoff = 1
        while not stop.is_set():
            try:
                for ev in self.api.stream_events():
                    backoff = 1
                    self.handle_event(ev)
                    if stop.is_set():
                        break
            except Exception as e:
                log.warning(f"event stream error: {e}; reconnecting in {backoff}s")
                time.sleep(backoff)
                backoff = min(60, backoff * 2)

    # -- outgoing challenges --------------------------------------------------
    def challenge_and_wait(self, user: str, limit_s: int, inc_s: int, rated: bool,
                           wait_s: float = 40.0) -> bool:
        cid = self.api.challenge(user, limit_s, inc_s, rated)
        if not cid:
            return False
        self.pending_challenge = cid
        log.info(f"challenged {user} ({limit_s // 60 if limit_s >= 60 else limit_s / 60:g}+{inc_s}, "
                 f"{'rated' if rated else 'casual'}) → {cid}")
        t0 = time.monotonic()
        while time.monotonic() - t0 < wait_s:
            if self.busy():
                return True
            if self.pending_challenge != cid:
                return False                  # declined / canceled
            time.sleep(0.5)
        self.api.cancel_challenge(cid)
        self.pending_challenge = None
        log.info(f"challenge {cid} not accepted in {wait_s:.0f}s; cancelled")
        return False

    def wait_idle(self, stop: threading.Event) -> None:
        while self.busy() and not stop.is_set():
            time.sleep(1.0)


def _parse_tc(tc: str) -> tuple[int, int]:
    base, _, inc = tc.partition("+")
    b = float(base)
    return (int(b * 60) if b < 10 else int(b)), int(float(inc or 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None,
                    help="network to play with (default: checkpoints/v2/bot.pt, the network that "
                         "passed the latest bullet gate, else checkpoints/best.pt); reloaded "
                         "between games when the file changes")
    ap.add_argument("--token-file", default=".lichess-token")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--upgrade-account", action="store_true",
                    help="turn the token's account into a BOT account (PERMANENT; needs --yes)")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--tc", default="1+0", help="time control for outgoing challenges (min+inc)")
    ap.add_argument("--casual", action="store_true", help="outgoing challenges are casual")
    ap.add_argument("--challenge", help="username to challenge")
    ap.add_argument("--games", type=int, default=1, help="games to play with --challenge")
    ap.add_argument("--matchmaking", action="store_true", help="challenge random online bots while idle")
    ap.add_argument("--mm-interval", type=float, default=30.0, help="seconds idle before challenging")
    ap.add_argument("--no-humans", action="store_true", help="decline challenges from humans")
    ap.add_argument("--no-bots", action="store_true", help="decline challenges from bots")
    ap.add_argument("--no-pause-training", action="store_true",
                    help="don't create logs/PAUSE during games")
    ap.add_argument("--overhead-ms", type=int, default=150, help="per-move latency allowance")
    args = ap.parse_args()

    setup_logging("bot", "INFO")
    api = LichessAPI(load_token(args.token_file))
    acct = api.account()
    my_id, title = acct.get("id", "?"), acct.get("title")
    log.info(f"logged in as {acct.get('username')} (title={title})")

    if args.upgrade_account:
        if title == "BOT":
            log.info("already a BOT account; nothing to do")
            return
        if not args.yes:
            raise SystemExit("Upgrading to BOT is permanent and only works for an account with no "
                             "games played. Re-run with --upgrade-account --yes to proceed.")
        ok = api.upgrade_to_bot()
        log.info("upgraded to BOT" if ok else "upgrade FAILED (see warning above)")
        return
    if title != "BOT":
        raise SystemExit(f"{acct.get('username')} is not a BOT account. Run with "
                         "--upgrade-account --yes first (new accounts only).")

    torch.backends.cudnn.benchmark = True
    dev = torch.device(args.device)
    ckpt = Path(args.checkpoint or ("checkpoints/v2/bot.pt" if Path("checkpoints/v2/bot.pt").exists()
                                    else "checkpoints/best.pt"))
    cfg = BulletConfig(overhead_ms=args.overhead_ms)
    net = {"mtime": None, "ev": None}

    def make_engine():
        """New engine per game; picks up a newly gated network between games."""
        try:
            mtime = ckpt.stat().st_mtime
            if mtime != net["mtime"]:
                ev = make_evaluator(ckpt, dev, args.channels, args.blocks)
                # warm up cuDNN so the first move isn't slow
                BulletEngine(ev, BulletConfig(ponder=False)).choose(chess.Board(), Clock(60_000, 60_000))
                net.update(mtime=mtime, ev=ev)
                log.info(f"network: {ckpt} ({type(ev).__name__}, modified "
                         f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))})")
        except Exception as e:                  # mid-copy or unreadable: keep the current one
            if net["ev"] is None:
                raise
            log.warning(f"could not reload {ckpt} ({e}); keeping the current network")
        return BulletEngine(net["ev"], cfg)

    make_engine().close()

    policy = ChallengePolicy(allow_humans=not args.no_humans, allow_bots=not args.no_bots)
    bot = Bot(api, my_id, make_engine, policy, pause_training=not args.no_pause_training)
    stop = threading.Event()
    threading.Thread(target=bot.run_events, args=(stop,), name="events", daemon=True).start()
    limit_s, inc_s = _parse_tc(args.tc)
    log.info(f"ready: accepting bullet challenges ({policy.min_est_s}-{policy.max_est_s} s estimated)")

    try:
        if args.challenge:
            played = 0
            while played < args.games:
                bot.wait_idle(stop)
                if bot.challenge_and_wait(args.challenge, limit_s, inc_s, not args.casual):
                    played += 1
                    bot.wait_idle(stop)
                else:
                    time.sleep(10)
            log.info(f"played {played} game(s) against {args.challenge}")
            return
        idle_since = time.monotonic()
        while True:
            time.sleep(1.0)
            if bot.busy():
                idle_since = time.monotonic()
                continue
            if args.matchmaking and time.monotonic() - idle_since > args.mm_interval:
                bots = [b for b in api.online_bots() if b.get("id", "").lower() != my_id.lower()]
                if bots:
                    target = random.choice(bots)
                    bot.challenge_and_wait(target["id"], limit_s, inc_s, not args.casual)
                idle_since = time.monotonic()
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        stop.set()
        PAUSE_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
