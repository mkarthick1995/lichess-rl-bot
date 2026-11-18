"""Supervised data from the Lichess open database (https://database.lichess.org).

Two resumable steps:

    # 1) download the first N GB of a monthly dump (parallel ranges, resumable)
    python -m src.lichess_db download --month 2026-10 --gb 4

    # 2) extract strong bullet games into compact position shards
    python -m src.lichess_db extract --month 2026-10 --min-elo 2000

Games kept: rated bullet (estimated duration 30-179 s, i.e. Lichess "Bullet"),
both players >= --min-elo, normal or time-forfeit endings. Every position of
a kept game becomes one sample with the human move (policy target), the
final result from the side to move (WDL target), the plies left in the game
(moves-left target) and both clocks (Lichess records them for every move).

Samples are numpy structured arrays (SAMPLE_DTYPE, ~120 bytes each) in
data/positions/<month>/shard_XXXXX.npy, plus a manifest so extraction can be
resumed; games are assigned to train/val by a hash of their URL.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Iterator, Optional

import chess
import numpy as np

from ._log import setup_logging
from . import encoding

log = logging.getLogger("lichess_db")

DATA = Path("data")
URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
PIECE = 32 * 1024 * 1024

SAMPLE_DTYPE = np.dtype([
    ("bb", "<u8", (12,)),        # piece bitboards, side-to-move frame (own P..K, opp P..K)
    ("flags", "u1"),             # bit0-3 castling (own K, own Q, opp K, opp Q), bit4 repetition
    ("halfmove", "u1"),          # halfmove clock, capped at 100
    ("move", "<i2"),             # policy index of the human move (STM frame)
    ("result", "i1"),            # final result from STM: +1 win, 0 draw, -1 loss
    ("plies_left", "<i2"),       # plies until the game ended (incl. this move)
    ("our_ds", "<u2"),           # our clock before moving, deciseconds
    ("opp_ds", "<u2"),           # opponent's clock, deciseconds
    ("base_s", "<u2"),           # time control base, seconds
    ("inc_s", "u1"),             # increment, seconds
    ("our_elo", "<u2"),
    ("opp_elo", "<u2"),
    ("timeout", "u1"),           # game ended on time
])


# ---------------------------------------------------------------------------
# 1) download

def _paths(month: str) -> tuple[Path, Path]:
    d = DATA / "lichess"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{month}.pgn.zst.part", d / f"{month}.download.json"


def download(month: str, gb: float, workers: int = 4) -> Path:
    import requests

    url = URL.format(month=month)
    total = int(gb * 1e9) // PIECE * PIECE
    path, man_path = _paths(month)
    man = json.loads(man_path.read_text()) if man_path.exists() else {"url": url, "done": []}
    done = set(man["done"])
    pieces = [i for i in range(total // PIECE) if i not in done]
    log.info(f"{url}: {len(done)} pieces done, {len(pieces)} to go "
             f"({len(pieces) * PIECE / 1e9:.1f} GB, {workers} connections)")
    if not path.exists():
        with open(path, "wb"):
            pass
    lock = threading.Lock()
    t0, got = time.monotonic(), 0

    def fetch(i: int) -> int:
        start = i * PIECE
        for attempt in range(8):
            try:
                r = requests.get(url, headers={"Range": f"bytes={start}-{start + PIECE - 1}"},
                                 timeout=60)
                if r.status_code != 206 or len(r.content) != PIECE:
                    raise IOError(f"status {r.status_code}, {len(r.content)} bytes")
                with lock:
                    with open(path, "r+b") as f:
                        f.seek(start)
                        f.write(r.content)
                return i
            except Exception as e:
                log.warning(f"piece {i}: {e}; retry {attempt + 1}")
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"piece {i} failed")

    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch, i) for i in pieces]
        for k, fu in enumerate(as_completed(futs), 1):
            i = fu.result()
            with lock:
                done.add(i)
                man["done"] = sorted(done)
                man_path.write_text(json.dumps(man))
            got += PIECE
            if k % 8 == 0 or k == len(pieces):
                rate = got / 1e6 / (time.monotonic() - t0)
                left = (len(pieces) - k) * PIECE / 1e6 / max(rate, 1e-6)
                log.info(f"{len(done)}/{total // PIECE} pieces  {rate:.1f} MB/s  ~{left / 60:.0f} min left")
    return path


def contiguous_bytes(month: str) -> int:
    """Length of the fully downloaded prefix."""
    _, man_path = _paths(month)
    if not man_path.exists():
        return 0
    done = set(json.loads(man_path.read_text())["done"])
    n = 0
    while n in done:
        n += 1
    return n * PIECE


# ---------------------------------------------------------------------------
# 2) extract

def iter_games(month: str) -> Iterator[str]:
    """Raw PGN text of each game in the downloaded prefix (the last, cut-off
    game is dropped)."""
    import zstandard

    path, _ = _paths(month)
    limit = contiguous_bytes(month)

    class Prefix(io.RawIOBase):
        def __init__(self):
            self.f = open(path, "rb")
            self.left = limit

        def readable(self):
            return True

        def readinto(self, b):
            n = min(len(b), self.left)
            if n <= 0:
                return 0
            data = self.f.read(n)
            b[:len(data)] = data
            self.left -= len(data)
            return len(data)

    reader = zstandard.ZstdDecompressor().stream_reader(Prefix(), read_across_frames=True)
    text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    game: list[str] = []
    try:
        for line in text:
            if line.startswith("[Event ") and game:
                yield "".join(game)
                game = []
            game.append(line)
    except zstandard.ZstdError:
        pass                                  # truncated prefix: drop the partial game


_HDR = re.compile(r'^\[(\w+) "(.*)"\]$', re.M)
_CLK = re.compile(r"\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]")
_COMMENT = re.compile(r"\{[^}]*\}")
_MOVENUM = re.compile(r"\d+\.(?:\.\.)?")


def keep_game(h: dict, min_elo: int) -> bool:
    if "Bullet" not in h.get("Event", "") or "Rated" not in h.get("Event", ""):
        return False
    if h.get("Termination") not in ("Normal", "Time forfeit"):
        return False
    if h.get("Result") not in ("1-0", "0-1", "1/2-1/2"):
        return False
    try:
        if min(int(h["WhiteElo"]), int(h["BlackElo"])) < min_elo:
            return False
        base, inc = (int(x) for x in h["TimeControl"].split("+"))
    except (KeyError, ValueError):
        return False
    return 30 <= base + 40 * inc <= 179


def game_samples(pgn: str, min_elo: int) -> Optional[tuple[np.ndarray, bool]]:
    """All positions of one kept game as SAMPLE_DTYPE rows, plus its split
    (True = validation). None if the game is filtered out or broken."""
    head, _, movetext = pgn.partition("\n\n")
    h = dict(_HDR.findall(head))
    if not keep_game(h, min_elo):
        return None
    clocks = [round(float(hh) * 36000 + float(mm) * 600 + float(ss) * 10)
              for hh, mm, ss in _CLK.findall(movetext)]
    sans = [t for t in _MOVENUM.sub(" ", _COMMENT.sub(" ", movetext)).split()
            if t not in ("1-0", "0-1", "1/2-1/2", "*")]
    if len(sans) < 10 or len(clocks) != len(sans):
        return None
    base, inc = (int(x) for x in h["TimeControl"].split("+"))
    res_white = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[h["Result"]]
    elo = {chess.WHITE: int(h["WhiteElo"]), chess.BLACK: int(h["BlackElo"])}
    timeout = h["Termination"] == "Time forfeit"

    rows = np.zeros(len(sans), dtype=SAMPLE_DTYPE)
    board = chess.Board()
    clk = {chess.WHITE: base * 10, chess.BLACK: base * 10}
    n = len(sans)
    try:
        for i, san in enumerate(sans):
            us = board.turn
            move = board.parse_san(san)
            own, opp = board.occupied_co[us], board.occupied_co[not us]
            bbs = [board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings]
            pieces = [bb & own for bb in bbs] + [bb & opp for bb in bbs]
            if us == chess.BLACK:
                pieces = [chess.flip_vertical(bb) for bb in pieces]
            r = rows[i]
            r["bb"] = pieces
            r["flags"] = (board.has_kingside_castling_rights(us)
                          | board.has_queenside_castling_rights(us) << 1
                          | board.has_kingside_castling_rights(not us) << 2
                          | board.has_queenside_castling_rights(not us) << 3
                          | encoding.is_repetition(board, 2) << 4)
            r["halfmove"] = min(board.halfmove_clock, 100)
            r["move"] = encoding.move_to_index(move, board)
            r["result"] = res_white if us == chess.WHITE else -res_white
            r["plies_left"] = n - i
            r["our_ds"] = min(clk[us], 65535)
            r["opp_ds"] = min(clk[not us], 65535)
            r["base_s"], r["inc_s"] = base, min(inc, 255)
            r["our_elo"], r["opp_elo"] = elo[us], elo[not us]
            r["timeout"] = timeout
            board.push(move)
            clk[us] = clocks[i]
    except (ValueError, AssertionError):
        return None
    val = int(hashlib.md5(h.get("Site", pgn[:200]).encode()).hexdigest(), 16) % 100 < 2
    return rows, val


def _work(args):
    batch, min_elo = args
    out_t, out_v, kept = [], [], 0
    for pgn in batch:
        r = game_samples(pgn, min_elo)
        if r is None:
            continue
        rows, val = r
        (out_v if val else out_t).append(rows)
        kept += 1
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, SAMPLE_DTYPE)
    return cat(out_t), cat(out_v), kept, len(batch)


def extract(month: str, min_elo: int, procs: int = 7, shard_size: int = 1_000_000,
            batch_games: int = 2000) -> None:
    out = DATA / "positions" / month
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {
        "games_scanned": 0, "games_kept": 0, "shards": [], "val_shards": [], "min_elo": min_elo}
    skip = man["games_scanned"]
    log.info(f"extracting {month} (min elo {min_elo}); resuming after {skip} scanned games")

    buf_t, buf_v = [], []
    n_t = n_v = 0

    def flush(split: str, force: bool = False):
        nonlocal buf_t, buf_v, n_t, n_v
        buf, n = (buf_t, n_t) if split == "train" else (buf_v, n_v)
        if not buf or (n < shard_size and not force):
            return
        arr = np.concatenate(buf)
        key = "shards" if split == "train" else "val_shards"
        name = f"{'shard' if split == 'train' else 'val'}_{len(man[key]):05d}.npy"
        np.save(out / name, arr)
        man[key].append({"file": name, "n": int(len(arr))})
        if split == "train":
            buf_t, n_t = [], 0
        else:
            buf_v, n_v = [], 0

    def batches():
        b, i = [], 0
        for pgn in iter_games(month):
            i += 1
            if i <= skip:
                continue
            b.append(pgn)
            if len(b) == batch_games:
                yield b
                b = []
        if b:
            yield b

    t0 = time.monotonic()
    scanned_since_save = 0
    with get_context("spawn").Pool(procs) as pool:
        for tr, va, kept, scanned in pool.imap(_work, ((b, min_elo) for b in batches()), chunksize=1):
            if len(tr):
                buf_t.append(tr); n_t += len(tr)
            if len(va):
                buf_v.append(va); n_v += len(va)
            man["games_kept"] += kept
            man["games_scanned"] += scanned
            scanned_since_save += scanned
            if n_t >= shard_size:
                flush("train")
                flush("val", force=True)       # keep the manifest's resume point consistent
                man_path.write_text(json.dumps(man, indent=1))
                scanned_since_save = 0
                pos = sum(s["n"] for s in man["shards"])
                log.info(f"scanned {man['games_scanned']:,} games, kept {man['games_kept']:,}; "
                         f"{len(man['shards'])} train shards ({pos:,} positions); "
                         f"{man['games_scanned'] / (time.monotonic() - t0):,.0f} games/s")
    flush("train", force=True)
    flush("val", force=True)
    man["complete_bytes"] = contiguous_bytes(month)
    man_path.write_text(json.dumps(man, indent=1))
    pos = sum(s["n"] for s in man["shards"])
    vpos = sum(s["n"] for s in man["val_shards"])
    log.info(f"done: kept {man['games_kept']:,} of {man['games_scanned']:,} games; "
             f"{pos:,} train + {vpos:,} val positions in {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--month", default="2026-10")
    d.add_argument("--gb", type=float, default=4.0)
    d.add_argument("--workers", type=int, default=4)
    e = sub.add_parser("extract")
    e.add_argument("--month", default="2026-10")
    e.add_argument("--min-elo", type=int, default=2000)
    e.add_argument("--procs", type=int, default=7)
    args = ap.parse_args()
    setup_logging(f"lichess_db-{args.cmd}", "INFO")
    if args.cmd == "download":
        download(args.month, args.gb, args.workers)
    else:
        extract(args.month, args.min_elo, args.procs)


if __name__ == "__main__":
    main()
