"""Replay buffer for generation-2 self-play: an in-memory ring of recent
positions, persisted as compressed chunks (one per batch of games).

Per position:
    states   uint8 (19, 8, 8)   encoding.encode_board_u8 planes
    pidx     int16 (POLICY_K,)  policy target: legal-move indices ...
    pprob    f16   (POLICY_K,)  ... and the Gumbel improved-policy probabilities
    result   int8               game result from the side to move (+1 / 0 / -1)
    plies    int16              plies until the game ended (-1 = unknown: move cap)

Chunks are named `<YYYYmmdd-HHMMSS>-<pid>-<seq>.npz`, so name order is time
order; on start the newest chunks are loaded up to capacity, and old chunks
are deleted once the disk holds clearly more than that.
"""
from __future__ import annotations

import itertools
import logging
import os
import time
from pathlib import Path

import numpy as np

from . import encoding

log = logging.getLogger("replay2")

POLICY_K = 96
FIELDS = {
    "states": ((encoding.N_PLANES, 8, 8), np.uint8),
    "pidx": ((POLICY_K,), np.int16),
    "pprob": ((POLICY_K,), np.float16),
    "result": ((), np.int8),
    "plies": ((), np.int16),
}
_seq = itertools.count()


class ReplayBuffer2:
    def __init__(self, root: str | Path, capacity: int = 2_000_000, load: bool = True,
                 disk_slack: float = 1.25):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity, self.disk_slack = capacity, disk_slack
        self.size = self._pos = 0
        self._mem = load
        if load:
            self.a = {k: np.zeros((capacity, *shape), dtype) for k, (shape, dtype) in FIELDS.items()}
            self._load_existing()

    def chunk_files(self) -> list[Path]:
        return sorted(self.root.glob("*-*-*.npz"))

    def _load_existing(self) -> None:
        chosen, total = [], 0
        for p in reversed(self.chunk_files()):
            if total >= self.capacity:
                break
            try:
                with np.load(p) as d:
                    total += len(d["result"])
                chosen.append(p)
            except Exception as e:
                log.warning(f"skipping unreadable chunk {p.name}: {e}")
        for p in reversed(chosen):
            with np.load(p) as d:
                self._insert({k: d[k] for k in FIELDS})
        log.info(f"replay buffer {self.root}: loaded {self.size} positions from {len(chosen)} chunks")

    def save_games(self, recs: list[dict]) -> int:
        """Persist game records (dicts holding the FIELDS arrays) as one chunk."""
        recs = [r for r in recs if "states" in r and len(r["states"])]
        if not recs:
            return 0
        cat = {k: np.concatenate([r[k] for r in recs]).astype(FIELDS[k][1], copy=False) for k in FIELDS}
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{next(_seq):06d}.npz"
        tmp = self.root / (name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **cat)
        os.replace(tmp, self.root / name)
        if self._mem:
            self._insert(cat)
        self._prune()
        return len(cat["result"])

    def _prune(self) -> None:
        files = self.chunk_files()
        if len(files) < 50:
            return
        sizes = []
        for p in files[-5:]:
            try:
                with np.load(p) as d:
                    sizes.append(len(d["result"]))
            except Exception:
                pass
        per = max(1, int(np.mean(sizes))) if sizes else 1
        keep = int(self.capacity * self.disk_slack / per) + 1
        for p in files[:max(0, len(files) - keep)]:
            try:
                p.unlink()
            except OSError:
                pass

    def _insert(self, cat: dict) -> None:
        n = len(cat["result"])
        if n >= self.capacity:
            cat = {k: v[-self.capacity:] for k, v in cat.items()}
            n = self.capacity
        first = min(n, self.capacity - self._pos)
        for k, dst in self.a.items():
            src = cat[k]
            dst[self._pos:self._pos + first] = src[:first]
            if first < n:
                dst[:n - first] = src[first:]
        self._pos = (self._pos + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict:
        if self.size == 0:
            raise RuntimeError("replay buffer is empty")
        i = np.sort(rng.integers(0, self.size, size=batch_size))
        return {k: v[i] for k, v in self.a.items()}
