"""Replay buffer: an in-memory ring of recent positions, persisted as chunks.

Positions are stored compactly so ~1.5M fit in a few GB of RAM:
    states  uint8  (19, 8, 8)    encoding.encode_board_u8 planes
    pidx    int16  (POLICY_K,)   policy target: legal-move indices …
    pprob   f16    (POLICY_K,)   … and their visit probabilities (0-padded)
    values  f32                  game result from the side-to-move POV

On disk each chunk is one .npz holding several games, named
`<YYYYmmdd-HHMMSS>-<pid>-<seq>.npz` so lexical order is time order. Any
process may write chunks (the trainer's workers, the Lichess farm); the
trainer picks up new files with `refresh()`.
"""
from __future__ import annotations

import itertools
import logging
import os
import time
from pathlib import Path

import numpy as np

from . import encoding

log = logging.getLogger("replay_buffer")

POLICY_K = 96   # max policy entries kept per position (top-K by visits)
_seq = itertools.count()


class ReplayBuffer:
    def __init__(self, root: str | Path, capacity: int = 1_500_000, load: bool = True,
                 disk_slack: float = 1.25):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.disk_slack = disk_slack
        self._seen: set[str] = set()
        self._mem = load
        self.size = 0
        self._pos = 0
        if load:
            self.states = np.zeros((capacity, encoding.N_PLANES, 8, 8), dtype=np.uint8)
            self.pidx = np.zeros((capacity, POLICY_K), dtype=np.int16)
            self.pprob = np.zeros((capacity, POLICY_K), dtype=np.float16)
            self.values = np.zeros(capacity, dtype=np.float32)
            self._load_existing()

    # -- disk ---------------------------------------------------------------
    def chunk_files(self) -> list[Path]:
        return sorted(p for p in self.root.glob("*-*-*.npz"))

    def _load_existing(self) -> None:
        files = self.chunk_files()
        legacy = [p for p in self.root.glob("*.npz") if p not in set(files)]
        if legacy:
            log.warning(f"ignoring {len(legacy)} legacy-format game files in {self.root}")
        chosen, total = [], 0
        for p in reversed(files):              # newest first until capacity
            if total >= self.capacity:
                break
            try:
                with np.load(p) as d:
                    n = len(d["values"])
            except Exception as e:
                log.warning(f"skipping unreadable chunk {p.name}: {e}")
                self._seen.add(p.name)
                continue
            chosen.append(p); total += n
        for p in files:
            self._seen.add(p.name)
        for p in reversed(chosen):
            self._ingest_file(p)
        log.info(f"replay buffer: loaded {self.size} positions from {len(chosen)} chunks")

    def _ingest_file(self, p: Path) -> int:
        try:
            with np.load(p) as d:
                self._insert(d["states"], d["pidx"], d["pprob"], d["values"])
                return len(d["values"])
        except Exception as e:
            log.warning(f"could not load chunk {p.name}: {e}")
            return 0

    def refresh(self) -> int:
        """Load chunk files written by other processes. Returns positions added."""
        added = 0
        for p in self.chunk_files():
            if p.name not in self._seen:
                self._seen.add(p.name)
                if self._mem:
                    added += self._ingest_file(p)
        return added

    def save_games(self, recs: list[dict]) -> Path | None:
        """Persist game records (dicts with states/pidx/pprob/values arrays,
        as produced by self_play.GameRunner) as one chunk; also add to memory."""
        recs = [r for r in recs if "states" in r and len(r["states"])]
        if not recs:
            return None
        cat = {k: np.concatenate([r[k] for r in recs]) for k in ("states", "pidx", "pprob", "values")}
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{next(_seq):06d}.npz"
        tmp = self.root / (name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, game_lengths=np.array([len(r["values"]) for r in recs]), **cat)
        path = self.root / name
        os.replace(tmp, path)
        self._seen.add(name)
        if self._mem:
            self._insert(cat["states"], cat["pidx"], cat["pprob"], cat["values"])
        self._prune()
        return path

    def save_game(self, states: np.ndarray, policies: np.ndarray, values: np.ndarray) -> Path | None:
        """Compat API: float32 planes (N,19,8,8) + dense (N,4672) policies."""
        n = len(values)
        u8 = np.rint(states * np.array([1] * 17 + [100, 1], np.float32)[None, :, None, None]).astype(np.uint8)
        pidx = np.zeros((n, POLICY_K), dtype=np.int16)
        pprob = np.zeros((n, POLICY_K), dtype=np.float16)
        for r in range(n):
            nz = np.flatnonzero(policies[r])
            if len(nz) > POLICY_K:
                nz = nz[np.argsort(-policies[r][nz])[:POLICY_K]]
            p = policies[r][nz]
            pidx[r, :len(nz)] = nz
            pprob[r, :len(nz)] = p / max(p.sum(), 1e-8)
        return self.save_games([{"states": u8, "pidx": pidx, "pprob": pprob,
                                 "values": values.astype(np.float32)}])

    def _prune(self) -> None:
        """Delete the oldest chunks once the disk holds well over `capacity`."""
        files = self.chunk_files()
        if len(files) < 50:
            return
        # Estimate positions per file from the newest few to avoid opening all.
        sizes = []
        for p in files[-5:]:
            try:
                with np.load(p) as d:
                    sizes.append(len(d["values"]))
            except Exception:
                pass
        per = max(1, int(np.mean(sizes))) if sizes else 1
        keep = int(self.capacity * self.disk_slack / per) + 1
        for p in files[:max(0, len(files) - keep)]:
            try:
                p.unlink()
            except OSError:
                pass

    # -- memory -------------------------------------------------------------
    def _insert(self, states, pidx, pprob, values) -> None:
        n = len(values)
        if n >= self.capacity:
            states, pidx, pprob, values = (a[-self.capacity:] for a in (states, pidx, pprob, values))
            n = self.capacity
        end = self._pos + n
        first = min(n, self.capacity - self._pos)
        for dst, src in ((self.states, states), (self.pidx, pidx),
                         (self.pprob, pprob), (self.values, values)):
            dst[self._pos:self._pos + first] = src[:first]
            if first < n:
                dst[:n - first] = src[first:]
        self._pos = end % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None):
        """Uniform batch → (states u8, pidx int16, pprob f16, values f32)."""
        if self.size == 0:
            raise RuntimeError(f"Replay buffer at {self.root} is empty")
        rng = rng or np.random.default_rng()
        i = np.sort(rng.integers(0, self.size, size=batch_size))
        return self.states[i], self.pidx[i], self.pprob[i], self.values[i]

    def n_games(self) -> int:
        """Chunk count on disk (each chunk holds several games)."""
        return len(self.chunk_files())
