"""Disk-backed replay buffer of self-play games."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


class ReplayBuffer:
    """One game per .npz file. Old games are pruned when the buffer exceeds
    `max_games`. Sampling reads a handful of games into memory and draws a
    batch from their concatenation — cheap, and avoids loading everything."""

    def __init__(self, root: str | Path, max_games: int = 5000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_games = max_games
        self._next_id = self._scan_next_id()

    def _scan_next_id(self) -> int:
        ids: list[int] = []
        for p in self.root.glob("*.npz"):
            try:
                ids.append(int(p.stem))
            except ValueError:
                pass
        return max(ids) + 1 if ids else 0

    def save_game(self, states: np.ndarray, policies: np.ndarray, values: np.ndarray) -> Path:
        path = self.root / f"{self._next_id:08d}.npz"
        np.savez_compressed(path, states=states, policies=policies, values=values)
        self._next_id += 1
        self._prune()
        return path

    def _prune(self) -> None:
        files = sorted(self.root.glob("*.npz"))
        if len(files) > self.max_games:
            for f in files[: len(files) - self.max_games]:
                f.unlink()

    def n_games(self) -> int:
        return len(list(self.root.glob("*.npz")))

    def files(self) -> list[Path]:
        return sorted(self.root.glob("*.npz"))

    def sample_batch(self, batch_size: int,
                     files_per_batch: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        files = self.files()
        if not files:
            raise RuntimeError(f"Replay buffer at {self.root} is empty")

        if files_per_batch is None:
            files_per_batch = min(len(files), max(8, batch_size // 4))
        chosen = np.random.choice(np.array(files, dtype=object), size=files_per_batch, replace=False)

        states_l, policies_l, values_l = [], [], []
        for fp in chosen:
            with np.load(fp) as data:
                states_l.append(data["states"])
                policies_l.append(data["policies"])
                values_l.append(data["values"])
        states   = np.concatenate(states_l)
        policies = np.concatenate(policies_l)
        values   = np.concatenate(values_l)

        idx = np.random.choice(len(states), size=batch_size, replace=len(states) < batch_size)
        return states[idx], policies[idx], values[idx]
