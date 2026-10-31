"""Batched network evaluation for MCTS leaves.

One forward pass scores a whole batch of positions; the policy logits are
gathered down to each position's legal moves on the GPU (so only a few
hundred floats per position cross back to the CPU) and softmaxed there.
"""
from __future__ import annotations

import weakref

import numpy as np
import torch

from . import encoding


class BatchEvaluator:
    """Callable: (list of uint8 planes, list of legal-move index arrays) →
    (list of prior arrays, values ndarray). Values are from each position's
    side-to-move POV."""

    def __init__(self, net, device, fp16: bool | None = None, pad_to: int = 16):
        self.net = net
        self.device = torch.device(device)
        self.fp16 = (self.device.type == "cuda") if fp16 is None else fp16
        self.pad_to = pad_to       # bucket batch sizes so cuDNN autotune caches stay small
        self.positions = 0         # running count, for throughput stats

    @torch.inference_mode()
    def __call__(self, planes: list, idx: list) -> tuple[list, np.ndarray]:
        b = len(planes)
        bp = -(-b // self.pad_to) * self.pad_to
        x = np.zeros((bp, encoding.N_PLANES, 8, 8), dtype=np.uint8)
        x[:b] = planes
        lens = [len(i) for i in idx]
        width = max(1, max(lens))
        gidx = np.zeros((b, width), dtype=np.int64)
        for r, ix in enumerate(idx):
            gidx[r, :len(ix)] = ix

        xt = encoding.planes_to_float(torch.from_numpy(x).to(self.device, non_blocking=True))
        gi = torch.from_numpy(gidx).to(self.device, non_blocking=True)
        mask = torch.arange(width, device=self.device)[None, :] < torch.tensor(lens, device=self.device)[:, None]

        if self.fp16:
            with torch.autocast("cuda", dtype=torch.float16):
                logits, value = self.net(xt)
        else:
            logits, value = self.net(xt)
        g = logits[:b].float().gather(1, gi).masked_fill_(~mask, float("-inf"))
        priors = torch.softmax(g, dim=1).cpu().numpy()
        values = value[:b].float().cpu().numpy()
        self.positions += b
        return [priors[r, :n] for r, n in enumerate(lens)], values


_cache: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def evaluator_for(net, device) -> BatchEvaluator:
    """Shared evaluator per network (the Lichess wiring calls run_mcts every move)."""
    ev = _cache.get(net)
    if ev is None or ev.device != torch.device(device):
        ev = _cache[net] = BatchEvaluator(net, device)
    return ev
