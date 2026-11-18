"""Batched evaluation for ChessNetV2 inside MCTS, plus checkpoint auto-detection.

Clocks: the search tree has no real clocks below the root, so every leaf gets
the root's clocks from its own side-to-move point of view (ours and the
opponent's swap every ply). `search()` passes each leaf's ply parity relative
to the root for evaluators that set `uses_parity`.

Moves-left utility: the value handed to the search is
    v' = v * (1 - ml_coef * min(plies_left, 200) / 200)
so when winning (v > 0) shorter games score higher — the search converts
instead of shuffling — and when losing (v < 0) longer games score higher,
which in bullet keeps the opponent's clock running.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import encoding
from .network2 import ChessNetV2, load_v2, planes_u8_to_input


class BatchEvaluatorV2:
    uses_parity = True

    def __init__(self, net: ChessNetV2, device, ml_coef: float = 0.1, pad_to: int = 16):
        self.net = net
        self.device = torch.device(device)
        self.ml_coef = ml_coef
        self.pad_to = pad_to
        self.positions = 0
        self.set_clock(None)

    def set_clock(self, clock) -> None:
        """`clock` = bullet.Clock from the ROOT side to move, or None (no clock)."""
        if clock is None:
            self.clk = (0.0, 0.0, 0.0, False)
        else:
            self.clk = (clock.our_ms / 1000.0, clock.opp_ms / 1000.0, clock.inc_ms / 1000.0, True)

    @torch.inference_mode()
    def __call__(self, planes: list, idx: list, parity=None):
        b = len(planes)
        bp = -(-b // self.pad_to) * self.pad_to
        x = np.zeros((bp, encoding.N_PLANES, 8, 8), dtype=np.uint8)
        x[:b] = planes
        par = np.zeros(bp, dtype=bool)
        if parity is not None:
            par[:b] = np.asarray(parity, dtype=bool)
        our_s, opp_s, inc_s, known = self.clk
        our = np.where(par, opp_s, our_s).astype(np.float32)
        opp = np.where(par, our_s, opp_s).astype(np.float32)
        dev = self.device
        xt = planes_u8_to_input(torch.from_numpy(x).to(dev),
                                torch.from_numpy(our).to(dev), torch.from_numpy(opp).to(dev),
                                torch.full((bp,), inc_s, device=dev),
                                torch.full((bp,), known, device=dev, dtype=torch.bool))

        lens = [len(i) for i in idx]
        width = max(1, max(lens))
        gidx = np.zeros((b, width), dtype=np.int64)
        for r, ix in enumerate(idx):
            gidx[r, :len(ix)] = ix
        gi = torch.from_numpy(gidx).to(dev)
        mask = torch.arange(width, device=dev)[None, :] < torch.tensor(lens, device=dev)[:, None]

        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logits, wdl, ml = self.net(xt.to(memory_format=torch.channels_last))
        g = logits[:b].float().gather(1, gi).masked_fill_(~mask, float("-inf"))
        priors = torch.softmax(g, dim=1).cpu().numpy()
        v = ChessNetV2.scalar_value(wdl[:b])
        if self.ml_coef:
            v = v * (1.0 - self.ml_coef * (ml[:b].float() * 100.0).clamp(max=200.0) / 200.0)
        self.positions += b
        return [priors[r, :n] for r, n in enumerate(lens)], v.cpu().numpy()


def is_v2_checkpoint(path) -> bool:
    ck = torch.load(path, map_location="cpu")
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    return any(k.startswith("mlh_fc") for k in sd)


def make_evaluator(path, device, channels_v1: int = 128, blocks_v1: int = 10):
    """Evaluator for either network generation, picked from the checkpoint."""
    if is_v2_checkpoint(path):
        return BatchEvaluatorV2(load_v2(path, device), device)
    from .inference import BatchEvaluator
    from .self_play import load_net
    net = load_net(str(path), channels_v1, blocks_v1, torch.device(device))
    net.eval()
    return BatchEvaluator(net, device)
