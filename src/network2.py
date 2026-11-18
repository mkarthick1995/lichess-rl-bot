"""Second-generation network: clock-aware, with WDL and moves-left heads.

Input  = the 19 board planes of encoding.py (side-to-move frame)
       + 4 clock planes: our time / 60 s, opponent time / 60 s (both clipped
         to [0, 3]), increment / 10 s, and a "clock known" flag. With the flag
         at 0 (and the other three at 0) the network evaluates the position
         alone — used for clock dropout in training and for self-play.
Body   = residual tower with squeeze-and-excitation (SE scales and biases
         each channel from a global summary of the board).
Heads  = policy: 4672 logits, same layout as encoding.move_to_index, so the
                 MCTS code and move tables are unchanged;
         wdl:    win/draw/loss logits from the side to move — with clocks,
                 the probability of winning *including on time*;
         mlh:    moves-left, plies until the game ends (/100).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import encoding

N_CLOCK_PLANES = 4
N_INPUT = encoding.N_PLANES + N_CLOCK_PLANES
CLOCK_CLIP = 3.0                 # minutes


class SEResBlock(nn.Module):
    def __init__(self, ch: int, se_ratio: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.se1 = nn.Linear(ch, ch // se_ratio)
        self.se2 = nn.Linear(ch // se_ratio, 2 * ch)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        s = F.relu(self.se1(y.mean(dim=(2, 3))))
        w, b = self.se2(s).chunk(2, dim=1)
        y = y * torch.sigmoid(w)[:, :, None, None] + b[:, :, None, None]
        return F.relu(x + y)


class ChessNetV2(nn.Module):
    def __init__(self, channels: int = 192, blocks: int = 12, se_ratio: int = 8):
        super().__init__()
        self.channels, self.blocks = channels, blocks
        self.stem = nn.Sequential(
            nn.Conv2d(N_INPUT, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(*[SEResBlock(channels, se_ratio) for _ in range(blocks)])

        self.policy = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, encoding.N_MOVE_TYPES, 1))
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.value_fc = nn.Sequential(nn.Linear(32 * 64, 128), nn.ReLU(inplace=True), nn.Linear(128, 3))
        self.mlh_conv = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True))
        self.mlh_fc = nn.Sequential(nn.Linear(8 * 64, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

    def forward(self, x):
        x = self.trunk(self.stem(x))
        p = self.policy(x)                                            # (B, 73, 8, 8)
        p = p.permute(0, 2, 3, 1).reshape(p.size(0), -1)              # (B, 4672)
        wdl = self.value_fc(self.value_conv(x).flatten(1))            # (B, 3)
        mlh = F.relu(self.mlh_fc(self.mlh_conv(x).flatten(1))).squeeze(-1)   # plies/100
        return p, wdl, mlh

    @staticmethod
    def scalar_value(wdl_logits: torch.Tensor) -> torch.Tensor:
        """Expected score in [-1, 1] from the side to move: P(win) - P(loss)."""
        pr = torch.softmax(wdl_logits.float(), dim=-1)
        return pr[..., 0] - pr[..., 2]


# ---------------------------------------------------------------------------
# Inputs

def clock_planes(our_s, opp_s, inc_s, known) -> torch.Tensor:
    """(B, 4, 8, 8) clock planes from per-sample tensors (seconds, bool)."""
    k = known.float()
    vals = torch.stack([
        (our_s / 60.0).clamp(0, CLOCK_CLIP) * k,
        (opp_s / 60.0).clamp(0, CLOCK_CLIP) * k,
        (inc_s / 10.0).clamp(0, CLOCK_CLIP) * k,
        k,
    ], dim=1)
    return vals[:, :, None, None].expand(-1, -1, 8, 8)


_SHIFTS = None


def samples_to_input(rows: np.ndarray, device, clock_dropout: float = 0.0,
                     generator: np.random.Generator | None = None) -> torch.Tensor:
    """lichess_db.SAMPLE_DTYPE rows → (B, N_INPUT, 8, 8) float input on `device`.
    Board planes match encoding.encode_board_u8 + planes_to_float exactly."""
    global _SHIFTS
    B = len(rows)
    bb = torch.from_numpy(np.ascontiguousarray(rows["bb"]).view(np.uint8).reshape(B, 12, 8))
    bb = bb.to(device, non_blocking=True)
    if _SHIFTS is None or _SHIFTS.device != bb.device:
        _SHIFTS = torch.arange(8, device=bb.device, dtype=torch.uint8)
    bits = ((bb[..., None] >> _SHIFTS) & 1).float()                  # (B,12,8,8) [rank, file]

    flags = torch.from_numpy(rows["flags"].astype(np.int64)).to(device)
    halfmove = torch.from_numpy(rows["halfmove"].astype(np.float32)).to(device)
    scal = torch.stack([
        torch.ones(B, device=device),
        (flags & 1).float(), ((flags >> 1) & 1).float(),
        ((flags >> 2) & 1).float(), ((flags >> 3) & 1).float(),
        halfmove / 100.0,
        ((flags >> 4) & 1).float(),
    ], dim=1)                                                        # (B, 7)

    known = np.ones(B, dtype=bool)
    if clock_dropout > 0:
        rng = generator or np.random.default_rng()
        known = rng.random(B) >= clock_dropout
    our = torch.from_numpy(rows["our_ds"].astype(np.float32) / 10.0).to(device)
    opp = torch.from_numpy(rows["opp_ds"].astype(np.float32) / 10.0).to(device)
    inc = torch.from_numpy(rows["inc_s"].astype(np.float32)).to(device)
    clk = clock_planes(our, opp, inc, torch.from_numpy(known).to(device))
    return torch.cat([bits, scal[:, :, None, None].expand(-1, -1, 8, 8), clk], dim=1)


def planes_u8_to_input(planes_u8: torch.Tensor, our_s, opp_s, inc_s, known) -> torch.Tensor:
    """Batched encode_board_u8 planes (B,19,8,8) + clocks → network input."""
    x = encoding.planes_to_float(planes_u8)
    return torch.cat([x, clock_planes(our_s, opp_s, inc_s, known)], dim=1)


def load_v2(path, device, channels: int = 192, blocks: int = 12) -> ChessNetV2:
    ck = torch.load(path, map_location=device)
    cfg = ck.get("config", {}) if isinstance(ck, dict) else {}
    net = ChessNetV2(cfg.get("channels", channels), cfg.get("blocks", blocks)).to(device)
    net.load_state_dict(ck["model"] if "model" in ck else ck)
    net.eval()
    return net
