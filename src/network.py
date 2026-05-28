"""Policy + value network (AlphaZero-style ResNet)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import encoding


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessNet(nn.Module):
    """ResNet trunk with policy (4672-dim logits) and value (scalar in [-1,1])."""

    def __init__(self, channels: int = 128, blocks: int = 10):
        super().__init__()
        self.channels = channels
        self.blocks   = blocks

        self.stem = nn.Sequential(
            nn.Conv2d(encoding.N_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        # Policy head: 1×1 conv to 73 channels, then permute so flatten order
        # matches encoding.move_to_index = (rank*8 + file)*73 + move_type.
        self.policy_conv = nn.Conv2d(channels, encoding.N_MOVE_TYPES, 1)
        self.policy_bn   = nn.BatchNorm2d(encoding.N_MOVE_TYPES)

        # Value head: 1×1 conv → fc(256) → fc(1) → tanh.
        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(64, 256)
        self.value_fc2  = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(self.stem(x))

        p = F.relu(self.policy_bn(self.policy_conv(x)))               # (B, 73, 8, 8)
        p = p.permute(0, 2, 3, 1).contiguous().view(p.size(0), -1)    # (B, 4672)

        v = F.relu(self.value_bn(self.value_conv(x)))                 # (B, 1, 8, 8)
        v = v.view(v.size(0), -1)                                     # (B, 64)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)                 # (B,)

        return p, v
