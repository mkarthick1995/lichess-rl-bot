"""Train the policy + value network on samples from the replay buffer."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD

from ._log import setup_logging
from .network import ChessNet
from .replay_buffer import ReplayBuffer

log = logging.getLogger("train")


def _loss(policy_logits: torch.Tensor, value_pred: torch.Tensor,
          policy_target: torch.Tensor, value_target: torch.Tensor):
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(policy_target * log_probs).sum(dim=1).mean()
    value_loss  = F.mse_loss(value_pred, value_target)
    return policy_loss + value_loss, policy_loss, value_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    ap.add_argument("--buffer", type=str, default="games")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    log_path = setup_logging("train", args.log_level)
    log.info(f"args = {vars(args)}")
    if log_path:
        log.info(f"log file: {log_path}")

    device = torch.device(args.device)
    log.info(f"device = {device}  cuda_available={torch.cuda.is_available()}")
    if device.type == "cuda":
        log.info(f"GPU = {torch.cuda.get_device_name(0)}")

    net = ChessNet(channels=args.channels, blocks=args.blocks).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    log.info(f"network: {args.channels}ch × {args.blocks} blocks = {n_params/1e6:.2f}M params")

    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        net.load_state_dict(torch.load(ckpt, map_location=device))
        log.info(f"resumed from {ckpt}")
    else:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        log.warning(f"no checkpoint at {ckpt}; training from random init")

    optimizer = SGD(net.parameters(), lr=args.lr,
                    momentum=0.9, weight_decay=args.weight_decay)

    buffer = ReplayBuffer(args.buffer)
    log.info(f"buffer at {args.buffer} ({buffer.n_games()} games)")
    if buffer.n_games() == 0:
        log.error("buffer is empty — generate self-play games first "
                  "(python -m src.self_play)")
        return

    net.train()
    running = {"total": 0.0, "policy": 0.0, "value": 0.0}
    t_log = time.monotonic()
    last_gn = 0.0

    for step in range(args.steps):
        try:
            s_np, p_np, v_np = buffer.sample_batch(args.batch_size)
        except RuntimeError as e:
            log.error(f"sample failed: {e}")
            return

        states   = torch.from_numpy(s_np).to(device)
        policies = torch.from_numpy(p_np).to(device)
        values   = torch.from_numpy(v_np).to(device)

        logits, value_pred = net(states)
        total, pl, vl = _loss(logits, value_pred, policies, values)

        optimizer.zero_grad()
        total.backward()
        last_gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
        optimizer.step()

        running["total"]  += total.item()
        running["policy"] += pl.item()
        running["value"]  += vl.item()

        if (step + 1) % args.log_every == 0:
            n = args.log_every
            dt = time.monotonic() - t_log
            log.info(f"step {step+1:5d}  loss={running['total']/n:.3f}  "
                     f"policy={running['policy']/n:.3f}  value={running['value']/n:.3f}  "
                     f"grad_norm={last_gn:.2f}  {dt/n*1000:.0f}ms/step")
            running = {"total": 0.0, "policy": 0.0, "value": 0.0}
            t_log = time.monotonic()

        if (step + 1) % args.save_every == 0:
            torch.save(net.state_dict(), ckpt)
            log.info(f"saved checkpoint: {ckpt}")

    torch.save(net.state_dict(), ckpt)
    log.info(f"done. final checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
