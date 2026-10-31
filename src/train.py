"""Train the policy + value network on samples from the replay buffer
(standalone; src/loop.py does this continuously during a training run)."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from torch.optim import SGD

from ._log import setup_logging
from .loop import train_step
from .network import ChessNet
from .replay_buffer import ReplayBuffer

log = logging.getLogger("train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/train.pt",
                    help="weights to resume from and save to")
    ap.add_argument("--buffer", type=str, default="games")
    ap.add_argument("--buffer-capacity", type=int, default=2_000_000)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    log_path = setup_logging("train", args.log_level)
    log.info(f"args = {vars(args)}")
    if log_path:
        log.info(f"log file: {log_path}")

    device = torch.device(args.device)
    net = ChessNet(channels=args.channels, blocks=args.blocks).to(device)
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        sd = torch.load(ckpt, map_location=device)
        net.load_state_dict(sd["model"] if "model" in sd else sd)
        log.info(f"resumed from {ckpt}")
    else:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        log.warning(f"no checkpoint at {ckpt}; training from random init")
    opt = SGD(net.parameters(), lr=args.lr, momentum=0.9,
              weight_decay=args.weight_decay, nesterov=True)

    buffer = ReplayBuffer(args.buffer, capacity=args.buffer_capacity)
    if buffer.size == 0:
        log.error("buffer is empty — generate games first (python -m src.self_play)")
        return

    net.train()
    t_log, acc = time.monotonic(), [0.0, 0.0]
    for step in range(1, args.steps + 1):
        pl, vl, gn = train_step(net, opt, buffer.sample(args.batch_size), device)
        acc[0] += pl; acc[1] += vl
        if step % args.log_every == 0:
            n, dt = args.log_every, time.monotonic() - t_log
            log.info(f"step {step:6d}  policy={acc[0]/n:.3f}  value={acc[1]/n:.3f}  "
                     f"grad_norm={gn:.2f}  {dt/n*1000:.0f}ms/step")
            t_log, acc = time.monotonic(), [0.0, 0.0]
        if step % args.save_every == 0:
            torch.save(net.state_dict(), ckpt)
    torch.save(net.state_dict(), ckpt)
    log.info(f"done. final checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
