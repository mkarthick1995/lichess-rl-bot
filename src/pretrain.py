"""Supervised pre-training of ChessNetV2 on strong human bullet games.

    python -m src.pretrain --epochs 2
    python -m src.pretrain --steps 2000 --batch-size 256      # quick smoke run

Reads the shards written by `python -m src.lichess_db extract`
(data/positions/*/manifest.json). Fully resumable: the model, optimizer, LR
schedule and the exact data position are checkpointed every --save-every
steps to checkpoints/v2/pretrain_latest.pt and restored on start. Create
logs/STOP_PRETRAIN to stop cleanly. The checkpoint with the best validation
loss is kept as checkpoints/v2/pretrain_best.pt.

Losses: policy = cross-entropy to the human move (half weight when the mover
had < 3 s, where moves are mostly panic); wdl = cross-entropy to the game
result from the side to move; mlh = Huber on plies-left / 100. Clock planes
are dropped (zeroed, flag off) for 20% of samples so the network also works
without clocks (self-play).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ._log import setup_logging
from .network2 import ChessNetV2, samples_to_input

log = logging.getLogger("pretrain")

CKPT_DIR = Path("checkpoints") / "v2"
STOP = Path("logs") / "STOP_PRETRAIN"
STOP_ALL = Path("logs") / "STOP"        # the supervisor's stop file (run_training.ps1)


def _keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (released on exit)."""
    import ctypes
    import sys
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def find_shards(root: Path = Path("data") / "positions") -> tuple[list[tuple[Path, int]], list[Path]]:
    train, val = [], []
    for man_path in sorted(root.glob("*/manifest.json")):
        man = json.loads(man_path.read_text())
        for s in man["shards"]:
            train.append((man_path.parent / s["file"], s["n"]))
        val += [man_path.parent / s["file"] for s in man["val_shards"]]
    return train, val


class ShardStream:
    """Epochs over shuffled groups of shards, shuffled within each group.
    Deterministic given (seed, epoch, group), so (epoch, group, offset) is a
    complete resume point."""

    def __init__(self, shards, batch_size: int, group: int = 4, seed: int = 0):
        self.shards, self.bs, self.group, self.seed = shards, batch_size, group, seed
        self.epoch = self.gi = self.offset = 0
        self._arr = None
        self._key = None

    def state(self) -> dict:
        return {"epoch": self.epoch, "gi": self.gi, "offset": self.offset}

    def load_state(self, st: dict) -> None:
        self.epoch, self.gi, self.offset = st["epoch"], st["gi"], st["offset"]
        self._arr = None

    def _groups(self, epoch: int) -> list[list[int]]:
        order = np.random.default_rng((self.seed, epoch)).permutation(len(self.shards))
        return [list(order[i:i + self.group]) for i in range(0, len(order), self.group)]

    def _load(self):
        groups = self._groups(self.epoch)
        key = (self.epoch, self.gi)
        if self._key != key:
            arr = np.concatenate([np.load(self.shards[i][0]) for i in groups[self.gi]])
            perm = np.random.default_rng((self.seed, self.epoch, self.gi, 1)).permutation(len(arr))
            self._arr, self._key = arr[perm], key
        return groups

    def next(self) -> np.ndarray:
        while True:
            groups = self._load()
            if self.offset + self.bs <= len(self._arr):
                rows = self._arr[self.offset:self.offset + self.bs]
                self.offset += self.bs
                return rows
            self.offset = 0
            self.gi += 1
            if self.gi >= len(groups):
                self.gi = 0
                self.epoch += 1
                log.info(f"--- epoch {self.epoch} ---")


def losses(net, rows: np.ndarray, device, clock_dropout: float, rng=None):
    x = samples_to_input(rows, device, clock_dropout, rng).to(memory_format=torch.channels_last)
    move = torch.from_numpy(rows["move"].astype(np.int64)).to(device)
    wdl_t = torch.from_numpy((1 - rows["result"].astype(np.int64))).to(device)     # +1→0 win, 0→1, -1→2
    ml_t = torch.from_numpy(rows["plies_left"].astype(np.float32) / 100.0).to(device)
    pw = torch.from_numpy(np.where(rows["our_ds"] < 30, 0.5, 1.0).astype(np.float32)).to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        p, wdl, ml = net(x)
    lp = (F.cross_entropy(p.float(), move, reduction="none") * pw).sum() / pw.sum()
    lv = F.cross_entropy(wdl.float(), wdl_t)
    lm = F.smooth_l1_loss(ml.float(), ml_t)
    acc = (p.argmax(1) == move).float().mean()
    return lp, lv, lm, acc


@torch.no_grad()
def validate(net, val: np.ndarray, device, bs: int = 1024) -> dict:
    net.eval()
    tot = {"policy": 0.0, "wdl": 0.0, "mlh": 0.0, "acc": 0.0}
    n = 0
    for i in range(0, len(val) - bs + 1, bs):
        lp, lv, lm, acc = losses(net, val[i:i + bs], device, 0.0)
        for k, v in zip(tot, (lp, lv, lm, acc)):
            tot[k] += float(v)
        n += 1
    net.train()
    return {k: v / max(1, n) for k, v in tot.items()}


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return 1e-2 * peak + (peak - 1e-2 * peak) * 0.5 * (1 + math.cos(math.pi * t))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=0, help="override --epochs with a fixed step count")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--clock-dropout", type=float, default=0.2)
    ap.add_argument("--channels", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--val-every", type=int, default=5000)
    ap.add_argument("--val-size", type=int, default=200_000)
    ap.add_argument("--tag", default="pretrain")
    ap.add_argument("--finish-stop", action="store_true",
                    help="when done, create logs/STOP (used under the supervisor)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    setup_logging(args.tag, "INFO")
    _keep_awake()
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_shards, val_shards = find_shards()
    if not train_shards:
        raise SystemExit("no training shards; run `python -m src.lichess_db extract` first")
    n_train = sum(n for _, n in train_shards)
    total = args.steps or int(args.epochs * n_train / args.batch_size)
    val = np.concatenate([np.load(p) for p in val_shards])[:args.val_size] if val_shards else None
    log.info(f"{len(train_shards)} shards, {n_train:,} train positions, "
             f"{0 if val is None else len(val):,} val; {total:,} steps of {args.batch_size}")

    net = ChessNetV2(args.channels, args.blocks).to(device).to(memory_format=torch.channels_last)
    log.info(f"ChessNetV2 {args.channels}x{args.blocks}: "
             f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params")
    decay = [p for n, p in net.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in net.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    stream = ShardStream(train_shards, args.batch_size)
    rng = np.random.default_rng(1234)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    latest, best = CKPT_DIR / f"{args.tag}_latest.pt", CKPT_DIR / f"{args.tag}_best.pt"
    step, best_val = 0, float("inf")
    config = {"channels": args.channels, "blocks": args.blocks}
    if latest.exists():
        ck = torch.load(latest, map_location=device)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        stream.load_state(ck["stream"])
        step, best_val = ck["step"], ck.get("best_val", best_val)
        log.info(f"resumed from {latest} at step {step} ({stream.state()})")

    def save(path: Path):
        tmp = path.with_suffix(".tmp")
        torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(), "stream": stream.state(),
                    "step": step, "best_val": best_val, "config": config, "total": total}, tmp)
        tmp.replace(path)

    net.train()
    t0, win = time.monotonic(), {"p": 0.0, "v": 0.0, "m": 0.0, "a": 0.0, "n": 0}
    while step < total:
        if STOP.exists() or STOP_ALL.exists():
            log.info("stop file found; saving and exiting")
            break
        lr = lr_at(step, total, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        lp, lv, lm, acc = losses(net, stream.next(), device, args.clock_dropout, rng)
        loss = lp + lv + 0.5 * lm
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        step += 1
        win["p"] += float(lp); win["v"] += float(lv); win["m"] += float(lm); win["a"] += float(acc)
        win["n"] += 1
        if step % 200 == 0:
            n = win["n"]
            rate = n * args.batch_size / (time.monotonic() - t0)
            eta = (total - step) * args.batch_size / max(rate, 1) / 3600
            log.info(f"step {step:,}/{total:,}  policy {win['p'] / n:.3f} (top-1 {win['a'] / n:.1%})  "
                     f"wdl {win['v'] / n:.3f}  mlh {win['m'] / n:.4f}  lr {lr:.2e}  "
                     f"{rate:,.0f} pos/s  ETA {eta:.1f} h")
            t0, win = time.monotonic(), {"p": 0.0, "v": 0.0, "m": 0.0, "a": 0.0, "n": 0}
        if val is not None and (step % args.val_every == 0 or step == total):
            vm = validate(net, val, device)
            vloss = vm["policy"] + vm["wdl"] + 0.5 * vm["mlh"]
            log.info(f"[val] step {step:,}: policy {vm['policy']:.3f} (top-1 {vm['acc']:.1%})  "
                     f"wdl {vm['wdl']:.3f}  mlh {vm['mlh']:.4f}  total {vloss:.3f}"
                     + ("  (best)" if vloss < best_val else ""))
            if vloss < best_val:
                best_val = vloss
                save(best)
        if step % args.save_every == 0:
            save(latest)
    save(latest)
    log.info(f"stopped at step {step:,}/{total:,}; best val loss {best_val:.3f}")
    if step >= total and args.finish_stop:
        # Finished for good: leave logs/STOP so logon launches don't rerun it.
        STOP_ALL.parent.mkdir(exist_ok=True)
        STOP_ALL.touch()
        log.info("pre-training complete; logs/STOP left in place")


if __name__ == "__main__":
    main()
