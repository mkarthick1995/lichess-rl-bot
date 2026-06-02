"""Print a one-shot training dashboard: process state, buffer, checkpoints,
recent losses, promotions, and supervisor restarts.

    python -m src.status
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

# Windows console defaults to cp1252; force UTF-8 so arrows/symbols don't crash.
_reconfigure = getattr(sys.stdout, "reconfigure", None)
if _reconfigure is not None:
    try:
        _reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass


def _latest(glob_pat: str, root: Path) -> Path | None:
    files = sorted(root.glob(glob_pat), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _fmt_age(ts: float) -> str:
    secs = (datetime.now().timestamp() - ts)
    if secs < 90:        return f"{secs:.0f}s ago"
    if secs < 5400:      return f"{secs/60:.0f}m ago"
    return f"{secs/3600:.1f}h ago"


def main():
    root = Path(".")
    logs = root / "logs"
    print("========== training status ==========")

    # Process liveness — is anything writing the loop log recently?
    loop_log = _latest("loop-*.log", logs) if logs.exists() else None
    if loop_log:
        age = datetime.now().timestamp() - loop_log.stat().st_mtime
        alive = "LIKELY RUNNING" if age < 600 else "STALE / STOPPED"
        print(f"loop log:        {loop_log.name}  (updated {_fmt_age(loop_log.stat().st_mtime)})  → {alive}")
    else:
        print("loop log:        none found — has training started?")

    # Replay buffer.
    games_dir = root / "games"
    n_games = len(list(games_dir.glob("*.npz"))) if games_dir.exists() else 0
    print(f"replay buffer:   {n_games} games")

    # Checkpoints.
    for name in ("best.pt", "latest.pt"):
        p = root / "checkpoints" / name
        if p.exists():
            mb = p.stat().st_size / 1e6
            print(f"checkpoint:      {name}  {mb:.1f}MB  (updated {_fmt_age(p.stat().st_mtime)})")
        else:
            print(f"checkpoint:      {name}  — not yet written")

    # Parse the loop log for progress signals.
    if loop_log:
        text = loop_log.read_text(encoding="utf-8", errors="replace").splitlines()
        cur_iter = None
        last_loss = None
        last_eval = None
        promotions = 0
        for line in text:
            m = re.search(r"iteration (\d+)/(\d+)", line)
            if m:
                cur_iter = (int(m.group(1)), int(m.group(2)))
            m = re.search(r"training done: loss [\d.]+ . ([\d.]+)", line)
            if m:
                last_loss = float(m.group(1))
            m = re.search(r"eval:.*win_rate=([\d.]+)", line)
            if m:
                last_eval = float(m.group(1))
            if "PROMOTED" in line:
                promotions += 1
        print("--------------------------------------")
        if cur_iter:
            print(f"current iter:    {cur_iter[0]} / {cur_iter[1]}")
        if last_loss is not None:
            print(f"last train loss: {last_loss:.3f}")
        if last_eval is not None:
            print(f"last eval wr:    {last_eval:.3f}")
        print(f"promotions:      {promotions}")

    # Supervisor restart history.
    sup = logs / "supervisor.log" if logs.exists() else None
    if sup and sup.exists():
        lines = sup.read_text(encoding="utf-8", errors="replace").splitlines()
        starts = sum(1 for l in lines if "starting training" in l)
        deaths = sum(1 for l in lines if "died" in l)
        print("--------------------------------------")
        print(f"supervisor:      {starts} starts, {deaths} restarts after death")
        if lines:
            print(f"  last: {lines[-1]}")

    print("======================================")


if __name__ == "__main__":
    main()
