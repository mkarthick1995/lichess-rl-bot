"""Print a one-shot training dashboard: liveness, counters, throughput,
recent arena / Stockfish results, and supervisor restarts.

    python -m src.status
"""
from __future__ import annotations

import json
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
    logs, ckpt = root / "logs", root / "checkpoints"
    print("================ training status ================")

    loop_log = _latest("loop-*.log", logs) if logs.exists() else None
    if loop_log:
        age = datetime.now().timestamp() - loop_log.stat().st_mtime
        alive = "RUNNING" if age < 600 else "STALE / STOPPED"
        print(f"loop log:     {loop_log.name}  (updated {_fmt_age(loop_log.stat().st_mtime)})  → {alive}")
    else:
        print("loop log:     none found — has training started?")
    if (logs / "STOP").exists():
        print("              logs/STOP present — training will stop / not restart")

    state_p = ckpt / "state.json"
    if state_p.exists():
        s = json.loads(state_p.read_text())
        print("-------------------------------------------------")
        print(f"run started:  {s.get('created')}")
        print(f"best net:     v{s['version']}  ({s['promotions']} promotions)")
        print(f"train steps:  {s['step']}")
        print(f"self-play:    {s['games']} games, {s['positions']} positions")
        print(f"stockfish:    next check at skill level {s['sf_level']}")
    else:
        print("no checkpoints/state.json yet")

    if loop_log:
        stats = [l for l in loop_log.read_text(encoding="utf-8", errors="replace").splitlines()
                 if "[stats]" in l]
        if stats:
            print("-------------------------------------------------")
            print("last progress line:")
            line = stats[-1].split("[stats] ", 1)[1]
            for part in line.split(" | "):
                print(f"  {part}")

    hist_p = ckpt / "history.jsonl"
    if hist_p.exists():
        events = [json.loads(l) for l in hist_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        matches = [e for e in events if e["kind"] == "match"]
        sf = [e for e in events if e["kind"] == "stockfish"]
        if matches:
            print("-------------------------------------------------")
            print(f"arena: {len(matches)} evaluations, last 5:")
            for e in matches[-5:]:
                w, d, l = e["wdl"]
                print(f"  {e['time']}  step {e['step']:>7} vs v{e['best']:<4} "
                      f"{w}W-{d}D-{l}L  {e['score']:.2f}  {'PROMOTED' if e['promoted'] else ''}")
        if sf:
            print("-------------------------------------------------")
            print("vs local Stockfish (skill level, 50ms/move):")
            for e in sf[-8:]:
                w, d, l = e["wdl"]
                print(f"  {e['time']}  best v{e['best']:<4} skill {e['level']:>2}: "
                      f"{w}W-{d}D-{l}L  score {e['score']:.2f}")

    sup = logs / "supervisor.log"
    if sup.exists():
        lines = sup.read_text(encoding="utf-8", errors="replace").splitlines()
        starts = sum(1 for l in lines if "starting training" in l)
        deaths = sum(1 for l in lines if "died" in l)
        print("-------------------------------------------------")
        print(f"supervisor:   {starts} starts, {deaths} restarts after death")
        if lines:
            print(f"  last: {lines[-1]}")
    print("=================================================")


if __name__ == "__main__":
    main()
