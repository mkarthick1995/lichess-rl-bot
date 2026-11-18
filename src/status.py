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

    pre_log = _latest("pretrain-*.log", logs) if logs.exists() else None
    if pre_log:
        text = pre_log.read_text(encoding="utf-8", errors="replace").splitlines()
        steps = [l for l in text if "  step " in l and "pos/s" in l]
        vals = [l for l in text if "[val]" in l]
        age = datetime.now().timestamp() - pre_log.stat().st_mtime
        print("------------- generation 2: pre-training -------------")
        print(f"log:          {pre_log.name}  (updated {_fmt_age(pre_log.stat().st_mtime)})  "
              f"→ {'RUNNING' if age < 600 else 'STOPPED'}")
        if steps:
            print(f"  {steps[-1].split('pretrain', 1)[-1].strip()}")
        for v in vals[-3:]:
            print(f"  {v.split('[val] ', 1)[-1]}")
        if any("pre-training complete" in l for l in text):
            print("  COMPLETE")

    rl_dir = ckpt / "v2" / "rl"
    rl_log = _latest("loop2-*.log", logs) if logs.exists() else None
    if (rl_dir / "state.json").exists() or rl_log:
        print("------------- generation 2: self-play RL -------------")
        if rl_log:
            age = datetime.now().timestamp() - rl_log.stat().st_mtime
            print(f"log:          {rl_log.name}  (updated {_fmt_age(rl_log.stat().st_mtime)})  "
                  f"→ {'RUNNING' if age < 600 else 'STOPPED'}")
        if (rl_dir / "state.json").exists():
            s = json.loads((rl_dir / "state.json").read_text())
            print(f"started:      {s.get('created')}   best v{s['version']} ({s['promotions']} promotions)"
                  f"   bot plays: {s.get('bot')}")
            print(f"progress:     step {s['step']:,}, {s['games']:,} games, {s['positions']:,} positions")
        if rl_log:
            text = rl_log.read_text(encoding="utf-8", errors="replace").splitlines()
            stats = [l for l in text if "[stats]" in l]
            if stats:
                for part in stats[-1].split("[stats] ", 1)[1].split(" | "):
                    print(f"  {part}")
            held = [l for l in text if "[bullet] gating" in l or "[bullet] best" in l]
            if held and "gating" in held[-1]:
                print("  bullet gate in progress (self-play holds)")
        if (rl_dir / "history.jsonl").exists():
            ev = [json.loads(l) for l in (rl_dir / "history.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
            for e in ev[-6:]:
                w, d, l = e["wdl"]
                what = (f"step {e['step']:>7} vs best v{e['best']}" if e["kind"] == "match"
                        else f"BULLET v{e['best']} vs bot ({e['bot']})")
                print(f"  {e['time']}  {what}: {w}W-{d}D-{l}L  {e['score']:.2f}  "
                      f"{'PROMOTED' if e['promoted'] else ''}")

    sup = logs / "supervisor.log"
    if sup.exists():
        lines = sup.read_text(encoding="utf-8", errors="replace").splitlines()
        starts = sum(1 for l in lines if "starting training" in l or "starting job" in l)
        deaths = sum(1 for l in lines if "died" in l)
        print("-------------------------------------------------")
        print(f"supervisor:   {starts} starts, {deaths} restarts after death")
        if lines:
            print(f"  last: {lines[-1]}")
    print("=================================================")


if __name__ == "__main__":
    main()
