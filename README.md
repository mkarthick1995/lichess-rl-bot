# lichess-rl-bot

An AlphaZero-style reinforcement-learning chess bot, trained from scratch by
self-play, that plays on Lichess via browser automation.

> **Fair-play notice.** Use the Lichess wiring only against the Lichess
> computer (Stockfish levels) or friend challenges. Playing a bot against real
> humans on Lichess violates their Terms of Service.

## How it works

- **Network** (`src/network.py`) — a ResNet trunk with a policy head (4672-move
  distribution) and a value head (scalar position evaluation).
- **Search** (`src/mcts.py`, `src/inference.py`) — PUCT Monte-Carlo Tree Search
  guided by the network. Leaves from many trees (or from one tree, via virtual
  loss) are scored in one batched GPU call, with the policy gathered to the
  legal moves on the GPU.
- **Game generation** (`src/self_play.py`) — `GameRunner` plays ~150 games at
  once per process so the GPU sees large batches. Each recorded position keeps
  its MCTS visit distribution and the final game result.
- **Replay buffer** (`src/replay_buffer.py`) — the last 2M positions in RAM in a
  compact form (uint8 planes and the top-96 sparse policy), persisted to
  `games/` as chunk files that any process can add to.
- **Orchestrator** (`src/loop.py`, `src/workers.py`) — 7 self-play worker
  processes stream games with the current *best* net. The trainer trains a
  *candidate* continuously, paced so each position is sampled about 4 times.
  An arena worker plays 160-game matches of candidate snapshots against best
  and promotes those that score ≥ 55%. Every 5 promotions, best is measured
  against a local Stockfish skill ladder (`src/evaluate.py`).
- **Lichess wiring** (`src/lichess/`) — Playwright reads the board from the DOM,
  picks a move with MCTS + the trained net, and clicks it. If the page and the
  tracked game ever disagree, the bot resyncs from the server's move list.
- **Stockfish farm** (`src/lichess/farm.py`) — starts and plays games in a row
  against Lichess's "Play with the computer" Stockfish levels, ramping the
  level by recent win rate. Each game is recorded into `games/`, and the
  trainer picks it up automatically.

Throughput on a Ryzen 7 5800X + RTX 3070 Ti: ~22k network evaluations/s, i.e.
~2–3k self-play games/hour at 100 simulations per move. That's about 100× the
original one-position-at-a-time search.

## Setup (Windows, RTX 3070 Ti)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# CUDA build of PyTorch (your driver supports CUDA 13.2, so cu124 wheels work):
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124

.\.venv\Scripts\python.exe -m pip install python-chess numpy tqdm playwright
.\.venv\Scripts\python.exe -m playwright install chromium
```

Verify CUDA is picked up:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Training

Training runs under a supervisor that restarts it if it dies. The loop resumes
on its own from `checkpoints/state.json`, `checkpoints/latest.pt` and `games/`:

```powershell
Start-Process powershell -ArgumentList "-ExecutionPolicy","Bypass","-WindowStyle","Hidden","-File",".
un_training.ps1"
```

A scheduled task, **"lichess-rl-bot training"**, starts the same script at
logon, so training resumes after a reboot. Only one supervisor can run at a
time (named mutex). While training runs, the loop keeps Windows from sleeping.

- **Progress:** `.\.venv\Scripts\python.exe -m src.status` (counters,
  throughput, arena results, Stockfish ladder, restarts). Full log:
  `logs\loop-*.log`; every evaluation is also appended to
  `checkpoints\history.jsonl`.
- **Stop:** `New-Item logs\STOP`. The loop saves and exits within seconds,
  the supervisor exits, and the logon task won't restart it while `logs\STOP`
  exists. Run the script by hand to resume (that clears `STOP`).
- **Remove the logon task:** `Unregister-ScheduledTask -TaskName "lichess-rl-bot training"`.

Outputs: `checkpointsest.pt` is always the newest promoted net (what the
Lichess tools use). `checkpoints\genest_NNNNN.pt` keeps the promotion
history: the last 10, plus every 10th.

Tunable knobs (`python -m src.loop --help`): `--workers`/`--parallel` (CPU
parallelism), `--gen-sims`/`--eval-sims`, `--sample-ratio`,
`--buffer-capacity`, `--lr-schedule`, `--eval-games`/`--promote-threshold`,
`--hours`.

Standalone tools:

```powershell
python -m src.self_play --checkpoint checkpoints/best.pt --n-games 64     # generate games
python -m src.train --steps 1000                                          # train on games/
python -m src.evaluate --a checkpoints/best.pt --b checkpoints/gen/best_00010.pt
python -m src.evaluate --a checkpoints/best.pt --stockfish-level 3        # vs local Stockfish
```

`src.evaluate --stockfish-level` and the in-loop ladder use Stockfish 17.1 at
`tools\stockfish\stockfish.exe` (official build, not in git; download
`stockfish-windows-x86-64-avx2.zip` from the Stockfish GitHub releases and
rename the exe).

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q
```

These cover encoding against a reference implementation, move indexing,
repetition detection, the search (mate-in-1, visit/virtual-loss accounting),
game records and value signs, and replay-buffer ring/reload/refresh.

## Playing on Lichess

Play one game against Lichess's computer and watch (Chromium opens; add
`--headless` to hide it):

```powershell
.\.venv\Scripts\python.exe -m src.lichess.play --level 3 --color random
```

Or pass a game URL (`python -m src.lichess.play https://lichess.org/<id>`).
With no arguments it opens lichess.org: start a game yourself, then press
Enter in the terminal. `--n-simulations` (default 800) sets search depth per
move; on the GPU, 800 simulations take about 0.4s.

Fully automatic games against Lichess's Stockfish levels, recorded as
training data:

```powershell
.\.venv\Scripts\python.exe -m src.lichess.farm --checkpoint checkpoints/best.pt --n-games 20 --start-level 1 --headless
```

## Logs

Every entry point writes to stdout **and** a timestamped file under `logs/`.
Use `--log-level DEBUG` for per-move MCTS detail (top candidates, value, timing)
and, in the Lichess wiring, every selector / coordinate / click. On errors the
wiring saves a screenshot to `logs/screenshots/` and logs the game data from
the page.

## Expectations

With 3M parameters (128ch × 10 blocks) on a single RTX 3070 Ti, realistic strength after a few
weeks of self-play is club level (~1500–1800 Lichess rapid). Surpassing strong
Stockfish is not achievable at this scale — that's a compute limit, not a code
one.
