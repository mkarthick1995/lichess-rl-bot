# lichess-rl-bot

An AlphaZero-style reinforcement-learning chess bot, trained from scratch by
self-play, that plays on Lichess via browser automation.

> **Fair-play notice.** Use the Lichess wiring only against the Lichess
> computer (Stockfish levels) or friend challenges. Playing a bot against real
> humans on Lichess violates their Terms of Service.

## How it works

- **Network** (`src/network.py`) — a ResNet trunk with a policy head (4672-move
  distribution) and a value head (scalar position evaluation).
- **Search** (`src/mcts.py`) — PUCT Monte-Carlo Tree Search guided by the
  network; the value head replaces hand-written evaluation.
- **Self-play** (`src/self_play.py`) — the current network plays itself; each
  position is stored with its MCTS visit distribution and the game outcome.
- **Training** (`src/train.py`) — cross-entropy on the policy + MSE on the
  value, sampled from a disk-backed replay buffer (`src/replay_buffer.py`).
- **Orchestrator** (`src/loop.py`) — automates self-play → train → evaluate →
  gate. A trained *candidate* only replaces the *best* network if it wins a
  win-rate margin in the arena (`src/evaluate.py`).
- **Lichess wiring** (`src/lichess/`) — Playwright reads the board from the DOM,
  picks a move with MCTS + the trained net, and clicks it. Heavily logged.

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

Run the full loop (resumable — reloads `checkpoints/best.pt` on restart):

```powershell
.\.venv\Scripts\python.exe -m src.loop --iterations 100 --games-per-iter 25 --train-steps 400 --eval-games 20
```

Tunable knobs (defaults in parentheses): `--gen-sims` (100) and `--eval-sims`
(100) MCTS simulations per move, `--batch-size` (256), `--channels` (128),
`--blocks` (10), `--promote-threshold` (0.55). On 8 GB VRAM the defaults fit;
lower `--batch-size` if you hit out-of-memory.

Individual phases can also be run standalone: `python -m src.self_play`,
`python -m src.train`, `python -m src.evaluate --a A.pt --b B.pt`.

## Playing on Lichess

```powershell
.\.venv\Scripts\python.exe -m src.lichess.play --checkpoint checkpoints/best.pt --log-level DEBUG
```

Chromium opens; navigate to a game (e.g. a challenge against the Stockfish
bot), return to the terminal and press Enter. The bot then plays its turns.

## Logs

Every entry point writes to stdout **and** a timestamped file under `logs/`.
Use `--log-level DEBUG` for per-move MCTS detail (top candidates, value, timing)
and, in the Lichess wiring, every selector / coordinate / click. On errors the
wiring saves a screenshot to `logs/screenshots/` and dumps
`window.lichess.round.data`.

## Expectations

With ~5M parameters on a single RTX 3070 Ti, realistic strength after a few
weeks of self-play is club level (~1500–1800 Lichess rapid). Surpassing strong
Stockfish is not achievable at this scale — that's a compute limit, not a code
one.
