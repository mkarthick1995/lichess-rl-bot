"""Worker processes for the training loop (spawned by src/loop.py).

selfplay_worker  plays self-play games forever with the current best network
                 (reloading whenever the shared `version` counter changes) and
                 streams finished game records to the trainer over a queue.
arena_worker     runs evaluation jobs (candidate vs best, net vs Stockfish)
                 from a task queue and reports results back.

Each worker owns its own CUDA context and network copy; the networks are
small, so this costs a few hundred MB of VRAM per worker and lets Python tree
search run on several cores in parallel.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import signal
import time
import traceback
from pathlib import Path

import torch

from ._log import setup_logging
from .inference import BatchEvaluator
from .self_play import SearchConfig, load_net, run_selfplay


def best_path(ckpt_dir: str | Path, version: int) -> Path:
    return Path(ckpt_dir) / "gen" / f"best_{version:05d}.pt"


def _detach_from_console() -> None:
    """Workers are stopped by the trainer (stop_evt / terminate), never by
    console signals: a stray Ctrl+C / Ctrl+Break sent to the console would
    otherwise kill every worker at once."""
    for name in ("SIGINT", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, signal.SIG_IGN)
            except (OSError, ValueError):
                pass


def _parent_watch(every: float = 5.0):
    """Returns a cheap callable that is True once the trainer process has
    died, so an orphaned worker exits instead of playing games nobody saves."""
    parent = mp.parent_process()
    state = {"t": 0.0, "dead": False}

    def dead() -> bool:
        if parent is None or state["dead"]:
            return state["dead"]
        now = time.monotonic()
        if now - state["t"] >= every:
            state["t"] = now
            state["dead"] = not parent.is_alive()
        return state["dead"]

    return dead


PAUSE_FILE = Path("logs") / "PAUSE"
PAUSE_MAX_AGE = 600.0      # a PAUSE file not touched for this long is stale


def training_paused() -> bool:
    """True while the Lichess bot is playing (it creates logs/PAUSE and
    touches it every move). A stale file — e.g. the bot crashed — is ignored."""
    try:
        return time.time() - PAUSE_FILE.stat().st_mtime < PAUSE_MAX_AGE
    except OSError:
        return False


def _pause_waiter(stop_evt, parent_dead, log, every: float = 1.0):
    """Callable that blocks while training is paused (checked at most every
    `every` seconds, so the per-step cost is negligible)."""
    state = {"t": 0.0}

    def wait() -> None:
        now = time.monotonic()
        if now - state["t"] < every:
            return
        state["t"] = now
        if not training_paused():
            return
        log.info("paused (logs/PAUSE: bot game in progress)")
        while training_paused() and not stop_evt.is_set() and not parent_dead():
            time.sleep(0.5)
        log.info("resumed")

    return wait


def _load_into(net, path: Path, device, log) -> bool:
    for attempt in range(10):
        try:
            sd = torch.load(path, map_location=device)
            net.load_state_dict(sd["model"] if "model" in sd else sd)
            return True
        except Exception as e:                    # file mid-write / locked: retry
            log.warning(f"loading {path.name} failed ({e}); retry {attempt + 1}")
            time.sleep(1.0)
    return False


def selfplay_worker(wid: int, cfg: dict, version, out_q, stop_evt) -> None:
    log = logging.getLogger(f"worker.sp{wid}")
    setup_logging(f"worker-sp{wid}", cfg["log_level"])
    _detach_from_console()
    parent_dead = _parent_watch()
    try:
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = True
        device = torch.device(cfg["device"])
        net = load_net(None, cfg["channels"], cfg["blocks"], device)
        ev = BatchEvaluator(net, device)
        state = {"version": -1}

        def maybe_reload():
            v = version.value
            if v != state["version"]:
                if _load_into(net, best_path(cfg["ckpt_dir"], v), device, log):
                    net.eval()
                    log.info(f"loaded best v{v}")
                    state["version"] = v

        maybe_reload()
        pause_wait = _pause_waiter(stop_evt, parent_dead, log)

        def between_steps():
            pause_wait()
            maybe_reload()

        search = SearchConfig(n_sims=cfg["gen_sims"], c_puct=cfg["c_puct"])
        t_last, n_games = time.monotonic(), 0

        def on_game(rec):
            nonlocal t_last, n_games
            rec["version"] = state["version"]
            rec["worker"] = wid
            rec["evals_total"] = ev.positions
            out_q.put(rec)
            n_games += 1
            if time.monotonic() - t_last > 300:
                log.info(f"{n_games} games so far; {ev.positions} evals")
                t_last = time.monotonic()

        run_selfplay(ev, search, 0, cfg["parallel"], on_game,
                     temp_plies=cfg["temp_plies"], max_plies=cfg["max_plies"],
                     should_stop=lambda: stop_evt.is_set() or parent_dead(),
                     between_steps=between_steps)
        if parent_dead():
            log.warning("trainer process is gone; exiting")
            out_q.cancel_join_thread()      # nobody will read the queue
    except Exception:
        log.error("self-play worker crashed:\n" + traceback.format_exc())
        raise


def arena_worker(cfg: dict, task_q, result_q, stop_evt) -> None:
    from .evaluate import play_match, play_stockfish

    log = logging.getLogger("worker.arena")
    setup_logging("worker-arena", cfg["log_level"])
    _detach_from_console()
    parent_dead = _parent_watch(every=0.0)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    device = torch.device(cfg["device"])
    search = SearchConfig(n_sims=cfg["eval_sims"], c_puct=cfg["c_puct"])
    pause_wait = _pause_waiter(stop_evt, parent_dead, log)

    while not stop_evt.is_set():
        if parent_dead():
            log.warning("trainer process is gone; exiting")
            return
        pause_wait()
        try:
            task = task_q.get(timeout=1.0)
        except queue.Empty:
            continue
        t0 = time.monotonic()
        try:
            ev_a = BatchEvaluator(load_net(task["a"], cfg["channels"], cfg["blocks"], device), device)
            if task["kind"] == "match":
                ev_b = BatchEvaluator(load_net(task["b"], cfg["channels"], cfg["blocks"], device), device)
                wdl = play_match(ev_a, ev_b, task["n_games"], search, parallel=task["n_games"],
                                 max_plies=cfg.get("arena_max_plies", 512), between_steps=pause_wait)
            else:
                wdl = play_stockfish(ev_a, task["level"], task["n_games"], search, parallel=32,
                                     between_steps=pause_wait)
            result_q.put({**task, "wdl": wdl, "seconds": time.monotonic() - t0})
            log.info(f"task {task['kind']} done: {wdl} in {time.monotonic() - t0:.0f}s")
        except Exception:
            log.error(f"arena task failed: {task}\n" + traceback.format_exc())
            result_q.put({**task, "error": traceback.format_exc()})
