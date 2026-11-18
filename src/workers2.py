"""Worker processes for the generation-2 loop (src/loop2.py).

selfplay2_worker  Gumbel self-play with the current best network (reloaded
                  whenever the shared version counter changes); streams game
                  records to the trainer.
arena2_worker     runs gate tasks:
                    "match"  candidate vs best, fixed simulations, many games
                             at once (decides the self-play network);
                    "bullet" new best vs the bot's network in real clocked
                             bullet games (decides what plays on Lichess).
                  While a bullet gate runs the trainer sets `hold`, so self-play
                  and training stop and the clocks are not distorted.
"""
from __future__ import annotations

import logging
import queue
import time
import traceback
from pathlib import Path

import chess
import torch

from ._log import setup_logging
from .workers import _detach_from_console, _load_into, _parent_watch, _pause_waiter


def best_path2(ckpt_dir: str | Path, version: int) -> Path:
    return Path(ckpt_dir) / "gen" / f"best_{version:05d}.pt"


def _hold_waiter(hold, stop_evt, parent_dead, log):
    def wait() -> None:
        if not hold.is_set():
            return
        log.info("holding (bullet gate in progress)")
        while hold.is_set() and not stop_evt.is_set() and not parent_dead():
            time.sleep(0.5)
        log.info("resumed")
    return wait


def selfplay2_worker(wid: int, cfg: dict, version, hold, out_q, stop_evt) -> None:
    from .inference2 import BatchEvaluatorV2
    from .network2 import ChessNetV2
    from .selfplay2 import GumbelConfig, run

    log = logging.getLogger(f"worker.sp{wid}")
    setup_logging(f"worker2-sp{wid}", cfg["log_level"])
    _detach_from_console()
    parent_dead = _parent_watch()
    try:
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = True
        device = torch.device(cfg["device"])
        net = ChessNetV2(cfg["channels"], cfg["blocks"]).to(device)
        net = net.to(memory_format=torch.channels_last).eval()
        ev = BatchEvaluatorV2(net, device, ml_coef=cfg["ml_coef"])
        state = {"version": -1}

        def maybe_reload():
            v = version.value
            if v != state["version"] and _load_into(net, best_path2(cfg["ckpt_dir"], v), device, log):
                net.eval()
                log.info(f"loaded best v{v}")
                state["version"] = v

        maybe_reload()
        pause_wait = _pause_waiter(stop_evt, parent_dead, log)
        hold_wait = _hold_waiter(hold, stop_evt, parent_dead, log)

        def between_steps():
            pause_wait()
            hold_wait()
            maybe_reload()

        t_last, n_games = time.monotonic(), 0

        def on_game(rec):
            nonlocal t_last, n_games
            rec.update(version=state["version"], worker=wid, evals_total=ev.positions)
            out_q.put(rec)
            n_games += 1
            if time.monotonic() - t_last > 300:
                log.info(f"{n_games} games so far; {ev.positions} evals")
                t_last = time.monotonic()

        run(ev, GumbelConfig(**cfg["gumbel"]), cfg["parallel"], on_game,
            should_stop=lambda: stop_evt.is_set() or parent_dead(), between_steps=between_steps)
        if parent_dead():
            log.warning("trainer process is gone; exiting")
            out_q.cancel_join_thread()
    except Exception:
        log.error("self-play worker crashed:\n" + traceback.format_exc())
        raise


def arena2_worker(cfg: dict, task_q, result_q, stop_evt) -> None:
    from .bullet_arena import NetPlayer, parse_tc, run_match
    from .evaluate import play_match
    from .inference2 import BatchEvaluatorV2
    from .network2 import load_v2
    from .self_play import SearchConfig

    log = logging.getLogger("worker.arena2")
    setup_logging("worker2-arena", cfg["log_level"])
    _detach_from_console()
    parent_dead = _parent_watch(every=0.0)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    device = torch.device(cfg["device"])
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
            if task["kind"] == "match":
                ev_a = BatchEvaluatorV2(load_v2(task["a"], device), device, ml_coef=cfg["ml_coef"])
                ev_b = BatchEvaluatorV2(load_v2(task["b"], device), device, ml_coef=cfg["ml_coef"])
                wdl = play_match(ev_a, ev_b, task["n_games"], SearchConfig(n_sims=cfg["eval_sims"]),
                                 parallel=task["n_games"], max_plies=cfg["arena_max_plies"],
                                 between_steps=pause_wait)
                del ev_a, ev_b
            else:                                    # "bullet"
                base_ms, inc_ms = parse_tc(task["tc"])
                a = NetPlayer(task["a"], device, ponder=False, name=Path(task["a"]).stem)
                b = NetPlayer(task["b"], device, ponder=False, name="bot")
                try:
                    for p in (a, b):              # warm-up so game 1's first move is not slow
                        p.move(chess.Board(), {chess.WHITE: 60000, chess.BLACK: 60000}, 0, 0)
                    r = run_match(a, b, task["n_games"], base_ms, inc_ms, task["lag_ms"],
                                  use_openings=True, before_game=pause_wait)
                finally:
                    a.engine.close(); b.engine.close()
                wdl = (r["w"], r["d"], r["l"])
            if device.type == "cuda":
                torch.cuda.empty_cache()
            result_q.put({**task, "wdl": wdl, "seconds": time.monotonic() - t0})
            log.info(f"task {task['kind']} done: {wdl} in {time.monotonic() - t0:.0f}s")
        except Exception:
            log.error(f"arena task failed: {task}\n" + traceback.format_exc())
            result_q.put({**task, "error": traceback.format_exc()})
