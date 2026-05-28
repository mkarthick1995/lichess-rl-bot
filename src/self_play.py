"""Self-play game generation using MCTS + the current network."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import chess

from . import encoding, mcts
from ._log import setup_logging
from .network import ChessNet
from .replay_buffer import ReplayBuffer

log = logging.getLogger("self_play")


def _outcome_str(outcome) -> str:
    if outcome is None:
        return "incomplete (move cap)"
    if outcome.winner is None:
        return f"draw ({outcome.termination.name})"
    return f"{'white' if outcome.winner else 'black'} wins ({outcome.termination.name})"


def _play_one_game(net, device, n_simulations: int,
                   temp_moves: int = 30,
                   max_moves: int = 512,
                   c_puct: float = 1.5,
                   game_id: str = "?",
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    board = chess.Board()
    rng = np.random.default_rng()
    states, policies, players = [], [], []
    move_count = 0
    t_game = time.monotonic()
    total_mcts_time = 0.0

    log.info(f"[game {game_id}] start  n_sims={n_simulations}  temp_moves={temp_moves}  max_moves={max_moves}")

    while not board.is_game_over(claim_draw=False) and move_count < max_moves:
        temperature = 1.0 if move_count < temp_moves else 0.0

        t_mcts = time.monotonic()
        root = mcts.run_mcts(board, net, device,
                             n_simulations=n_simulations, c_puct=c_puct,
                             add_noise=True, rng=rng)
        dt_mcts = time.monotonic() - t_mcts
        total_mcts_time += dt_mcts

        state = encoding.encode_board(board)
        policy = np.zeros(encoding.N_MOVES, dtype=np.float32)
        visit_probs = mcts.visit_count_policy(root, temperature=1.0)
        for m, p in visit_probs.items():
            policy[encoding.move_to_index(m, board)] = p
        states.append(state); policies.append(policy); players.append(board.turn)

        sample_probs = mcts.visit_count_policy(root, temperature=temperature)
        moves_list = list(sample_probs.keys())
        probs = np.array(list(sample_probs.values()))
        chosen = moves_list[rng.choice(len(moves_list), p=probs)]

        if log.isEnabledFor(logging.DEBUG):
            top = sorted(visit_probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
            top_str = ", ".join(f"{board.san(m)}:{p:.2f}" for m, p in top)
            log.debug(f"[game {game_id}] ply {move_count+1} "
                      f"({'W' if board.turn == chess.WHITE else 'B'}) "
                      f"played={board.san(chosen)}  v={root.value:+.2f}  "
                      f"top=[{top_str}]  mcts={dt_mcts*1000:.0f}ms")

        board.push(chosen)
        move_count += 1

    outcome = board.outcome(claim_draw=True)
    dt_game = time.monotonic() - t_game
    log.info(f"[game {game_id}] end    plies={move_count}  total={dt_game:.1f}s  "
             f"mcts={total_mcts_time:.1f}s  ({dt_game/max(1,move_count):.2f}s/move)  "
             f"-- {_outcome_str(outcome)}")

    winner = outcome.winner if outcome else None
    values = np.zeros(len(states), dtype=np.float32)
    if winner is not None:
        for i, p in enumerate(players):
            values[i] = 1.0 if p == winner else -1.0

    stats = {
        "plies": move_count,
        "duration_s": dt_game,
        "mcts_s": total_mcts_time,
        "outcome": _outcome_str(outcome),
    }
    return np.stack(states), np.stack(policies), values, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="model weights; uses random init if missing")
    ap.add_argument("--buffer", type=str, default="games")
    ap.add_argument("--n-games", type=int, default=10)
    ap.add_argument("--n-simulations", type=int, default=100)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-level", type=str, default="INFO",
                    help="DEBUG / INFO / WARNING / ERROR")
    args = ap.parse_args()

    log_path = setup_logging("self-play", args.log_level)
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

    if args.checkpoint and Path(args.checkpoint).exists():
        net.load_state_dict(torch.load(args.checkpoint, map_location=device))
        log.info(f"loaded checkpoint: {args.checkpoint}")
    else:
        if args.checkpoint:
            log.warning(f"checkpoint not found at {args.checkpoint}; using random init")
        else:
            log.warning("no checkpoint provided; using random init")
    net.eval()

    buffer = ReplayBuffer(args.buffer)
    log.info(f"replay buffer at {args.buffer} (currently {buffer.n_games()} games)")

    t_total = time.monotonic()
    for i in range(args.n_games):
        states, policies, values, _ = _play_one_game(
            net, device, args.n_simulations, game_id=f"{i+1}/{args.n_games}")
        path = buffer.save_game(states, policies, values)
        log.debug(f"saved {path.name}  ({states.shape[0]} positions)")

    log.info(f"generated {args.n_games} games in {time.monotonic()-t_total:.1f}s; "
             f"buffer now has {buffer.n_games()} games")


if __name__ == "__main__":
    main()
