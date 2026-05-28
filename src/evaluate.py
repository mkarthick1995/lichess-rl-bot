"""Arena: pit two checkpoints against each other and report Elo gap."""
from __future__ import annotations

import argparse
import logging
import math

import numpy as np
import torch
import chess

from . import mcts
from ._log import setup_logging
from .network import ChessNet

log = logging.getLogger("evaluate")


@torch.inference_mode()
def _play_game(net_a, net_b, device, n_simulations: int,
               a_plays_white: bool, game_id: str = "?") -> int:
    board = chess.Board()
    rng = np.random.default_rng()
    plies = 0
    while not board.is_game_over(claim_draw=True) and board.fullmove_number < 400:
        is_a_turn = (board.turn == chess.WHITE) == a_plays_white
        net = net_a if is_a_turn else net_b
        root = mcts.run_mcts(board, net, device,
                             n_simulations=n_simulations,
                             add_noise=False, rng=rng)
        # Deterministic: max-visit move.
        move = max(root.children.items(), key=lambda kv: kv[1].visit_count)[0]
        board.push(move)
        plies += 1

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        result = 0
        kind = "draw"
    else:
        a_won = (outcome.winner == chess.WHITE) == a_plays_white
        result = 1 if a_won else -1
        kind = "A win" if a_won else "B win"
    log.info(f"[game {game_id}] A={'W' if a_plays_white else 'B'}  plies={plies}  {kind}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="challenger checkpoint")
    ap.add_argument("--b", required=True, help="incumbent checkpoint")
    ap.add_argument("--n-games", type=int, default=20)
    ap.add_argument("--n-simulations", type=int, default=100)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    setup_logging("evaluate", args.log_level)
    log.info(f"args = {vars(args)}")

    device = torch.device(args.device)
    log.info(f"device = {device}")

    def _load(path: str):
        n = ChessNet(channels=args.channels, blocks=args.blocks).to(device)
        n.load_state_dict(torch.load(path, map_location=device))
        n.eval()
        return n

    net_a = _load(args.a); net_b = _load(args.b)
    log.info(f"A = {args.a}")
    log.info(f"B = {args.b}")

    wins, draws, losses = 0, 0, 0
    for i in range(args.n_games):
        r = _play_game(net_a, net_b, device, args.n_simulations,
                       a_plays_white=(i % 2 == 0),
                       game_id=f"{i+1}/{args.n_games}")
        if   r > 0: wins   += 1
        elif r < 0: losses += 1
        else:       draws  += 1
        log.info(f"   running: {wins}W-{draws}D-{losses}L")

    score = wins + 0.5 * draws
    wr = score / args.n_games
    if 0 < wr < 1:
        elo = -400 * math.log10(1 / wr - 1)
        log.info(f"final: {wins}W-{draws}D-{losses}L  win_rate={wr:.3f}  Elo gap = {elo:+.0f}")
    else:
        sign = "+inf" if wr == 1 else "-inf"
        log.info(f"final: {wins}W-{draws}D-{losses}L  win_rate={wr:.3f}  Elo gap = {sign}")


if __name__ == "__main__":
    main()
