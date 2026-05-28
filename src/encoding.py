"""
Board ↔ tensor encoding (AlphaZero-style).

The network always sees positions from the side-to-move's perspective: when
black is to move we vertically flip the board so own pieces always live in the
bottom planes. Moves are encoded the same way so the network's output indices
are symmetric across colours.

Move encoding: 64 from-squares × 73 move types = 4672.
    0–55  queen-like  (8 directions × 7 distances)
    56–63 knight
    64–72 underpromotion to N/B/R (queen promotions use the queen-like slot)
"""
from __future__ import annotations

import numpy as np
import chess

PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING)

N_PLANES = 19          # 12 piece planes + side + 4 castling + halfmove + repetition
N_MOVE_TYPES = 73
N_MOVES = 64 * N_MOVE_TYPES   # 4672

# Queen-like move directions (dr, df), row-major: N, NE, E, SE, S, SW, W, NW.
QUEEN_DIRECTIONS: tuple[tuple[int, int], ...] = (
    ( 1,  0), ( 1,  1), ( 0,  1), (-1,  1),
    (-1,  0), (-1, -1), ( 0, -1), ( 1, -1),
)

# Knight (dr, df).
KNIGHT_DELTAS: tuple[tuple[int, int], ...] = (
    ( 2,  1), ( 1,  2), (-1,  2), (-2,  1),
    (-2, -1), (-1, -2), ( 1, -2), ( 2, -1),
)

# Underpromotion: pawn moves toward higher rank in current-player frame.
UP_DIRECTIONS: tuple[tuple[int, int], ...] = ((1, -1), (1, 0), (1, 1))
UP_PIECES: tuple[int, ...] = (chess.KNIGHT, chess.BISHOP, chess.ROOK)


def _flip_square(sq: int) -> int:
    return chess.square(chess.square_file(sq), 7 - chess.square_rank(sq))


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode `board` as a (N_PLANES, 8, 8) float32 tensor from STM's POV."""
    us = board.turn
    them = not us
    flip = us == chess.BLACK

    planes = np.zeros((N_PLANES, 8, 8), dtype=np.float32)

    own_offset, opp_offset = 0, 6
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        idx = PIECE_TYPES.index(piece.piece_type)
        plane = (own_offset if piece.color == us else opp_offset) + idx
        s = _flip_square(sq) if flip else sq
        planes[plane, chess.square_rank(s), chess.square_file(s)] = 1.0

    # Plane 12: side-to-move (always 1.0 because we flip).
    planes[12, :, :] = 1.0

    if board.has_kingside_castling_rights(us):    planes[13, :, :] = 1.0
    if board.has_queenside_castling_rights(us):   planes[14, :, :] = 1.0
    if board.has_kingside_castling_rights(them):  planes[15, :, :] = 1.0
    if board.has_queenside_castling_rights(them): planes[16, :, :] = 1.0

    planes[17, :, :] = min(board.halfmove_clock / 100.0, 1.0)
    planes[18, :, :] = 1.0 if board.is_repetition(2) else 0.0

    return planes


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Encode `move` (legal on `board`) to a 0..4671 index in STM's frame."""
    flip = board.turn == chess.BLACK
    from_sq = _flip_square(move.from_square) if flip else move.from_square
    to_sq   = _flip_square(move.to_square)   if flip else move.to_square

    fr = chess.square_rank(from_sq); ff = chess.square_file(from_sq)
    tr = chess.square_rank(to_sq);   tf = chess.square_file(to_sq)
    dr, dc = tr - fr, tf - ff

    if move.promotion and move.promotion != chess.QUEEN:
        try:
            dir_idx   = UP_DIRECTIONS.index((dr, dc))
            piece_idx = UP_PIECES.index(move.promotion)
            move_type = 64 + dir_idx * 3 + piece_idx
            return from_sq * N_MOVE_TYPES + move_type
        except ValueError:
            pass

    if (dr, dc) in KNIGHT_DELTAS:
        move_type = 56 + KNIGHT_DELTAS.index((dr, dc))
        return from_sq * N_MOVE_TYPES + move_type

    distance = max(abs(dr), abs(dc))
    if distance == 0:
        raise ValueError(f"null move not encodable: {move}")
    ndr = dr // distance if dr else 0
    ndc = dc // distance if dc else 0
    if (ndr, ndc) not in QUEEN_DIRECTIONS:
        raise ValueError(f"cannot encode move {move}")
    dir_idx   = QUEEN_DIRECTIONS.index((ndr, ndc))
    move_type = dir_idx * 7 + (distance - 1)
    return from_sq * N_MOVE_TYPES + move_type


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """Decode index → chess.Move on `board`. May return illegal moves; mask first."""
    flip = board.turn == chess.BLACK
    from_sq_flat = idx // N_MOVE_TYPES
    move_type    = idx %  N_MOVE_TYPES
    fr, ff = divmod(from_sq_flat, 8)

    promotion = None
    if move_type < 56:
        dir_idx   = move_type // 7
        distance  = (move_type % 7) + 1
        dr, dc    = QUEEN_DIRECTIONS[dir_idx]
        tr, tf    = fr + dr * distance, ff + dc * distance
    elif move_type < 64:
        dr, dc    = KNIGHT_DELTAS[move_type - 56]
        tr, tf    = fr + dr, ff + dc
    else:
        up_idx    = move_type - 64
        dir_idx   = up_idx // 3
        piece_idx = up_idx %  3
        dr, dc    = UP_DIRECTIONS[dir_idx]
        tr, tf    = fr + dr, ff + dc
        promotion = UP_PIECES[piece_idx]

    fr_actual = 7 - fr if flip else fr
    tr_actual = 7 - tr if flip else tr
    from_real = chess.square(ff, fr_actual)
    to_real   = chess.square(tf, tr_actual)

    # Promote a queen-move-encoded pawn push/capture to the last rank.
    if promotion is None and 0 <= tf <= 7 and 0 <= tr <= 7:
        piece = board.piece_at(from_real)
        if piece and piece.piece_type == chess.PAWN and tr == 7:
            promotion = chess.QUEEN

    return chess.Move(from_real, to_real, promotion=promotion)


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Boolean (N_MOVES,) mask of legal moves in the network's index space."""
    mask = np.zeros(N_MOVES, dtype=bool)
    for m in board.legal_moves:
        try:
            mask[move_to_index(m, board)] = True
        except ValueError:
            pass
    return mask
