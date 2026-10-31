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


HALFMOVE_PLANE = 17


def is_repetition(board: chess.Board, count: int) -> bool:
    """Same result as board.is_repetition(count), but only scans positions
    since the last capture / pawn move (nothing earlier can repeat) and only
    those with the same side to move. python-chess scans the whole game
    stack on every call, which dominates MCTS time in long games."""
    hm = board.halfmove_clock
    if hm < 4:
        return False
    stack = board._stack
    occ = board.occupied
    seen = 1
    for i in range(len(stack) - 2, max(len(stack) - hm - 1, -1), -2):
        if stack[i].occupied == occ:
            seen += 1
            if seen >= count:
                return board.is_repetition(count)     # confirm with the full check
    return False


def encode_board_u8(board: chess.Board) -> np.ndarray:
    """Encode `board` as a compact (N_PLANES, 8, 8) uint8 array from STM's POV.

    Identical to encode_board except plane 17 holds the raw halfmove clock
    (capped at 100) instead of clock/100 — see planes_to_float. Built from
    bitboards, ~10x faster than a per-square loop; this sits on the MCTS hot
    path (one call per leaf evaluation)."""
    us = board.turn
    own, opp = board.occupied_co[us], board.occupied_co[not us]
    bbs = (board.pawns, board.knights, board.bishops,
           board.rooks, board.queens, board.kings)
    bits = np.unpackbits(
        np.array([bb & own for bb in bbs] + [bb & opp for bb in bbs],
                 dtype=np.uint64).view(np.uint8),
        bitorder="little").reshape(12, 8, 8)            # [plane, rank, file]
    if us == chess.BLACK:
        bits = bits[:, ::-1, :]

    planes = np.empty((N_PLANES, 8, 8), dtype=np.uint8)
    planes[:12] = bits
    planes[12] = 1                                      # side-to-move (always us)
    planes[13] = board.has_kingside_castling_rights(us)
    planes[14] = board.has_queenside_castling_rights(us)
    planes[15] = board.has_kingside_castling_rights(not us)
    planes[16] = board.has_queenside_castling_rights(not us)
    planes[17] = min(board.halfmove_clock, 100)
    planes[18] = is_repetition(board, 2)
    return planes


def planes_to_float(planes):
    """uint8 planes from encode_board_u8 (numpy or torch, batched (B,19,8,8)
    or single) → float32 network input matching encode_board."""
    if isinstance(planes, np.ndarray):
        x = planes.astype(np.float32)
    else:
        x = planes.float()
    x[..., HALFMOVE_PLANE, :, :] /= 100.0
    return x


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode `board` as a (N_PLANES, 8, 8) float32 tensor from STM's POV."""
    return planes_to_float(encode_board_u8(board))


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


# ---------------------------------------------------------------------------
# Fast move → index lookup (MCTS hot path). Tables are keyed in the STM frame,
# i.e. after flipping squares for black, so one table serves both colours.

def _build_move_tables() -> tuple[list[int], list[int]]:
    white = chess.Board()                 # move_to_index only reads board.turn
    plain = [-1] * (64 * 64)              # queen-like / knight / queen-promo
    under = [-1] * (64 * 64 * 3)          # underpromotion to N/B/R
    for f in range(64):
        for t in range(64):
            if f == t:
                continue
            try:
                plain[f * 64 + t] = move_to_index(chess.Move(f, t), white)
            except ValueError:
                pass
            if chess.square_rank(f) == 6 and chess.square_rank(t) == 7:
                for k, piece in enumerate(UP_PIECES):
                    try:
                        under[(f * 64 + t) * 3 + k] = move_to_index(
                            chess.Move(f, t, promotion=piece), white)
                    except ValueError:
                        pass
    return plain, under


_PLAIN_IDX, _UNDER_IDX = _build_move_tables()


def legal_moves_and_indices(board: chess.Board) -> tuple[list[chess.Move], np.ndarray]:
    """All legal moves of `board` and their policy indices (int64 array)."""
    flip = 56 if board.turn == chess.BLACK else 0
    moves = list(board.legal_moves)
    plain, under = _PLAIN_IDX, _UNDER_IDX
    idx = []
    for m in moves:
        k = (m.from_square ^ flip) * 64 + (m.to_square ^ flip)
        p = m.promotion
        idx.append(under[k * 3 + p - 2] if p and p != chess.QUEEN else plain[k])
    return moves, np.array(idx, dtype=np.int64)
