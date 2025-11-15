import torch
import numpy as np
import chess

NUM_PLANES = 17

def encode_board(board):
    planes = []

    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                    chess.ROOK, chess.QUEEN, chess.KING]

    # 12 piece planes
    for piece_type in piece_types:
        # white
        p = np.zeros((8, 8), np.float32)
        for sq in board.pieces(piece_type, chess.WHITE):
            r = 7 - chess.square_rank(sq)
            c = chess.square_file(sq)
            p[r, c] = 1.0
        planes.append(p)

        # black
        p = np.zeros((8, 8), np.float32)
        for sq in board.pieces(piece_type, chess.BLACK):
            r = 7 - chess.square_rank(sq)
            c = chess.square_file(sq)
            p[r, c] = 1.0
        planes.append(p)

    # side-to-move
    stm = np.ones((8,8), np.float32) if board.turn else np.zeros((8,8), np.float32)
    planes.append(stm)

    # castling rights
    for flag in [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ]:
        planes.append(np.ones((8,8), np.float32) if flag else np.zeros((8,8), np.float32))

    arr = np.stack(planes)
    assert len(arr) == NUM_PLANES
    
    return torch.tensor(arr, dtype=torch.float32)

# ---- Move encoding utilities: 4672 legal UCI moves ----
ALL_SQUARES = [f"{f}{r}" for f in "abcdefgh" for r in "12345678"]
ALL_MOVES = []

for s1 in ALL_SQUARES:
    for s2 in ALL_SQUARES:
        if s1 == s2:
            continue

        ALL_MOVES.append(s1 + s2)  # quiet + capture

        # promotions
        if s1[1] == "7" and s2[1] == "8":
            for promo in ["q", "r", "b", "n"]:
                ALL_MOVES.append(s1 + s2 + promo)
        elif s1[1] == "2" and s2[1] == "1":
            for promo in ["q", "r", "b", "n"]:
                ALL_MOVES.append(s1 + s2 + promo)

MOVE_TO_IDX = {m: i for i, m in enumerate(ALL_MOVES)}
NUM_MOVES = len(ALL_MOVES)
