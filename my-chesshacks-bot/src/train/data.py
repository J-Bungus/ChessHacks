import chess
import numpy as np
import torch
from torch.utils.data import Dataset

PIECE_MAP = {
    chess.PAWN:   0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  4,
    chess.KING:   5,
}

def encode_board_alpha_zero(board: chess.Board):
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    # --- 12 piece planes ---
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        piece_idx = PIECE_MAP[piece.piece_type]
        color_offset = 0 if piece.color == chess.WHITE else 6
        p = piece_idx + color_offset

        row = 7 - (square // 8)
        col = square % 8
        planes[p, row, col] = 1.0

    # --- Side to move (plane 12) ---
    planes[12] = 1.0 if board.turn == chess.WHITE else 0.0

    # --- Castling rights (planes 13–16) ---
    planes[13] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # --- En passant square (plane 17) ---
    if board.ep_square is not None:
        row = 7 - (board.ep_square // 8)
        col = board.ep_square % 8
        planes[17, row, col] = 1.0

    return torch.tensor(planes)


class ChessPGNDataset(Dataset):
    def __init__(self, pgn_path):
        boards, moves = # TODO

        self.move_to_idx, self.idx_to_move = build_move_index(moves)

        self.X = boards
        self.y = [self.move_to_idx[m.uci()] for m in moves]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        encoded = encode_board_alpha_zero(self.X[idx])   # (18, 8, 8)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return encoded, label
