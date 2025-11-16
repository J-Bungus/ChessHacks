import chess
import numpy as np

# Map python-chess piece types to plane offsets
PIECE_INDEX = {
    chess.PAWN:   0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  4,
    chess.KING:   5,
}

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encode a chess.Board into an 18x8x8 tensor with:
      - 12 piece planes (6 white, 6 black)
      - 4 castling rights planes
      - 1 en-passant plane
      - 1 side-to-move plane
    """
    tensor = np.zeros((18, 8, 8), dtype=np.float32)

    # ----------------------------------------------------------------------
    # 12 Piece planes
    # ----------------------------------------------------------------------
    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type
        color = piece.color  # True = white, False = black

        # 0–5 = white pieces, 6–11 = black pieces
        plane = PIECE_INDEX[piece_type] + (0 if color else 6)

        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        tensor[plane, row, col] = 1.0

    # ----------------------------------------------------------------------
    # 4 Castling rights planes (filled with 1s if right exists)
    # ----------------------------------------------------------------------
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[12, :, :] = 1.0

    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[14, :, :] = 1.0

    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0

    # ----------------------------------------------------------------------
    # 1 En-passant target square plane
    # ----------------------------------------------------------------------
    if board.ep_square is not None:
        ep_row = 7 - chess.square_rank(board.ep_square)
        ep_col = chess.square_file(board.ep_square)
        tensor[16, ep_row, ep_col] = 1.0

    # ----------------------------------------------------------------------
    # 1 Side-to-move plane
    # ----------------------------------------------------------------------
    # All 1's if white to move; all 0's if black to move
    if board.turn == chess.WHITE:
        tensor[17, :, :] = 1.0

    return tensor

def encode_pair(board_a: chess.Board, board_b: chess.Board) -> np.ndarray:
    """
    Encode two chess.Board objects into a single tensor of shape (36, 8, 8).
    The boards are simply stacked depth-wise: [board_a_planes, board_b_planes].
    """
    a = encode_board(board_a)   # shape (18, 8, 8)
    b = encode_board(board_b)   # shape (18, 8, 8)
    return np.concatenate([a, b], axis=0)

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionRankNet(nn.Module):
    """
    Input:  tensor of shape (batch, 36, 8, 8)
    Output: scalar in [-1, 1] indicating which position is better
    """
    def __init__(self):
        super().__init__()

        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(36, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Fully-connected head
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()   # restrict output to [-1, 1]
        )

    def forward(self, x):
        # x: (batch_size, 36, 8, 8)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_layers(x)
        return x
