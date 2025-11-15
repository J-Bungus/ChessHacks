import torch
import torch.nn as nn
import chess
import numpy as np

# ----------------------------
# Board Encoding (same as before)
# ----------------------------
def encode_board(board: chess.Board):
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()

    piece_to_plane = {
        (chess.PAWN, True): 0,
        (chess.KNIGHT, True): 1,
        (chess.BISHOP, True): 2,
        (chess.ROOK, True): 3,
        (chess.QUEEN, True): 4,
        (chess.KING, True): 5,
        (chess.PAWN, False): 6,
        (chess.KNIGHT, False): 7,
        (chess.BISHOP, False): 8,
        (chess.ROOK, False): 9,
        (chess.QUEEN, False): 10,
        (chess.KING, False): 11,
    }

    for square, piece in piece_map.items():
        row = 7 - (square // 8)
        col = square % 8
        plane = piece_to_plane[(piece.piece_type, piece.color)]
        planes[plane, row, col] = 1.0

    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    planes[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    if board.ep_square is not None:
        ep_file = board.ep_square % 8
        planes[17, :, ep_file] = 1.0

    return torch.tensor(planes)


# ----------------------------
# Model (same as before)
# ----------------------------
class ChessEvaluator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        raw = self.fc2(x)

        # Clamp the output to real chess evaluation scale
        return 10.0 * torch.tanh(raw)