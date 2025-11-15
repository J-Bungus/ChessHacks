from .utils import chess_manager, GameContext
from chess import Move
import random
import time

# For Model
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

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ChessEvaluator().to(device)
model.load_state_dict(torch.load("train/archive/chess_evaluator4-1.pth", map_location=device))
model.eval()

def evaluate_board(board: chess.Board):
    x = encode_board(board).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(x).item()


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Cooking move...")
    print(ctx.board.move_stack)
    # time.sleep(0.1)

    best_move = None
    best_eval = None

    # Iterate over all legal moves
    for move in ctx.board.legal_moves:
        board_copy = ctx.board.copy()
        board_copy.push(move)

        # Convert board to input tensor
        x = encode_board(board_copy)  # shape: (18, 8, 8)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)  # (1,18,8,8)

        # NN evaluation
        with torch.no_grad():
            eval_value = model(x).item()
        
        # Choose max for White, min for Black
        if best_eval is None:
            best_eval = eval_value
            best_move = move
        else:
            if ctx.board.turn == chess.WHITE and eval_value > best_eval:
                best_eval = eval_value
                best_move = move
            elif ctx.board.turn == chess.BLACK and eval_value < best_eval:
                best_eval = eval_value
                best_move = move
    
    if best_move is None:
        raise ValueError("No legal moves")
    return best_move
    
    # legal_moves = list(ctx.board.generate_legal_moves())
    # if not legal_moves:
    #     ctx.logProbabilities({})
    #     raise ValueError("No legal moves available (i probably lost didn't i)")

    # move_weights = [random.random() for _ in legal_moves]
    # total_weight = sum(move_weights)
    # # Normalize so probabilities sum to 1
    # move_probs = {
    #     move: weight / total_weight
    #     for move, weight in zip(legal_moves, move_weights)
    # }
    # ctx.logProbabilities(move_probs)

    # return random.choices(legal_moves, weights=move_weights, k=1)[0]


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
