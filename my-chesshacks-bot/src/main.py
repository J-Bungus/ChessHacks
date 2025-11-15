from .utils import chess_manager, GameContext
from chess import Move
import random
import time
from transformers import AutoModel, AutoConfig
import os
from train.src.model import ChessModel, ChessModelConfig
from train.src.data import encode_board, MOVE_TO_IDX
import torch

# Load model from Hugging Face
AutoConfig.register("chess-model", ChessModelConfig)
AutoModel.register(ChessModelConfig, ChessModel)

model = AutoModel.from_pretrained(
    "darren-lo/chess-bot-model",
    cache_dir="./.model_cache"  # Cache locally
)

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    print("HELLO WORLD")
    """
    Called each time the model needs to make a move.
    Returns a python-chess Move object (legal move) for current position.
    """

    print("Cooking move...")
    print(ctx.board.move_stack)
    time.sleep(0.1)

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (probably lost)")

    # --- Convert board to model input ---
    # Assuming you have a helper function to convert a board to tensor
    # e.g., shape [1, input_channels, 8, 8]
    board_tensor = encode_board(ctx.board)  # implement this function
    board_tensor = board_tensor.to(next(model.parameters()).device)

    with torch.no_grad():
        policy_logits, value = model(board_tensor)

    # Policy logits -> probabilities
    policy_probs = torch.softmax(policy_logits, dim=-1).squeeze(0)  # [num_moves]

    # Filter out illegal moves
    legal_indices = [MOVE_TO_IDX[m] for m in legal_moves]  # implement move_to_index
    legal_probs = policy_probs[legal_indices]

    # Normalize to sum to 1
    legal_probs = legal_probs / legal_probs.sum()
    move_probs = {
        move: float(prob)
        for move, prob in zip(legal_moves, legal_probs)
    }

    ctx.logProbabilities(move_probs)

    # Pick the move with highest probability
    best_move = max(move_probs, key=move_probs.get)
    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
