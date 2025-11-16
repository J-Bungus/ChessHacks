from .utils import chess_manager, GameContext
from chess import Move
import random
import time
from transformers import AutoModel, AutoConfig
import os
from train.src.model import ChessModel, ChessModelConfig
from train.src.utils import encode_board, MOVE_TO_IDX
import torch
from src.search import MCTS

# Load model from Hugging Face
AutoConfig.register("chess-transformer", ChessModelConfig)
AutoModel.register(ChessModelConfig, ChessModel)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(
    "darren-lo/chess-bot-model",
    cache_dir="./.model_cache"  # Cache locally
)
model = model.to(device=device)


def mcts(ctx: GameContext):
    mcts = MCTS(model, simulations=200, device=next(model.parameters()).device)

    root = mcts.run(ctx.board)
    best_move = mcts.choose_move(root)

    # This also logs probabilities:
    move_probs = {move: child.N for move, child in root.children.items()}
    total = sum(move_probs.values())
    ctx.logProbabilities({m: n/total for m, n in move_probs.items()})

    return best_move

# def linear_search(ctx: GameContext):
#     legal_moves = list(ctx.board.generate_legal_moves())
#     if not legal_moves:
#         ctx.logProbabilities({})
#         raise ValueError("No legal moves available (probably lost)")

#     board_tensor = encode_board(ctx.board)  # implement this function
#     board_tensor = board_tensor.to(next(model.parameters()).device)
#     board_tensor = board_tensor.unsqueeze(0)

#     with torch.no_grad():
#         policy_logits, value = model(board_tensor)

#     # Policy logits -> probabilities
#     policy_probs = torch.softmax(policy_logits, dim=-1).squeeze(0)  # [num_moves]

#     # Filter out illegal moves
#     legal_indices = []  # implement move_to_index
#     for m in legal_moves:
#         idx = MOVE_TO_IDX.get(str(m), None)
#         if idx is None:
#             print(f"Could not map move {m}")
#         else:
#             legal_indices.append(idx)
#     legal_probs = policy_probs[legal_indices]

#     # Normalize to sum to 1
#     legal_probs = legal_probs / legal_probs.sum()
#     move_probs = {
#         move: float(prob)
#         for move, prob in zip(legal_moves, legal_probs)
#     }

#     ctx.logProbabilities(move_probs)

#     # Pick the move with highest probability
#     best_move = max(move_probs, key=move_probs.get)
#     return best_move

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    Called each time the model needs to make a move.
    Returns a python-chess Move object (legal move) for current position.
    """
    try:
        print("Cooking move...")
        print(ctx.board.move_stack)

        return mcts(ctx)
    except Exception as e:
        print(e)


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
