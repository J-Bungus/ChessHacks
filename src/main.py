from .utils import chess_manager, GameContext

import torch
from .model import PositionRankNet, rank_siblings

device = "cuda" if torch.cuda.is_available() else "cpu"

model = PositionRankNet().to(device)
model.load_state_dict(torch.load("src/model/800k_pairs.pth", map_location=device))
model.eval()  # VERY IMPORTANT!


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Cooking move...")
    print(ctx.board.move_stack)
    # time.sleep(0.1)

    moves = list(ctx.board.legal_moves)
    children = []

    for move in moves:
        child = ctx.board.copy()
        child.push(move)
        children.append(child)
    
    ranked_children = rank_siblings(children, ctx.board, model, device)
    best_child = ranked_children[0]
    return best_child.move_stack[-1]


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
