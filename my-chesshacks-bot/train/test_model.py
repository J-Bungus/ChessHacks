import torch
import chess

from stockfish_eval_model import encode_board, ChessEvaluator

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ChessEvaluator().to(device)
model.load_state_dict(torch.load("archive/chess_evaluator3-1.pth", map_location=device))
model.eval()

print("Model loaded successfully!")

def evaluate_fen(fen: str):
    board = chess.Board(fen)
    x = encode_board(board).unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(x).item()
    return score

def evaluate_board(board: chess.Board):
    x = encode_board(board).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(x).item()


if __name__ == "__main__":
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    print("Evaluation:", evaluate_fen(fen))

    board = chess.Board()
    print("Start position evaluation:", evaluate_board(board))

