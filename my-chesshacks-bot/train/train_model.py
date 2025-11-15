import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import chess

from stockfish_eval_model import encode_board, ChessEvaluator

# ----------------------------
# Dataset Class for FEN → Tensor
# ----------------------------
class ChessDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.fens = df["fen"].tolist()
        self.scores = df["score"].astype(float).tolist()

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        board = chess.Board(self.fens[idx])
        x = encode_board(board)
        y = torch.tensor([self.scores[idx]], dtype=torch.float32)
        return x, y


# ----------------------------
# Training Loop
# ----------------------------
def train_model(csv_path, epochs=10, batch_size=64, lr=0.001):

    dataset = ChessDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ChessEvaluator()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() 

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for x, y in dataloader:
            x = x.float()
            y = y.float()

            # Add batch dimension to input: (B, 18, 8, 8)
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            predictions = model(x).squeeze(1)
            loss = criterion(predictions, y.squeeze(1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    # Save the trained model
    torch.save(model.state_dict(), "chess_evaluator.pth")
    print("Model saved to chess_evaluator.pth")

    return model


# ----------------------------
# Run Training
# ----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on:", device)

    train_model("archive/stockfish_positions3.csv", epochs=20, batch_size=128, lr=0.0005)
