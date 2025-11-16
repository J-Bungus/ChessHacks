import torch
import pandas as pd
import chess
import chess.pgn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from stockfish_eval_model import encode_pair, PositionRankNet

device = "cuda" if torch.cuda.is_available() else "cpu"

model = PositionRankNet().to(device)
model.load_state_dict(torch.load("position_rank_net.pth", map_location=device))
model.eval()

print("Model loaded successfully!")

class ValidationPairs(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        board_a = chess.Board(row["fen_a"])
        board_b = chess.Board(row["fen_b"])
        label = row["label"]  # +1 or -1

        x = encode_pair(board_a, board_b)  # ← USE YOUR EXISTING FUNCTION
        return x, torch.tensor(label, dtype=torch.float32)
    
def validate(csv_path, batch_size=64):
    dataset = ValidationPairs(csv_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    mse_loss = torch.nn.MSELoss(reduction="sum")
    total_loss = 0

    with torch.no_grad():
        for x, labels in tqdm(loader, desc="Validating"):
            preds = model(x)  # shape (batch, 1)
            preds = preds.view(-1)

            labels = labels.float()

            # pairwise accuracy (sign match)
            sign_correct = (preds * labels > 0).sum().item()
            correct += sign_correct
            total += len(labels)

            # accumulate MSE loss
            total_loss += mse_loss(preds, labels).item()

    accuracy = correct / total
    avg_loss = total_loss / total

    print("\n==============================")
    print(f"Pairwise accuracy: {accuracy:.4f}")
    print(f"Average MSE loss : {avg_loss:.4f}")
    print("==============================")


validate("validation.csv")
