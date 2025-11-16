import chess
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from stockfish_eval_model import encode_pair, PositionRankNet

class PairRankingCSVDataset(Dataset):
    """
    Loads training_pairs.csv generated earlier.
    Each row: fen_a, fen_b, label
    """
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        fen_a = row["fen_a"]
        fen_b = row["fen_b"]
        label = row["label"]

        board_a = chess.Board(fen_a)
        board_b = chess.Board(fen_b)

        # YOUR encoder functions (already implemented earlier)
        x = encode_pair(board_a, board_b)    # (36, 8, 8)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor([label], dtype=torch.float32)  # (1,)

        return x, y

import torch
from torch.utils.data import DataLoader

def train_ranking_model(
    csv_path,
    model,
    epochs=5,
    batch_size=32,
    lr=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    dataset = PairRankingCSVDataset(csv_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in tqdm(loader):
            x = x.to(device)         # (batch, 36, 8, 8)
            y = y.to(device)         # (batch, 1)

            optimizer.zero_grad()
            preds = model(x)         # (batch, 1)

            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        epoch_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{epochs}  Loss: {epoch_loss:.4f}")

    return model

model = PositionRankNet()

trained_model = train_ranking_model(
    csv_path="train.csv",
    model=model,
    epochs=5,
    batch_size=64,
    lr=1e-4
)

torch.save(trained_model.state_dict(), "position_rank_net.pth")
print("Model saved to position_rank_net.pth")

