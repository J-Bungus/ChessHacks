from torch.utils.data import DataLoader
import lightning as L

from model import ChessPolicyModel        # policy-only model (4672 moves)
from data import ChessPolicyDataset       # policy dataset you just requested

if __name__ == "__main__":
    # Create dataset
    dataset = ChessPolicyDataset(
        n_positions=1000,
        max_random_moves=30,
        stockfish_path="/opt/homebrew/bin/stockfish",
        stockfish_depth=12
    )

    print(f"Dataset length: {len(dataset)}")  # must be >0

    # PyTorch DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True
    )

    # Policy model: outputs logits over 4672 moves
    model = ChessPolicyModel(num_moves=4672)

    # Lightning trainer
    trainer = L.Trainer(
        max_epochs=5,
        devices=1
    )

    trainer.fit(model, train_loader)
