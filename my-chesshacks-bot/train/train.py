from torch.utils.data import DataLoader
import lightning as L
from model import ChessModel
from data import AlphaZeroChessDataset

if __name__ == "__main__":
    dataset = AlphaZeroChessDataset(
        n_positions=1000,
        max_random_moves=30,
        stockfish_path="/usr/bin/stockfish",
        stockfish_depth=12
    )

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ChessModel(num_moves=len(dataset.move_to_idx))

    trainer = L.Trainer(max_epochs=5)
    trainer.fit(model, train_loader)
