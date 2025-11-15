from torch.utils.data import DataLoader
import lightning as L
from model import ChessModel
from data import ChessPGNDataset

if __name__ == "__main__":
    dataset = ChessPGNDataset("your_games.pgn")

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ChessModel(num_moves=len(dataset.move_to_idx))

    trainer = L.Trainer(max_epochs=5)
    trainer.fit(model, train_loader)
