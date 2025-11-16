from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from data import ChessPolicyValueDataset
from utils import NUM_MOVES
from model import ChessModelConfig
from trainer import ChessTrainer

if __name__ == "__main__":
    wandb_logger = WandbLogger(
        project="chess-hacks",
        name="chess-hacks"
    )

    dataset = ChessPolicyValueDataset(
        n_positions=32,
        max_random_moves=30,
        stockfish_path="/usr/games/stockfish",
        stockfish_depth=12,
        multipv=5
    )

    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True
    )

    model_wrapper = ChessTrainer(
        model_config=ChessModelConfig(
            num_moves=NUM_MOVES
        )
    )

    trainer = L.Trainer(
        max_epochs=5,
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=10  # optional, smoother wandb logs
    )

    trainer.fit(model_wrapper, train_loader)

    # Save to automodel
    model = model_wrapper.model
    model.push_to_hub("darren-lo/chess-bot-model")

    wandb_logger.experiment.finish()
