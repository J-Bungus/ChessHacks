from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from data import SelfPlayDataModule, SelfPlayDataset
from utils import NUM_MOVES
from model import ChessModelConfig
from trainer import ChessTrainer

if __name__ == "__main__":
    wandb_logger = WandbLogger(
        project="chess-hacks",
        name="chess-hacks"
    )

    datamodule = SelfPlayDataModule(
        stockfish_path="/usr/games/stockfish",
        batch_size=256,
        num_train_games=1000,
        num_val_games=10
    )

    model_wrapper = ChessTrainer(
        model_config=ChessModelConfig(
            num_moves=NUM_MOVES,
            num_layers=4,
            hidden_size=256,
            num_heads=8
        ),
        lr=5e-4
    )

    trainer = L.Trainer(
        max_epochs=1000,
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=2
    )

    trainer.fit(model_wrapper, datamodule=datamodule)

    # Save to automodel
    model = model_wrapper.model
    model.push_to_hub("darren-lo/chess-bot-model")

    wandb_logger.experiment.finish()
