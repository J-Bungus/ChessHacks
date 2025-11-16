from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from data import SelfPlayDataset
from utils import NUM_MOVES
from model import ChessModelConfig
from trainer import ChessTrainer

if __name__ == "__main__":
    wandb_logger = WandbLogger(
        project="chess-hacks",
        name="chess-hacks"
    )

    dataset = SelfPlayDataset(
        stockfish_path="/usr/games/stockfish",
        num_games=50
    )

    print("Dataset size: ", len(dataset))

    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True
    )

    model_wrapper = ChessTrainer(
        model_config=ChessModelConfig(
            num_moves=NUM_MOVES,
            num_layers=6,
            hidden_size=512,
            num_heads=8
        ),
        lr=1e-5
    )

    trainer = L.Trainer(
        max_epochs=100,
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=10  # optional, smoother wandb logs
    )

    trainer.fit(model_wrapper, train_loader)

    # Save to automodel
    model = model_wrapper.model
    model.push_to_hub("darren-lo/chess-bot-model")

    wandb_logger.experiment.finish()
