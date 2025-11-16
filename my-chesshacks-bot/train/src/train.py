from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from data import OnlineSelfPlayDataset
from utils import NUM_MOVES
from model import ChessModelConfig
from trainer import ChessTrainer

if __name__ == "__main__":
    wandb_logger = WandbLogger(
        project="chess-hacks",
        name="chess-hacks"
    )

    dataset = OnlineSelfPlayDataset(
        n_positions=1000,
        max_moves=65,
        stockfish_path="/usr/games/stockfish",
        movetime=0.01
    )

    print("Dataset size: ", len(dataset))

    train_loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=0
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
    model.push_to_hub("jbungus/chess-bot-model")

    wandb_logger.experiment.finish()
