import pytorch_lightning as pl
import torch.nn as nn
import torch

class ChessModel(pl.LightningModule):
    def __init__(self, num_moves, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.net = nn.Sequential(
            nn.Conv2d(18, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, num_moves)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
