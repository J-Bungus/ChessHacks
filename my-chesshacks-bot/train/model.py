import lightning as L
import torch.nn as nn
import torch

class ChessPolicyModel(L.LightningModule):
    def __init__(self, num_moves=4672, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.num_moves = num_moves

        self.net = nn.Sequential(
            nn.Conv2d(18, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, num_moves)  # logits for all moves
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)  # logits: (B, 4672)

    def training_step(self, batch, batch_idx):
        x, move_idx = batch   # move_idx is integer in [0, 4671]
        logits = self(x)
        loss = self.loss_fn(logits, move_idx)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
