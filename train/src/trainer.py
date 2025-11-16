import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ChessModelConfig, ChessModel

class ChessTrainer(L.LightningModule):
    def __init__(self, model_config: ChessModelConfig, lr=1e-3, value_coeff=1.0):
        super().__init__()
        self.save_hyperparameters()

        self.value_coeff = value_coeff
        self.model = ChessModel(model_config)
        self.upload_freq = 10


    def on_train_epoch_start(self):
        self.trainer.train_dataloader.dataset.resample(fraction=0.001)

    def on_train_epoch_end(self):
        epoch = self.current_epoch
        if (epoch + 1) % self.upload_freq == 0:  # +1 if you want epoch counting from 1
            branch_name = "checkpoint"
            print(f"Pushing model to Hugging Face Hub: {branch_name}")
            self.model.push_to_hub(
                "darren-lo/chess-bot-model",
                commit_message=f"Checkpoint after epoch {epoch+1}",
                branch=branch_name
            )


    def training_step(self, batch, batch_idx):
        x, state, pi_target, z_target = batch

        policy_logits, value_pred = self.model(x, state)

        # --- Policy loss: cross-entropy on full distribution ---
        policy_loss = F.cross_entropy(policy_logits, pi_target)

        # --- Value loss ---
        value_pred = value_pred.squeeze(-1)
        z_target = z_target.view_as(value_pred)
        value_loss = F.mse_loss(value_pred, z_target)

        # --- Total loss ---
        loss = policy_loss + self.value_coeff * value_loss

        self.log("train_policy_loss", policy_loss, prog_bar=True)
        self.log("train_value_loss", value_loss, prog_bar=True)
        self.log("train_total_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, state, pi_target, z_target = batch

        policy_logits, value_pred = self.model(x, state)

        policy_loss = F.cross_entropy(policy_logits, pi_target)
        value_pred = value_pred.squeeze(-1)
        z_target = z_target.view_as(value_pred)
        value_loss = F.mse_loss(value_pred, z_target)

        loss = policy_loss + self.value_coeff * value_loss

        # Log validation losses
        self.log("val_policy_loss", policy_loss, prog_bar=True, sync_dist=True)
        self.log("val_value_loss", value_loss, prog_bar=True, sync_dist=True)
        self.log("val_total_loss", loss, prog_bar=True, sync_dist=True)

        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
