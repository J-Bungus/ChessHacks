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

    def training_step(self, batch, batch_idx):
        x, pi_target, z_target = batch

        policy_logits, value_pred = self.model(x)

        # --- Policy loss:
        if pi_target.dtype == torch.long or pi_target.dtype == torch.int64:
            # pi_target is a class index: shape [B]
            policy_loss = F.cross_entropy(policy_logits, pi_target)
        else:
            # pi_target is a full distribution over moves: shape [B, num_moves]
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(pi_target * log_probs).sum(dim=1).mean()

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
