import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers import AutoConfig, AutoModel

class ChessModelConfig(PretrainedConfig):
    model_type = "chess-model"
    def __init__(self, input_channels=17, hidden_size=256, num_moves=4672, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_moves = num_moves
        self.input_channels = input_channels

class ChessModel(PreTrainedModel):
    config_class = ChessModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.conv = nn.Sequential(
            nn.Conv2d(config.input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        flat_size = 128 * 8 * 8

        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, config.num_moves)
        )

        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),          # scalar
            nn.Tanh()                   # restrict to [-1, 1]
        )

    def forward(self, x):
        h = self.conv(x)
        policy_logits = self.policy_head(h)
        value = self.value_head(h)
        return policy_logits, value