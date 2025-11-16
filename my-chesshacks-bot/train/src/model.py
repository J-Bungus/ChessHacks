import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
BOARD_VEC_SIZE = 13

class ChessModelConfig(PretrainedConfig):
    model_type = "chess-transformer"

    def __init__(
        self,
        hidden_size=256,
        num_moves=4672,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_moves = num_moves
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

class ChessModel(PreTrainedModel):
    config_class = ChessModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.num_toks = 70

        H = config.hidden_size

        self.pos_embedding = nn.Parameter(torch.zeros(self.num_toks, H))
        self.state_embeddings = nn.ModuleList([nn.Embedding(2, H) for _ in range(5)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, H))

        # 6 piece types + empty space
        self.piece_embedding = nn.Embedding(BOARD_VEC_SIZE, H)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H,
            nhead=config.num_heads,
            dim_feedforward=H * 4,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )

        self.policy_head = nn.Sequential(
            nn.Linear(H, config.num_moves)
        )

        self.value_head = nn.Sequential(
            nn.Linear(H, H * 8),
            nn.ReLU(),
            nn.Linear(H * 8, 1),
            nn.Tanh()
        )

    def forward(self, x, state):
        B, D = x.shape
        assert D == 64

        tok = self.piece_embedding(x)      # [B, 64, H]

        state_toks = []

        for i in range(len(self.state_embeddings)):
            state_tok_i = self.state_embeddings[i](state[:, i])  # [B, H]
            state_toks.append(state_tok_i.unsqueeze(1))  # [B, 1, H]

        state_toks = torch.cat(state_toks, axis=1)
        cls = self.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls, state_toks, tok], axis=1) # [B, 69, H]

        tok = tok + self.pos_embedding

        h = self.transformer(tok)   # [B, 69, H]
        h = h[:, 0, :]

        policy_logits = self.policy_head(h)
        value = self.value_head(h)

        return policy_logits, value
