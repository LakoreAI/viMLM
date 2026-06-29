import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers.attention import MultiHeadSelfAttention
from src.models.layers.feedforward import FeedForward
from src.models.layers.embedding import BertEmbeddings


class BertEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ff_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(hidden_size, ff_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class BertForPreTraining(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embeddings = BertEmbeddings(cfg)
        self.encoder = BertEncoder(cfg)
        self.norm = nn.LayerNorm(cfg.hidden_size)

        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_size),
            nn.Linear(cfg.hidden_size, cfg.vocab_size),
        )

        # NSP head  — pools [CLS] token (position 0)
        #   P(y | A,B) = softmax(W * h_[CLS] + b),  y ∈ {IsNext, NotNext}
        self.nsp_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.Tanh(),
            nn.Linear(cfg.hidden_size, 2),  # binary
        )

    def forward(self, input_ids, segment_ids, mlm_labels=None, nsp_labels=None):
        pad_mask = (input_ids != cfg.pad_token_id).unsqueeze(1).unsqueeze(2)

        x = self.embeddings(input_ids, segment_ids)
        x = self.encoder(x, mask=pad_mask)
        x = self.norm(x)  # (B, L, H)

        # MLM: all positions
        mlm_logits = self.mlm_head(x)  # (B, L, vocab_size)

        # NSP: [CLS] is always position 0
        cls_hidden = x[:, 0, :]  # (B, H)
        nsp_logits = self.nsp_head(cls_hidden)  # (B, 2)

        loss = None
        if mlm_labels is not None and nsp_labels is not None:
            loss_mlm = F.cross_entropy(
                mlm_logits.view(-1, cfg.vocab_size),
                mlm_labels.view(-1),
                ignore_index=-100,
            )
            loss_nsp = F.cross_entropy(nsp_logits, nsp_labels)
            loss = loss_mlm + loss_nsp  # equal weighting as in paper

        return {"loss": loss, "mlm_logits": mlm_logits, "nsp_logits": nsp_logits}
