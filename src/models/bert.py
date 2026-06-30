import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers.attention import MultiHeadSelfAttention
from src.models.layers.feedforward import FeedForward
from src.models.layers.embedding import BertEmbeddings
from src.models.layers.encoder import EncoderLayer


class BertEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ff_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        use_rope: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(hidden_size, ff_dim, num_heads, dropout, use_rope=use_rope)
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
        self.cfg = cfg
        self.embeddings = BertEmbeddings(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            max_seq_len=cfg.max_seq_len,
            dropout=cfg.dropout,
            use_rope=getattr(cfg, "use_rope", False),
        )
        self.encoder = BertEncoder(
            hidden_size=cfg.hidden_size,
            ff_dim=cfg.ff_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            use_rope=getattr(cfg, "use_rope", False),
        )
        self.norm = nn.LayerNorm(cfg.hidden_size)

        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_size),
            nn.Linear(cfg.hidden_size, cfg.vocab_size),
        )
        # Tie weights between token embeddings and output projection layer
        self.mlm_head[3].weight = self.embeddings.token_emb.weight

    def forward(self, input_ids, segment_ids, mask=None, mlm_labels=None):
        if mask is None:
            mask = (input_ids != self.cfg.pad_token_id).unsqueeze(1).unsqueeze(2)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(2)

        x = self.embeddings(input_ids, segment_ids)
        x = self.encoder(x, mask=mask)
        x = self.norm(x)  # (B, L, H)

        # MLM: all positions
        mlm_logits = self.mlm_head(x)  # (B, L, vocab_size)

        loss = None
        if mlm_labels is not None:
            loss = F.cross_entropy(
                mlm_logits.view(-1, self.cfg.vocab_size),
                mlm_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "mlm_logits": mlm_logits}

