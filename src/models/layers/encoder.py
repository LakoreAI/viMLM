import torch
import torch.nn as nn
from src.models.layers.attention import MultiHeadSelfAttention
from src.models.layers.feedforward import FeedForward


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ff_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rope: bool = False,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            num_heads, hidden_size, dropout, use_rope=use_rope
        )
        self.ff = FeedForward(hidden_size, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x
