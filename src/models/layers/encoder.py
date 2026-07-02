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
        ffn_type: str = "gelu",
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            num_heads, hidden_size, dropout, use_rope=use_rope
        )
        self.ff = FeedForward(hidden_size, ff_dim, dropout, ffn_type=ffn_type)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class UNetEncoderLayer(nn.Module):
    """
    Encoder layer used in the U-Net-shaped stack. `use_skip=True` marks an
    "up-path" layer that receives the cached activation from its symmetric
    "down-path" layer (same out_features) and fuses it in, mirroring how a
    real U-Net decoder block concatenates the matching encoder feature map
    before convolving back down to the target channel count.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ff_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rope: bool = False,
        use_skip: bool = False,
        ffn_type: str = "gelu",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if in_features != out_features:
            self.proj_in = nn.Linear(in_features, out_features)
        else:
            self.proj_in = None

        self.attn = MultiHeadSelfAttention(
            num_heads, out_features, dropout, use_rope=use_rope
        )
        self.ff = FeedForward(out_features, ff_dim, dropout, ffn_type=ffn_type)
        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

        self.use_skip = use_skip
        if use_skip:
            self.skip_fuse = nn.Linear(out_features * 2, out_features)
            self.skip_norm = nn.LayerNorm(out_features)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, skip: torch.Tensor = None
    ) -> torch.Tensor:
        if self.proj_in is not None:
            x = self.proj_in(x)
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        if self.use_skip and skip is not None:
            x = self.skip_norm(self.skip_fuse(torch.cat([x, skip], dim=-1)))
        return x
