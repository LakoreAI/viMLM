import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers.embedding import RotaryEmbedding, apply_rotary_pos_emb


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        dropout: float = 0.1,
        use_rope: bool = False,
    ):
        super().__init__()
        assert (
            hidden_size % num_heads == 0
        ), f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        self.h = num_heads
        self.d_k = hidden_size // num_heads
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.use_rope = use_rope
        if use_rope:
            self.rotary_emb = RotaryEmbedding(self.d_k)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, L, D = x.shape

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, L, self.h, self.d_k).transpose(1, 2)

        Q, K, V = split(self.W_q(x)), split(self.W_k(x)), split(self.W_v(x))

        if self.use_rope:
            cos, sin = self.rotary_emb(Q, seq_len=L)
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        attn_mask = None
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            attn_mask = mask.bool()

        # FlashAttention / memory-efficient kernel via PyTorch SDPA: tiles the
        # (B, H, L, L) score matrix instead of materializing it in HBM.
        out = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)
