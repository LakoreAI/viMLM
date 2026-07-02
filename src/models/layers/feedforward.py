import torch
import torch.nn as nn
import torch.nn.functional as F

VALID_FFN_TYPES = ("gelu", "swiglu", "geglu")


class FeedForward(nn.Module):
    """Feed-forward network used in the encoder layer.

    Args:
        hidden_size (int): The dimension of the input and output features.
        ff_dim (int): The dimension of the hidden layer.
        dropout (float): The dropout rate.
        ffn_type (str): One of "gelu" (plain Linear -> GELU -> Linear),
            "swiglu" (Shazeer 2020, used in LLaMA/PaLM), or "geglu"
            (used in Gemma/ModernBERT).

    For the gated variants, the gate/up projection width is scaled to
    ~2/3 * ff_dim so the total parameter count roughly matches the plain
    GELU FFN (which only has one up-projection instead of two).
    """

    def __init__(
        self,
        hidden_size: int,
        ff_dim: int,
        dropout: float = 0.1,
        ffn_type: str = "gelu",
    ):
        super().__init__()
        assert ffn_type in VALID_FFN_TYPES, (
            f"ffn_type must be one of {VALID_FFN_TYPES}, got {ffn_type!r}"
        )
        self.ffn_type = ffn_type

        if ffn_type in ("swiglu", "geglu"):
            gate_dim = max(1, int(ff_dim * 2 / 3))
            self.w_gate = nn.Linear(hidden_size, gate_dim, bias=False)
            self.w_up = nn.Linear(hidden_size, gate_dim, bias=False)
            self.w_down = nn.Linear(gate_dim, hidden_size, bias=False)
            self.dropout = nn.Dropout(dropout)
            self.act_fn = F.silu if ffn_type == "swiglu" else F.gelu
            self.net = None
        else:
            self.net = nn.Sequential(
                nn.Linear(hidden_size, ff_dim, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, hidden_size, bias=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net is not None:
            return self.net(x)
        gated = self.act_fn(self.w_gate(x)) * self.w_up(x)
        return self.w_down(self.dropout(gated))
