import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Feed-forward network used in the ViLM architecture.

    Attributes:
        net: Sequential model containing linear layers and GELU activation.

    Args:
        hidden_size (int): The dimension of the input and output features.
        ff_dim (int): The dimension of the hidden layer.
        dropout (float): The dropout rate.
    """

    def __init__(self, hidden_size: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
