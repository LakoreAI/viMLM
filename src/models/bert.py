import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers.attention import MultiHeadSelfAttention
from src.models.layers.feedforward import FeedForward
from src.models.layers.embedding import BertEmbeddings
from src.models.layers.encoder import EncoderLayer, UNetEncoderLayer


class BertEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ff_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        use_rope: bool = False,
        use_unet_shrink: bool = False,
        unet_bottleneck_ratio: float = 0.5,
        layer_hidden_sizes: list = None,
        use_unet_skip: bool = False,
    ):
        super().__init__()
        self.layer_sizes = []
        if use_unet_shrink:
            bottleneck_size = int(hidden_size * unet_bottleneck_ratio)
            if num_layers > 1:
                mid = (num_layers - 1) / 2
                for i in range(num_layers):
                    dist = min(i, num_layers - 1 - i)
                    r = dist / mid if mid > 0 else 0.0
                    size = hidden_size - r * (hidden_size - bottleneck_size)
                    # Round to nearest multiple of 2 * num_heads to ensure head_dim is even (required for RoPE)
                    size = max(
                        2 * num_heads,
                        round(size / (2 * num_heads)) * (2 * num_heads),
                    )
                    self.layer_sizes.append(size)
            else:
                self.layer_sizes = [hidden_size]
        elif layer_hidden_sizes is not None:
            self.layer_sizes = layer_hidden_sizes
        else:
            self.layer_sizes = [hidden_size] * num_layers

        print(f"Encoder Layer Hidden Sizes: {self.layer_sizes}")

        is_unet = use_unet_shrink or layer_hidden_sizes is not None

        # Pair each "down-path" layer index with its symmetric "up-path" index
        # (same out_features), so the up-path layer can fuse in the cached
        # down-path activation the way a U-Net decoder block concatenates the
        # matching encoder feature map. The true bottleneck layer (i == j)
        # has no pair, same as real U-Net.
        self.use_skip = use_unet_skip and is_unet
        self.pair_of = {}
        if self.use_skip:
            n = len(self.layer_sizes)
            for i in range(n):
                j = n - 1 - i
                if i < j and self.layer_sizes[i] == self.layer_sizes[j]:
                    self.pair_of[j] = i
        self.skip_source_idxs = set(self.pair_of.values())

        self.layers = nn.ModuleList()
        in_features = hidden_size
        for i, out_features in enumerate(self.layer_sizes):
            ff_dim_i = int(out_features * (ff_dim / hidden_size))
            if is_unet:
                self.layers.append(
                    UNetEncoderLayer(
                        in_features=in_features,
                        out_features=out_features,
                        ff_dim=ff_dim_i,
                        num_heads=num_heads,
                        dropout=dropout,
                        use_rope=use_rope,
                        use_skip=i in self.pair_of,
                    )
                )
            else:
                self.layers.append(
                    EncoderLayer(
                        hidden_size=out_features,
                        ff_dim=ff_dim_i,
                        num_heads=num_heads,
                        dropout=dropout,
                        use_rope=use_rope,
                    )
                )
            in_features = out_features

        if in_features != hidden_size:
            self.final_proj = nn.Linear(in_features, hidden_size)
        else:
            self.final_proj = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        down_cache = {}
        for i, layer in enumerate(self.layers):
            if self.use_skip and i in self.pair_of:
                x = layer(x, mask, skip=down_cache.get(self.pair_of[i]))
            else:
                x = layer(x, mask)
            if self.use_skip and i in self.skip_source_idxs:
                down_cache[i] = x
        if self.final_proj is not None:
            x = self.final_proj(x)
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
            use_unet_shrink=getattr(cfg, "use_unet_shrink", False),
            unet_bottleneck_ratio=getattr(cfg, "unet_bottleneck_ratio", 0.5),
            layer_hidden_sizes=getattr(cfg, "layer_hidden_sizes", None),
            use_unet_skip=getattr(cfg, "use_unet_skip", False),
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
