import torch
import torch.nn as nn


class BertEmbeddings(nn.Module):
    """
    Standard BERT embeddings: token + position + segment embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_seq_len: int,
        dropout: float = 0.1,
        use_rope: bool = False,
    ):
        super().__init__()
        self.use_rope = use_rope
        self.token_emb = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if not use_rope:
            self.position_emb = nn.Embedding(max_seq_len, hidden_size)
        self.segment_emb = nn.Embedding(2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, input_ids: torch.Tensor, segment_ids: torch.Tensor
    ) -> torch.Tensor:
        B, L = input_ids.shape
        emb = self.token_emb(input_ids) + self.segment_emb(segment_ids)
        if not self.use_rope:
            pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
            emb = emb + self.position_emb(pos)
        return self.dropout(self.norm(emb))

