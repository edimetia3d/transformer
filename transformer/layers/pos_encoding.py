import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_vec_len: int, dropout: float = 0.1, max_position=5000) -> None:
        super().__init__()

        assert embed_vec_len % 2 == 0

        pos = torch.arange(0, max_position, dtype=torch.float).unsqueeze(1)
        i_seq = torch.div(torch.arange(0, embed_vec_len, dtype=torch.int), 2, rounding_mode='trunc').unsqueeze(0)
        pe = pos / (10000 ** (2 * i_seq / embed_vec_len))
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])

        # Add a batch dimension: (1, max_positions, embed_vec_len)
        pe = pe.unsqueeze(0)

        # Register as non-learnable parameters
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x has shape (batch_size, seq_len, embed_vec_len)
        return has shape (batch_size, seq_len, embed_vec_len)

        note that PE has a fixed shape of (1, max_positions, embed_vec_len) where  seq_len == x.size(1) <= max_positions
        """
        # Max sequence length within the current batch
        seq_len = x.size(1)
        assert seq_len <= self.pe.size(1)

        # Add positional encoding up to the max sequence length
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)
