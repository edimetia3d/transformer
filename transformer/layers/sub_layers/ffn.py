import torch
from torch import nn


class PositionwiseFFN(nn.Module):
    def __init__(self, embed_vec_len: int, hidden_size: int, dropout: float = 0.1):
        super(PositionwiseFFN, self).__init__()
        self.w_1 = nn.Linear(embed_vec_len, hidden_size)
        self.w_2 = nn.Linear(hidden_size, embed_vec_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x and return both has shape (batch_size, seq_len, embed_vec_len)"""
        return self.w_2(self.dropout(self.w_1(x).relu()))
