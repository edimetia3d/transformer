import torch
import torch.nn as nn
import math


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_vec_len: int):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab_size, embed_vec_len)
        self._embed_vec_len = embed_vec_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x has shape (batch_size, seq_len)
        return has shape (batch_size, seq_len, embed_vec_len)
        """
        return self.lut(x) * math.sqrt(self._embed_vec_len)
