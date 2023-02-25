import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    "Same as nn.LayerNorm"

    def __init__(self, embed_vec_len: int, eps: float = 1e-5):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(embed_vec_len))
        self.b_2 = nn.Parameter(torch.zeros(embed_vec_len))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x and return both has shape (batch_size, seq_len, embed_vec_len)"""
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.a_2 * (x - mean) / torch.sqrt(var + self.eps) + self.b_2
