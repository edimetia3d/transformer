import torch
import math
from torch import nn


def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None, dropout: nn.Dropout = None):
    """
    Q shape: (batch_size, y_seq_len, d_k)
    K shape: (batch_size, x_seq_len, d_k)
    V shape: (batch_size, x_seq_len, d_v)
    mask shape: (batch_size,y_seq_len, x_seq_len)
    return shape (batch_size, y_seq_len, d_v)

    Note that the local var `score` and `p_attn` both has a shape of (batch_size, y_seq_len, x_seq_len)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, V), p_attn


class SingleHeadAttention(nn.Module):
    def __init__(self,
                 x_embed_vec_len: int,
                 y_embed_vec_len: int,
                 dropout: float,
                 d_k: int = None,
                 d_v: int = None):
        super(SingleHeadAttention, self).__init__()
        if d_k is None:
            d_k = max(x_embed_vec_len, y_embed_vec_len)
        if d_v is None:
            d_v = y_embed_vec_len

        self.Wq = nn.Linear(y_embed_vec_len, d_k)
        self.Wk = nn.Linear(x_embed_vec_len, d_k)
        self.Wv = nn.Linear(x_embed_vec_len, d_v)

        self.dropout = nn.Dropout(dropout)

    def forward(self, y: torch.Tensor, x: torch.Tensor, mask: torch.Tensor = None):
        """
        y shape: (batch_size, y_seq_len, y_embed_vec_len)
        x shape: (batch_size, x_seq_len, x_embed_vec_len)
        mask shape: (batch_size, y_seq_len, x_seq_len)
        return shape: (batch_size, y_seq_len, d_v)

        """
        Q = self.Wq(y)
        K = self.Wk(x)
        V = self.Wv(x)
        return attention(Q, K, V, mask, dropout=self.dropout)


class MultiHeadAttention(nn.Module):
    """A naive multi-head attention module, which is a linear combination of single-head attention modules.

    Note:
        This implementation is not efficient, it only serves as a simple example to make the code more readable.

    """

    def __init__(self,
                 x_embed_vec_len: int,
                 y_embed_vec_len: int,
                 num_heads: int,
                 dropout: float,
                 d_k: int = None,
                 final_d_v: int = None):
        """
        Args:
            embed_vec_len: the length of embedding vector
        """
        super(MultiHeadAttention, self).__init__()
        if d_k is None:
            d_k = max(x_embed_vec_len, y_embed_vec_len)
        if final_d_v is None:
            final_d_v = y_embed_vec_len

        self.attention_heads = []
        if final_d_v % num_heads == 0:
            single_head_output_dim = final_d_v // num_heads
        else:
            single_head_output_dim = (final_d_v + num_heads - 1) // num_heads
        for _ in range(num_heads):
            self.attention_heads.append(SingleHeadAttention(
                x_embed_vec_len,
                y_embed_vec_len,
                dropout,
                d_k,
                single_head_output_dim))
        if final_d_v != single_head_output_dim * num_heads:
            self.Wo = nn.Linear(single_head_output_dim * num_heads, final_d_v)
        else:
            self.Wo = None

    def forward(self, y, x, mask=None):
        vs = []
        for attention_head in self.attention_heads:
            v, _ = attention_head(y, x, mask)
            vs.append(v)
        ret = torch.cat(vs, dim=-1)
        if self.Wo is not None:
            return self.Wo(ret)
        return ret
