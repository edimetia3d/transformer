import torch
import torch.nn as nn
from transformer.layers.sub_layers import multi_head_attention
from transformer.layers.sub_layers import ffn
from transformer.layers.sub_layers import layernorm


# Naming convention note: The "Encoder" contains multiple "EncoderLayer", each layer has the same structure
class EncoderLayer(nn.Module):

    def __init__(self, embed_vec_len: int, ffn_hidden_size: int, num_heads: int, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = multi_head_attention.MultiHeadAttention(x_embed_vec_len=embed_vec_len,
                                                                 y_embed_vec_len=embed_vec_len,
                                                                 num_heads=num_heads,
                                                                 dropout=dropout,
                                                                 final_d_v=embed_vec_len)
        self.layer_norm0 = layernorm.LayerNorm(embed_vec_len)
        self.dropout0 = nn.Dropout(dropout)

        self.feed_forward = ffn.PositionwiseFFN(embed_vec_len, ffn_hidden_size, dropout)
        self.layer_norm1 = layernorm.LayerNorm(embed_vec_len)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, src_seq_len, src_embed_vec_len)
        mask shape: (batch_size, 1, src_seq_len)
        return shape: (batch_size, src_seq_len, src_embed_vec_len)
        """
        x_tmp = self.layer_norm0(x)
        x = x + self.dropout0(self.self_attn(x_tmp, x_tmp, mask))
        x_tmp = self.layer_norm1(x)
        return x + self.dropout1(self.feed_forward(x_tmp))
