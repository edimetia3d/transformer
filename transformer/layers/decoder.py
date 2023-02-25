import torch
import torch.nn as nn
from transformer.layers.sub_layers import multi_head_attention
from transformer.layers.sub_layers import ffn
from transformer.layers.sub_layers import layernorm


# Naming convention note: The "Decoder" contains multiple "DecoderLayer", each layer has the same structure
class DecoderLayer(nn.Module):
    """

    Note:
        For DecoderLayer will be reused multiple times in transformer, it is designed to have the X and Y in
        `Y = DecoderLayer(X)` has same shape
    """

    self_attn: multi_head_attention.MultiHeadAttention
    cross_attn: multi_head_attention.MultiHeadAttention
    ffn: ffn.PositionwiseFFN

    def __init__(self, tgt_embed_vec_len: int, src_embed_vec_len: int, ffn_hidden_size: int, num_heads: int,
                 dropout: float):
        super(DecoderLayer, self).__init__()
        self.self_attn = multi_head_attention.MultiHeadAttention(x_embed_vec_len=tgt_embed_vec_len,
                                                                 y_embed_vec_len=tgt_embed_vec_len,
                                                                 num_heads=num_heads,
                                                                 dropout=dropout,
                                                                 final_d_v=tgt_embed_vec_len)
        self.layer_norm0 = layernorm.LayerNorm(tgt_embed_vec_len)
        self.dropout0 = nn.Dropout(dropout)

        self.cross_attn = multi_head_attention.MultiHeadAttention(x_embed_vec_len=src_embed_vec_len,
                                                                  y_embed_vec_len=tgt_embed_vec_len,
                                                                  num_heads=num_heads,
                                                                  dropout=dropout,
                                                                  final_d_v=tgt_embed_vec_len)
        self.layer_norm1 = layernorm.LayerNorm(tgt_embed_vec_len)
        self.dropout1 = nn.Dropout(dropout)

        self.feed_forward = ffn.PositionwiseFFN(tgt_embed_vec_len, ffn_hidden_size, dropout)
        self.layer_norm2 = layernorm.LayerNorm(tgt_embed_vec_len)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, y: torch.Tensor, mask: torch.Tensor, context: torch.Tensor, src_mask: torch.Tensor):
        """
        y shape: (batch_size, tgt_seq_len, tgt_embed_vec_len)
        mask shape: (batch_size, tgt_seq_len, tgt_seq_len), it is the padding_mask & subsequent_mask
        context shape: (batch_size, src_seq_len, src_embed_vec_len)
        src_mask shape: (batch_size, 1, src_seq_len)
        return shape: (batch_size, tgt_seq_len, tgt_embed_vec_len)
        """
        y_tmp = self.layer_norm0(y)
        y = y + self.dropout0(self.self_attn(y_tmp, y_tmp, mask))

        y_tmp = self.layer_norm1(y)
        y = y + self.dropout1(self.cross_attn(y_tmp, context, src_mask))

        y_tmp = self.layer_norm2(y)
        return y + self.dropout2(self.feed_forward(y_tmp))
