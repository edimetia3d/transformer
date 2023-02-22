from transformer.layers.sub_layers import multi_head_attention
from transformer.layers.sub_layers import ffn


# Naming convention note: The "Encoder" contains multiple "EncoderLayer", each layer has the same structure
class EncoderLayer:
    """

    Note:
        For EncoderLayer will be reused multiple times in transformer, it is designed to have the X and Y in
        `Y = EncoderLayer(X)` has same shape
    """

    self_attn: multi_head_attention.MultiHeadAttention
    ffn: ffn.PositionwiseFFN

    def __init__(self, embed_vec_len: int, ffn_hidden_size: int, num_heads: int, dropout: float):
        """

        Args:
            embed_vec_len: the length of embedding vector
        """
        pass
