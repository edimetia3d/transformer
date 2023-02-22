# Naming convention note: The "Decoder" contains multiple "DecoderLayer", each layer has the same structure
class DecoderLayer:
    """

    Note:
        For DecoderLayer will be reused multiple times in transformer, it is designed to have the X and Y in
        `Y = DecoderLayer(X)` has same shape
    """

    def __init__(self, embed_vec_len: int, ffn_hidden_size: int, num_heads: int, dropout: float):
        """

        Args:
            embed_vec_len: the length of embedding vector
        """
        pass
