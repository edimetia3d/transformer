import torch.nn as nn

from typing import List

import transformer.layers.embedding as embedding_layer
import transformer.layers.pos_encoding as pos_encoding_layer
import transformer.layers.encoder as encoder_layer
import transformer.layers.decoder as decoder_layer
from transformer import common


class Encoder:
    encoder_layers: List[encoder_layer.EncoderLayer]

    def __init__(self, setting: common.CodexSetting):
        self.encoder_layers = []
        for _ in range(setting.repeat_count):
            self.encoder_layers.append(encoder_layer.EncoderLayer(embed_vec_len=setting.d_model,
                                                                  ffn_hidden_size=setting.ffn_hidden_size,
                                                                  num_heads=setting.num_heads,
                                                                  dropout=setting.dropout))


class Decoder:
    decoder_layers: List[decoder_layer.DecoderLayer]

    def __init__(self, setting: common.CodexSetting):
        self.decoder_layers = []
        for _ in range(setting.repeat_count):
            self.decoder_layers.append(decoder_layer.DecoderLayer(embed_vec_len=setting.d_model,
                                                                  ffn_hidden_size=setting.ffn_hidden_size,
                                                                  num_heads=setting.num_heads,
                                                                  dropout=setting.dropout))


class EncoderDecoder(nn.Module):
    encoder: Encoder
    src_embed: embedding_layer.Embedding
    src_pe = pos_encoding_layer.PositionalEncoding

    decoder: Decoder
    tgt_embed: embedding_layer.Embedding
    tgt_pe = pos_encoding_layer.PositionalEncoding

    def __init__(self,
                 src_vocab_sz: int,
                 tgt_vocab_sz: int,
                 enc_setting: common.CodexSetting = None,
                 dec_setting: common.CodexSetting = None):
        if enc_setting is None:
            enc_setting = common.CodexSetting.default_setting()
        if dec_setting is None:
            dec_setting = common.CodexSetting.default_setting()

        self.encoder = Encoder(enc_setting)
        self.src_pe = pos_encoding_layer.PositionalEncoding(enc_setting.d_model)
        self.src_embed = embedding_layer.Embedding(src_vocab_sz, enc_setting.d_model)

        self.decoder = Decoder(dec_setting)
        self.tgt_pe = pos_encoding_layer.PositionalEncoding(dec_setting.d_model)
        self.src_embed = embedding_layer.Embedding(tgt_vocab_sz, dec_setting.d_model)
