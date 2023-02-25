import torch
import torch.nn as nn

from typing import List

import transformer.layers.embedding as embedding_layer
import transformer.layers.pos_encoding as pos_encoding_layer
import transformer.layers.encoder as encoder_layer
import transformer.layers.decoder as decoder_layer
import transformer.layers.sub_layers.layernorm as layernorm_layer
from transformer import common


class Encoder(nn.Module):
    layers: List[encoder_layer.EncoderLayer]
    norm: nn.LayerNorm

    def __init__(self, setting: common.CodexSetting):
        super(Encoder, self).__init__()
        self.layers = []
        for _ in range(setting.repeat_count):
            self.layers.append(encoder_layer.EncoderLayer(
                embed_vec_len=setting.d_model,
                ffn_hidden_size=setting.ffn_hidden_size,
                num_heads=setting.num_heads,
                dropout=setting.dropout))
        self.norm = layernorm_layer.LayerNorm(setting.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, src_seq_len, src_embed_vec_len)
        mask shape: (batch_size, 1, src_seq_len)
        return shape: (batch_size, src_seq_len, src_embed_vec_len)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    layers: List[decoder_layer.DecoderLayer]

    def __init__(self, encoder_d_model: int, setting: common.CodexSetting):
        super(Decoder, self).__init__()
        self.layers = []
        for _ in range(setting.repeat_count):
            self.layers.append(decoder_layer.DecoderLayer(
                src_embed_vec_len=encoder_d_model,
                tgt_embed_vec_len=setting.d_model,
                ffn_hidden_size=setting.ffn_hidden_size,
                num_heads=setting.num_heads,
                dropout=setting.dropout))
        self.norm = layernorm_layer.LayerNorm(setting.d_model)

    def forward(self, y: torch.Tensor, mask: torch.Tensor, context: torch.Tensor,
                src_mask: torch.Tensor) -> torch.Tensor:
        """
        y shape: (batch_size, tgt_seq_len, tgt_embed_vec_len)
        mask shape: (batch_size, tgt_seq_len, tgt_seq_len)
        context shape: (batch_size, src_seq_len, src_embed_vec_len)
        src_mask shape: (batch_size, 1, src_seq_len)
        return shape: (batch_size, tgt_seq_len, tgt_embed_vec_len)
        """
        for layer in self.layers:
            y = layer(y, mask, context, src_mask)
        return self.norm(y)


class EncoderDecoder(nn.Module):
    encoder: Encoder
    src_embed: embedding_layer.Embedding
    src_pe: pos_encoding_layer.PositionalEncoding

    decoder: Decoder
    tgt_embed: embedding_layer.Embedding
    tgt_pe: pos_encoding_layer.PositionalEncoding

    proj: nn.Linear

    def __init__(self,
                 src_vocab_sz: int,
                 tgt_vocab_sz: int,
                 enc_setting: common.CodexSetting = None,
                 dec_setting: common.CodexSetting = None):
        super(EncoderDecoder, self).__init__()
        if enc_setting is None:
            enc_setting = common.CodexSetting.default_setting()
        if dec_setting is None:
            dec_setting = common.CodexSetting.default_setting()

        self.encoder = Encoder(enc_setting)
        self.src_pe = pos_encoding_layer.PositionalEncoding(enc_setting.d_model)
        self.src_embed = embedding_layer.Embedding(src_vocab_sz, enc_setting.d_model)

        self.decoder = Decoder(encoder_d_model=enc_setting.d_model, setting=dec_setting)
        self.tgt_pe = pos_encoding_layer.PositionalEncoding(dec_setting.d_model)
        self.tgt_embed = embedding_layer.Embedding(tgt_vocab_sz, dec_setting.d_model)

        self.proj = nn.Linear(dec_setting.d_model, tgt_vocab_sz)

    def forward(self, tgt: torch.Tensor, tgt_mask: torch.Tensor, src: torch.Tensor,
                src_mask: torch.Tensor, ) -> torch.Tensor:
        """
        tgt shape: (batch_size, tgt_seq_len)
        tgt_mask shape: (batch_size, tgt_seq_len, tgt_seq_len)
        src shape: (batch_size, src_seq_len)
        src_mask shape: (batch_size, 1, src_seq_len)
        return shape: (batch_size, tgt_seq_len, tgt_vocab_sz)
        """
        context = self.encode(src, src_mask)
        return self.decode(tgt, tgt_mask, context, src_mask)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """
        src shape: (batch_size, src_seq_len)
        src_mask shape: (batch_size, 1, src_seq_len)
        return shape: (batch_size, src_seq_len, src_embed_vec_len)
        """
        X = self.src_embed(src)
        X = self.src_pe(X)
        context = self.encoder(X, src_mask)
        return context

    def decode(self, tgt: torch.Tensor, tgt_mask: torch.Tensor, context: torch.Tensor,
               src_mask: torch.Tensor, ) -> torch.Tensor:
        """
        tgt shape: (batch_size, tgt_seq_len)
        tgt_mask shape: (batch_size, tgt_seq_len, tgt_seq_len)
        context shape: (batch_size, src_seq_len, src_embed_vec_len)
        src_mask shape: (batch_size, 1, src_seq_len)
        return shape: (batch_size, tgt_seq_len, tgt_vocab_sz)

        """
        Y = self.tgt_embed(tgt)
        Y = self.tgt_pe(Y)
        state = self.decoder(Y, tgt_mask, context, src_mask)
        return self.proj(state)
