# model.py
import torch
from torch import nn
from ._encoder import Encoder
from ._decoder import Decoder
from ._coupler import coupler
class Seq2SeqModel(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 embed_dim,
                 hidden_dim,
                 n_layers,
                 pad_idx,
                 dropout=0.0,
                 device="cpu"):
        super().__init__()
        self.encoder = Encoder(
            input_dim=src_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            pad_idx=pad_idx,
            dropout=dropout
        )
        self.decoder = Decoder(
            output_dim=tgt_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            pad_idx=pad_idx,
            dropout=dropout
        )
        self.seq2seq = coupler(self.encoder, self.decoder, device=device)
    def forward(self, src, tgt):
        return self.seq2seq(src, tgt)
