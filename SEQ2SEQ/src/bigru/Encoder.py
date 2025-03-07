from torch import nn 
import torch 
from ._bigru import BiGRULayer
# nn.gru
class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=input_vocab_size,
            embedding_dim=embed_size,
            padding_idx=pad_idx  # Ensure the pad token embedding remains fixed
        )
        self.dropout = nn.Dropout(dropout)
        self.bigru = BiGRULayer(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, src):
        # src shape: (batch, src_len)
        embedded = self.dropout(self.embedding(src))  # (batch, src_len, embed_size)
        outputs, hidden = self.bigru(embedded)
        return outputs, hidden  # outputs: (batch, src_len, 2*hidden_size)