from torch import nn 
import torch 
from ._bigru import BiGRULayer
class Decoder(nn.Module):
    def __init__(self, output_vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=output_vocab_size,
            embedding_dim=embed_size,
            padding_idx=pad_idx
        )
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0)
        )
        self.fc_out = nn.Linear(hidden_size, output_vocab_size)
        self.hidden_size = hidden_size

    def forward(self, input_token, hidden):
        # input_token shape: (batch, 1)
        embedded = self.dropout(self.embedding(input_token))  # (batch, 1, embed_size)
        output, hidden = self.gru(embedded, hidden)  # output: (batch, 1, hidden_size)
        prediction = self.fc_out(output)  # (batch, 1, output_vocab_size)
        return prediction, hidden