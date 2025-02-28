from torch import nn 
import torch

class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 embed_dim,
                 hidden_dim,
                 n_layers,
                 pad_idx,
                 dropout = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim= embed_dim,
            padding_idx = pad_idx
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self,tgt,hidden,cell):
        embedded = self.embedding(tgt)
        outputs,(hidden,cell)=self.lstm(embedded,(hidden,cell))
        logits = self.fc_out(outputs)
        return logits,(hidden,cell)
    
    