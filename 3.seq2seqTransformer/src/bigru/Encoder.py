from torch import nn 
import torch 
from ._bigru import BiGRULayer
# nn.gru
class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        hidden_size,
        num_layers=1,
        dropout=0.1,
        batch_first=True
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # Embedding layer for source language
        self.embedding = nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=embed_size)
        
        # Dropout to apply after embeddings or between encoder layers
        self.dropout = nn.Dropout(dropout)
        
        # We stack multiple BiGRU layers:
        #   - The input to the first layer is `embed_size`
        #   - The input to subsequent layers is `2*hidden_size` because we get
        #     forward+backward hidden states concatenated from the previous layer
        self.bigru_layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_size = embed_size if i == 0 else 2 * hidden_size
            self.bigru_layers.append(
                BiGRULayer(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=batch_first,
                    dropout=0.0  # Using manual dropout instead to show control
                )
            )

    def forward(self, src):
        """
        src shape: (batch, src_len) if batch_first=True
        returns: outputs, hidden_states
         - outputs: (batch, src_len, 2*hidden_size)
         - hidden_states: (num_layers, batch, 2*hidden_size) from the top BiGRULayer
        """
        # 1) Embed the source tokens
        embedded = self.embedding(src)  # (batch, src_len, embed_size)
        
        # 2) Pass through stacked BiGRULayers
        x = self.dropout(embedded)
        hidden_out = None
        
        for layer in self.bigru_layers:
            x, h = layer(x)  # x: (batch, src_len, 2*hidden_size), h: (1, batch, 2*hidden_size)
            x = self.dropout(x)
            hidden_out = h
        
        # hidden_out is from the last layer
        return x, hidden_out

