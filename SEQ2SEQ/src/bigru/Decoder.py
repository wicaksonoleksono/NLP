from torch import nn 
import torch 
from ._bigru import BiGRULayer
class Decoder(nn.Module):
    def __init__(
        self,
        tgt_vocab_size,
        embed_size,
        hidden_size,
        num_layers=1,
        dropout=0.1,
        batch_first=True
    ):
        super().__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        self.embedding = nn.Embedding(num_embeddings=tgt_vocab_size, embedding_dim=embed_size)
        self.dropout = nn.Dropout(dropout)
        
        # Note: we use a unidirectional GRU here
        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )
        
        # Final output layer -> predict target vocab
        self.fc_out = nn.Linear(hidden_size, tgt_vocab_size)
        
    def forward(self, tgt_in, hidden):
        """
        tgt_in: (batch, tgt_len) if batch_first=True
        hidden: initial hidden state (num_layers, batch, hidden_size)
        
        Returns: 
          - logits: (batch, tgt_len, tgt_vocab_size)
          - hidden: updated hidden state (num_layers, batch, hidden_size)
        """
        # 1) Embed the input tokens
        embedded = self.embedding(tgt_in)   # (batch, tgt_len, embed_size)
        embedded = self.dropout(embedded)
        
        # 2) Pass embedded tokens + hidden state into GRU
        output, hidden = self.gru(embedded, hidden)  # output: (batch, tgt_len, hidden_size)
        
        # 3) Predict next-token distribution from GRU outputs
        output = self.dropout(output)            # (batch, tgt_len, hidden_size)
        logits = self.fc_out(output)             # (batch, tgt_len, tgt_vocab_size)
        
        return logits, hidden
