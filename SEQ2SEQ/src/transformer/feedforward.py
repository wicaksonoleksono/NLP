
import torch 
from torch import nn
class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size, ff_size, dropout):
        super().__init__()

        self.ff_layer = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.ReLU(),

            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size)
        )

    def forward(self, input):
        output = self.ff_layer(input)
        return output
