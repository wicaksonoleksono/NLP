from torch import nn 
import torch 
# Written by @wicaksonolxn 06.03.25

# Simple logic
# Forward Layer:
# [x₁ → x₂ → x₃ → ... → xₙ] → [h₁ᶠ → h₂ᶠ → h₃ᶠ → ... → hₙᶠ]

# Backward Layer:
# [x₁ ← x₂ ← x₃ ← ... ← xₙ] → [h₁ᵇ ← h₂ᵇ ← h₃ᵇ ← ... ← hₙᵇ]
# Final representation:
# h₁ = concat(h₁ᶠ, h₁ᵇ), 
# h₂ = concat(h₂ᶠ, h₂ᵇ), ...,
# hₙ = concat(hₙᶠ, hₙᵇ)

import torch
from torch import nn
import torch.nn.functional as F

class BiGRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        # Forward GRU
        self.gru_f = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=(dropout if num_layers > 1 else 0.0)
        )
        # Backward GRU
        self.gru_b = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=(dropout if num_layers > 1 else 0.0)
        )

    def forward(self, x, h_f=None, h_b=None):
        # Forward pass
        out_f, h_f = self.gru_f(x, h_f)  # (batch, seq_len, hidden_size)
        # Reverse input sequence for backward GRU
        x_rev = torch.flip(x, dims=[1])
        out_b, h_b = self.gru_b(x_rev, h_b)  # (batch, seq_len, hidden_size)
        # Reverse backward outputs to restore original order
        out_b = torch.flip(out_b, dims=[1])
        # Concatenate forward and backward outputs along the feature dimension
        out = torch.cat([out_f, out_b], dim=2)  # (batch, seq_len, 2 * hidden_size)
        # Concatenate the final hidden states (if using the top layer only)
        h = torch.cat([h_f, h_b], dim=2)  # (num_layers, batch, 2 * hidden_size)
        return out, h

