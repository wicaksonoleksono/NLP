
from torch import nn 
import torch
class coupler(nn.Module):
    def __init__(self,encoder,decoder,device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self,src,tgt):
        _, (hidden, cell) = self.encoder(src)
        outputs, _ = self.decoder(tgt, hidden, cell)
        return outputs
