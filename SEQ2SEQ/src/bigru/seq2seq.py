from torch import nn
import torch 
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device='cpu'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        encoder_outputs, hidden_enc = self.encoder(src)
        hidden_dec = nn.functional.relu(
            nn.Linear(hidden_enc.size(2), self.decoder.hidden_size, bias=False)(hidden_enc[0])
        )
        hidden_dec = hidden_dec.unsqueeze(0)  # shape: (1, batch, hidden_size)
        logits_list = []

        # Let's say the first input to decoder is tgt[:,0] (like <sos> token)
        dec_input = tgt[:, 0].unsqueeze(1)  # shape: (batch, 1)
        for t in range(1, tgt_len):
            # Pass one token at a time
            output, hidden_dec = self.decoder(dec_input, hidden_dec)
            # output shape: (batch, 1, tgt_vocab_size)
            logits_list.append(output)
            # Determine next input to the decoder
            if torch.rand(1).item() < teacher_forcing_ratio:
                # Use teacher forcing: feed the ground truth
                dec_input = tgt[:, t].unsqueeze(1)
            else:
                # Use the predicted token
                pred_token = output.argmax(dim=2)  # (batch, 1)
                dec_input = pred_token
        logits = torch.cat(logits_list, dim=1)  # (batch, tgt_len-1, vocab_size)
        return logits
