from torch import nn
import torch 
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, enc_hidden_size, dec_hidden_size):
        """
        enc_hidden_size: hidden size used in the encoder (before doubling due to bidirectionality)
        dec_hidden_size: hidden size used in the decoder
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        # The encoder's hidden state is (num_layers, batch, 2 * enc_hidden_size)
        # If 2*enc_hidden_size doesn't match dec_hidden_size, create a bridge to transform it.
        if (enc_hidden_size * 2) != dec_hidden_size:
            self.bridge = nn.Linear(enc_hidden_size * 2, dec_hidden_size)
        else:
            self.bridge = None

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        src: (batch, src_len)
        tgt: (batch, tgt_len)
        teacher_forcing_ratio: probability to use ground truth token as the next input
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        output_vocab_size = self.decoder.fc_out.out_features

        # Tensor to hold predictions
        outputs = torch.zeros(batch_size, tgt_len, output_vocab_size).to(self.device)
        # Encode the source sequence
        encoder_outputs, hidden_enc = self.encoder(src)
        # hidden_enc shape: (num_layers, batch, 2 * hidden_size)
        dec_hidden = hidden_enc
        # If needed, use the bridge to convert the encoder's hidden state dimension
        if self.bridge is not None:
            dec_hidden = torch.tanh(self.bridge(dec_hidden))
        # The first token is assumed to be the <sos> (start of sequence) token
        input_token = tgt[:, 0].unsqueeze(1)  # (batch, 1)
        # Decode token by token
        for t in range(1, tgt_len):
            output, dec_hidden = self.decoder(input_token, dec_hidden)
            outputs[:, t, :] = output.squeeze(1)
            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            # Get the highest predicted token from our predictions
            top1 = output.argmax(2)
            input_token = tgt[:, t].unsqueeze(1) if teacher_force else top1
        return outputs,dec_hidden