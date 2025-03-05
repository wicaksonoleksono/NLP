import torch 
from torch import Tensor
from torch import nn

from .attention import MultiHeadAttention
from .pos_encode import PositionalEncoding
from .feedforward import FeedForwardLayer


# ########################################################################
# # ENCODER LAYER :
# 1. Performs Multiheaded self attention
# 2. Residual + Layer norm
# 3. Performs FFN
# 4. Residual + layer norm
# ########################################################################

class TransformerEncoderLayer(nn.Module):
	def __init__(self,
	             dim_model ,
	             num_heads ,
	             dim_feedforward ,
	             dropout ,
	             device,
	             ):
		super().__init__()

		self.attention = MultiHeadAttention(dim_model, num_heads, dropout, device)
		self.norm1 = nn.LayerNorm(dim_model)

		self.feed_forward = FeedForwardLayer(dim_model, dim_feedforward, dropout)
		self.norm2 = nn.LayerNorm(dim_model)

		self.dropout = nn.Dropout(dropout)

	def forward(self, src, mask):
		# self attention
		attention, _ = self.attention(src, src, src, mask)

		attention_norm = self.norm1(src + self.dropout(attention))
		forward = self.feed_forward(attention_norm)

		output = self.norm2(attention_norm + self.dropout(forward))
		return output

# ########################################################################
# # ENCODER BLOCK :
# 1. Performs word+pos embedding
# 2. Runs Encoder layers sequentially
# ########################################################################
class TransformerEncoder(nn.Module):
	def __init__(self,
	             src_vocab_size,
	             dim_model,
	             num_layers,
	             num_heads ,
	             dim_feedforward ,
	             dropout ,
	             device ,
	             MAX_LENGTH=100):
		super().__init__()
		self.device = device
		self.dim_model = dim_model

		self.word_embedding = nn.Embedding(src_vocab_size, dim_model)
		self.coefficient = torch.sqrt(torch.FloatTensor([self.dim_model])).to(device)
		self.position_encoding = PositionalEncoding()

		self.dropout = nn.Dropout(dropout)

		encoding_layers = []
		for _ in range(num_layers):
			encoding_layers.append(TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout, device))
		self.layers = nn.Sequential(*encoding_layers)

	def forward(self, src, mask):
		input_size = src.shape[1]

		src = self.dropout((self.word_embedding(src) * self.coefficient)+ self.position_encoding(input_size, self.dim_model, self.device))

		for layer in self.layers:
			src = layer(src, mask)
		return src