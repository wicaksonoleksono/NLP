import torch 
from torch import Tensor
from torch import nn

from .attention import MultiHeadAttention
from .pos_encode import PositionalEncoding
from .feedforward import FeedForwardLayer

# ########################################################################
# # DECODER LAYER :
# 1. Performs Multiheaded self attention on Q,K,V from Target
# 2. Residual + Layer norm
# 3. Performs Multiheaded Enc-Dec attention on Q from target and K,V from Encoder
# 4. Residual + Layer norm
# 5. FFN
# 6. Residual + Layer norm
# ########################################################################
class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 dim_model,
                 num_heads,
                 dim_feedforward,
                 dropout,
                 device):
        super().__init__()
        # initializing 
        self.attn_1 = MultiHeadAttention(dim_model,num_heads,dropout,device)
        self.norm_1=nn.LayerNorm(dim_model)
        
        self.attn_2 = MultiHeadAttention(dim_model,num_heads,dropout,device)
        self.norm_2=nn.LayerNorm(dim_model)
        
        self.feed_forward = FeedForwardLayer(dim_model,dim_feedforward,dropout)
        self.norm_3 = nn.LayerNorm(dim_model)

        self.dropout = nn.Dropout(dropout)
    def forward(self,target,encoded_input,target_mask,input_mask):
        
        # self attn 
        attention_1,_ = self.attn_1(target,target,target,target_mask) # Query, key, value dan mask utk input valid
        attention_1_norm = self.norm_1(target+self.dropout(attention_1)) # norm ini mksudnya q/(kv)^{1/2}

        # Encoder-decoder self attn 

        attention_2,attention = self.attn_2(attention_1_norm,encoded_input,encoded_input,input_mask) # Input Norm baru, 
        attention_2_norm = self.norm_2(target+self.dropout(attention_2))

        # Forward FFN
        forward = self.feed_forward(attention_2_norm)

        # Norm forward 
        output = self.norm_3(attention_2_norm+self.dropout(forward))

        return output,attention
    

        
##################################################
# decoder block : 
# 1. word+pos embedding
# 2. running decoder layers sequentialy 
# 3. performs ffn
##################################################

    
class TransformerDecoder(nn.Module):
    def __init__(self,
                 tgt_vocab_size,
                 dim_model,
                 num_layers,
                 num_heads,
                 dim_feedforward,
                 dropout,
                 device,
                 MAX_LENGTH=100):
        super().__init__()
        self.device = device
        self.dim_model = dim_model

        self.word_embedding = nn.Embedding(tgt_vocab_size,dim_model) # que

        self.coefficient = torch.sqrt(torch.FloatTensor([self.dim_model])).to(device)
        self.positional_encoding=PositionalEncoding()

        self.dropout = nn.Dropout(dropout)

        decoding_layers=[]
        for _ in range(num_layers):
            decoding_layers.append(TransformerDecoderLayer(dim_model,num_heads,dim_feedforward,dropout,device))
        self.layers = nn.Sequential(*decoding_layers)
        self.linear=nn.Linear(dim_model,tgt_vocab_size)
    def forward(self,target,encoded_input,target_mask,input_mask):
        target_size=target.shape[1]
        target =self.dropout((self.word_embedding(target)*self.coefficient)+self.positional_encoding(target_size,self.dim_model,self.device))

        for layer in self.layers:
            target,attention = layer(target,encoded_input,target_mask,input_mask)

            output = self.linear(target)
            return output,attention
        
