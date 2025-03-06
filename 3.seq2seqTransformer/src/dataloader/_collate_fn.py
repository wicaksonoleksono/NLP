
import torch
from torch.nn.utils.rnn import pad_sequence
import utils 

def collate_fn(batch):
    src_batch = [
        item['src'].clone().detach().to(torch.long)
        if isinstance(item['src'], torch.Tensor)
        else torch.tensor(item['src'], dtype=torch.long)
        for item in batch
    ]
    tgt_batch = [
        item['tgt'].clone().detach().to(torch.long)
        if isinstance(item['tgt'], torch.Tensor)
        else torch.tensor(item['tgt'], dtype=torch.long)
        for item in batch
    ]
    
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True,padding_value=utils.PAD_TOKEN)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True,padding_value=utils.PAD_TOKEN)
    
    return {'src': src_batch, 'tgt': tgt_batch}

