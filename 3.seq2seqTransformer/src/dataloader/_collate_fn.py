import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    src_batch = [torch.tensor(item['src'], dtype=torch.long) for item in batch]
    tgt_batch = [torch.tensor(item['tgt'], dtype=torch.long) for item in batch]
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return {'src': src_batch, 'tgt': tgt_batch}
