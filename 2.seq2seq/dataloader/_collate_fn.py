from torch.nn.utils.rnn import pad_sequence
import torch
def collate_fn(batch):

    src_list = []
    tgt_list = []

    for (src, tgt) in batch:
        src_list.append(torch.tensor(src, dtype=torch.long))
        tgt_list.append(torch.tensor(tgt, dtype=torch.long))
    pad_idx = 3  
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded
