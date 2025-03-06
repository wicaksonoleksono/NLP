
import torch
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, tokenized_data):
        self.src_data = tokenized_data['src']
        self.tgt_data = tokenized_data['tgt']
        self.num_examples = len(self.src_data)
    def __len__(self):
        return self.num_examples
    def __getitem__(self, item):
        return {'src': torch.tensor(self.src_data[item], dtype=torch.long),
                'tgt': torch.tensor(self.tgt_data[item], dtype=torch.long)}
