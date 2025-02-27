from torch import nn
import os

class Aggregate(nn.Module):
    def __init__(self, pth, data, lang_1, lang_2):
        super(Aggregate, self).__init__()
        self.pth = pth
        self.data = data
        self.lang_1 = lang_1
        self.lang_2 = lang_2

    def _s(self, data):
        train = data["train"]
        val = data["validation"]
        test = data["test"]
        return train, val, test

    def _filter_fn(self, a, b):
        return lambda x: ((x["text_1_lang"] == a and x["text_2_lang"] == b) or 
                          (x["text_1_lang"] == b and x["text_2_lang"] == a))

    def forward(self):
        train, val, test = self._s(self.data)
        filter_fn = self._filter_fn(self.lang_1, self.lang_2)
        train_filtered = train.filter(filter_fn)
        val_filtered   = val.filter(filter_fn)
        test_filtered  = test.filter(filter_fn)
        base_path = os.path.join(self.pth, f"{self.lang_1}_{self.lang_2}")
        os.makedirs(base_path, exist_ok=True)
        train_filtered.to_pandas().to_csv(os.path.join(base_path, "train.csv"), index=False)
        val_filtered.to_pandas().to_csv(os.path.join(base_path, "validation.csv"), index=False)
        test_filtered.to_pandas().to_csv(os.path.join(base_path, "test.csv"), index=False)
        
        return base_path
