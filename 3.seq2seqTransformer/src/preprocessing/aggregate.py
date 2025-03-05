from torch import nn
import os
class Aggregate(nn.Module):
    def __init__(self, pth, data):
        super(Aggregate, self).__init__()
        self.pth = pth
        self.data = data
    def _s(self, data):
        train = data["train"]
        val = data["validation"]
        test = data["test"]
        return train, val, test
    def _filter_fn(self, a, b):
        return lambda x: ((x["text_1_lang"] == a and x["text_2_lang"] == b) or 
                          (x["text_1_lang"] == b and x["text_2_lang"] == a))
    def _swap_if_needed(self, df, src_lang, tgt_lang):
        df = df.copy()
        mask = df["text_1_lang"] != src_lang
        if mask.any():
            df.loc[mask, ["text_1", "text_2"]] = df.loc[mask, ["text_2", "text_1"]].values
            df.loc[mask, ["text_1_lang", "text_2_lang"]] = df.loc[mask, ["text_2_lang", "text_1_lang"]].values
        return df

    def _rename_cols(self, df):
        df = df.copy()
        if not df.empty:
            src_col_name = df.loc[df.index[0], "text_1_lang"]
            tgt_col_name = df.loc[df.index[0], "text_2_lang"]
            
            df.rename(columns={"text_1": src_col_name, "text_2": tgt_col_name}, inplace=True)
            
            df.drop(columns=["text_1_lang", "text_2_lang"], inplace=True)
        return df

    def forward(self, src_lang, tgt_lang):
        train, val, test = self._s(self.data)
        
        filter_fn = self._filter_fn(src_lang, tgt_lang)
        
        train_filtered = train.filter(filter_fn)
        val_filtered   = val.filter(filter_fn)
        test_filtered  = test.filter(filter_fn)

        train_df = self._swap_if_needed(train_filtered.to_pandas(), src_lang, tgt_lang)
        val_df   = self._swap_if_needed(val_filtered.to_pandas(), src_lang, tgt_lang)
        test_df  = self._swap_if_needed(test_filtered.to_pandas(), src_lang, tgt_lang)
        train_df = self._rename_cols(train_df)
        val_df   = self._rename_cols(val_df)
        test_df  = self._rename_cols(test_df)
        base_path = os.path.join(self.pth, f"{src_lang}_{tgt_lang}")
        os.makedirs(base_path, exist_ok=True)
        train_df.to_csv(os.path.join(base_path, "train.csv"), index=False)
        val_df.to_csv(os.path.join(base_path, "validation.csv"), index=False)
        test_df.to_csv(os.path.join(base_path, "test.csv"), index=False)
        
        return base_path
