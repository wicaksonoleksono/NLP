import os
import random
import pandas as pd
from torch import nn

class Aggregate(nn.Module):
    def __init__(self, pth, data):
        """
        Args:
            pth  (str): Base folder path where we save CSVs
            data (DatasetDict or dict-like): 
                { "train": Dataset, "validation": Dataset, "test": Dataset }
        """
        super(Aggregate, self).__init__()
        self.pth = pth
        self.data = data

    def _filter_fn(self, a, b):
        """
        Only keep rows where (text_1_lang, text_2_lang) matches (a, b) in either order.
        """
        return lambda x: (
            (x["text_1_lang"] == a and x["text_2_lang"] == b) or
            (x["text_1_lang"] == b and x["text_2_lang"] == a)
        )

    def _swap_if_needed(self, df, src_lang, tgt_lang):
        """
        If text_1_lang != src_lang, swap columns so text_1 is always src, text_2 always tgt.
        """
        df = df.copy()
        mask = df["text_1_lang"] != src_lang
        if mask.any():
            # Swap the text columns
            df.loc[mask, ["text_1", "text_2"]] = df.loc[mask, ["text_2", "text_1"]].values
            # Swap the lang columns
            df.loc[mask, ["text_1_lang", "text_2_lang"]] = df.loc[mask, ["text_2_lang", "text_1_lang"]].values
        return df

    def _rename_cols(self, df):
        """
        Rename "text_1" -> <actual src lang>, "text_2" -> <actual tgt lang>.
        Drop "text_1_lang" and "text_2_lang".
        """
        df = df.copy()
        if not df.empty:
            # Get the first row to see what the final source & target language columns are
            src_col_name = df.loc[df.index[0], "text_1_lang"]
            tgt_col_name = df.loc[df.index[0], "text_2_lang"]

            df.rename(columns={"text_1": src_col_name, "text_2": tgt_col_name}, inplace=True)
            df.drop(columns=["text_1_lang", "text_2_lang"], inplace=True)
        return df

    def forward(self, 
                src_lang, 
                tgt_lang, 
                from_split="test", 
                to_split="train", 
                num_to_move=300):
        """
        1) Filters train/val/test for (src_lang, tgt_lang).
        2) Converts to pandas, swaps if needed, then merges some rows from 'from_split' into 'to_split'.
        3) Renames columns and saves CSVs to {pth}/{src_lang}_{tgt_lang}/{train,validation,test}.csv
        """
        # 1) Filter each split
        filter_fn = self._filter_fn(src_lang, tgt_lang)
        split_dfs = {}
        for split_name in ["train", "validation", "test"]:
            if split_name in self.data:
                ds_filtered = self.data[split_name].filter(filter_fn)
                df = self._swap_if_needed(ds_filtered.to_pandas(), src_lang, tgt_lang)
                split_dfs[split_name] = df
            else:
                # If the dataset doesn't have that split (rare case), store an empty DF
                split_dfs[split_name] = None

        # 2) Move some rows from from_split -> to_split
        if split_dfs[from_split] is not None and split_dfs[to_split] is not None:
            from_df = split_dfs[from_split]
            to_df   = split_dfs[to_split]

            # Randomly shuffle and pick 'num_to_move' indices
            all_indices = from_df.index.tolist()
            random.shuffle(all_indices)
            selected_indices = all_indices[:num_to_move]
            # Rows to move
            rows_to_move = from_df.loc[selected_indices]
            # Append them to the target
            to_df = pd.concat([to_df, rows_to_move], ignore_index=True)
            # Remove them from the original
            from_df = from_df.drop(selected_indices)

            # Store back
            split_dfs[from_split] = from_df
            split_dfs[to_split]   = to_df
        else:
            print(f"Warning: Either split '{from_split}' or split '{to_split}' doesn't exist. Skipping move.")
        base_path = os.path.join(self.pth, f"{src_lang}_{tgt_lang}")
        os.makedirs(base_path, exist_ok=True)
        for split_name in ["train", "validation", "test"]:
            df = split_dfs[split_name]
            if df is not None and not df.empty:
                df = self._rename_cols(df)
                csv_name = f"{split_name}.csv"
                df.to_csv(os.path.join(base_path, csv_name), index=False)
            else:
                # Even if empty or None, still create an empty CSV to be consistent
                csv_name = f"{split_name}.csv"
                empty_df = df if df is not None else []
                pd.DataFrame(empty_df).to_csv(os.path.join(base_path, csv_name), index=False)

        return base_path
