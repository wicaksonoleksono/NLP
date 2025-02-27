import os
import pandas as pd
import pickle
from transformers import T5Tokenizer

class Tokenize:
    def __init__(
        self,
        path,
        source_lang,
        target_lang,
        tokenizer_name="google/mt5-small",
        max_length=128
    ):
        self.path = path
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def load_data(self):
        train = pd.read_csv(os.path.join(self.path, "train.csv"))
        validation = pd.read_csv(os.path.join(self.path, "validation.csv"))
        test = pd.read_csv(os.path.join(self.path, "test.csv"))
        return {"train": train, "validation": validation, "test": test}

    def _prepare_text_pairs(self, df):
        source_texts = []
        target_texts = []

        for idx, row in df.iterrows():
            if row["text_1_lang"] == self.source_lang:
                source_texts.append(row["text_1"])
                target_texts.append(row["text_2"])
            else:
                source_texts.append(row["text_2"])
                target_texts.append(row["text_1"])
        return source_texts, target_texts
    def _batch_tokenize(self, source_texts, target_texts):
        prompts = [
            f"translate {self.source_lang} to {self.target_lang}: {txt}"
            for txt in source_texts
        ]

        # 1) Tokenize source (prompts)
        model_inputs = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )

        # 2) Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True
            )
        
        # 3) Replace pad_token_id with -100 in the labels
        pad_token_id = self.tokenizer.pad_token_id
        new_labels = []
        for seq in labels["input_ids"]:
            seq = [(-100 if token_id == pad_token_id else token_id) for token_id in seq]
            new_labels.append(seq)

        model_inputs["labels"] = new_labels

        # 4) Build the final list of tokenized examples
        tokenized_examples = []
        for i in range(len(source_texts)):
            tokenized_examples.append({
                "input_ids": model_inputs["input_ids"][i],
                "attention_mask": model_inputs["attention_mask"][i],
                "labels": model_inputs["labels"][i],
            })

        return tokenized_examples



    def process_and_save(self, output_filename="tokenized_dataset.pkl"):
        data_splits = self.load_data()
        tokenized_data = {}
        for split_name, df in data_splits.items():
            print(f"Tokenizing '{split_name}' split with {len(df)} examples...")
            source_texts, target_texts = self._prepare_text_pairs(df)
            tokenized_dataset = self._batch_tokenize(source_texts, target_texts)
            tokenized_data[split_name] = tokenized_dataset
            print(f"  -> {len(tokenized_dataset)} examples tokenized.")
        output_path = os.path.join(self.path, output_filename)
        with open(output_path, "wb") as f:
            pickle.dump(tokenized_data, f)
        print(f"Tokenized dataset saved to: {output_path}")
        return tokenized_data
