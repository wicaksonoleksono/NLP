import os
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
import pickle

def whitespace_tokenizer(text: str):
    return text.split()
class Tokenize:
    def __init__(self, path, src_lang="eng", tgt_lang="min"):
        self.path = path
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_tokenizer = whitespace_tokenizer
        self.tgt_tokenizer = whitespace_tokenizer
        self.special_tokens = ["<sos>", "<eos>", "<unk>", "<pad>"]
        self.src_vocab = None
        self.tgt_vocab = None
        self.tp = os.path.join(self.path, f"{self.src_lang}_{self.tgt_lang}")

    def _read(self):
        train = pd.read_csv(os.path.join(self.tp, "train.csv"))
        val   = pd.read_csv(os.path.join(self.tp, "validation.csv"))
        test  = pd.read_csv(os.path.join(self.tp, "test.csv"))
        return train, val, test

    def yield_tokens(self, data, col, is_source=True):
        tokenizer = self.src_tokenizer if is_source else self.tgt_tokenizer
        for text in data[col]:
            yield tokenizer(str(text))

    def build_vocab(self):
        # Build from train set only
        train, _, _ = self._read()
        self.src_vocab = build_vocab_from_iterator(
            self.yield_tokens(train, self.src_lang, is_source=True),
            specials=self.special_tokens
        )
        self.tgt_vocab = build_vocab_from_iterator(
            self.yield_tokens(train, self.tgt_lang, is_source=False),
            specials=self.special_tokens
        )
        # Set <unk> as default index
        self.src_vocab.set_default_index(self.src_vocab["<unk>"])
        self.tgt_vocab.set_default_index(self.tgt_vocab["<unk>"])

    def tokenize_text(self, text, is_source=True):
        tokenizer = self.src_tokenizer if is_source else self.tgt_tokenizer
        tokens = tokenizer(str(text))
        return ["<sos>"] + tokens + ["<eos>"]

    def numericalize(self, text, is_source=True):
        vocab = self.src_vocab if is_source else self.tgt_vocab
        tokens = self.tokenize_text(text, is_source=is_source)
        return [vocab[token] for token in tokens]

    def detokenize(self, indices, is_source=True):
        vocab = self.src_vocab if is_source else self.tgt_vocab
        itos = vocab.get_itos()  # index -> token
        tokens = []
        for idx in indices:
            if idx < len(itos):
                tokens.append(itos[idx])
            else:
                tokens.append("<unk>")  # out-of-range safety
        if tokens and tokens[0] == "<sos>":
            tokens = tokens[1:]
        if tokens and tokens[-1] == "<eos>":
            tokens = tokens[:-1]
        return " ".join(tokens)

    def save_vocab(self, filename_src="src_vocab.pkl", filename_tgt="tgt_vocab.pkl"):
        if self.src_vocab and self.tgt_vocab:
            with open(os.path.join(self.tp, filename_src), "wb") as f:
                pickle.dump(self.src_vocab, f)
            with open(os.path.join(self.tp, filename_tgt), "wb") as f:
                pickle.dump(self.tgt_vocab, f)

    def load_vocab(self, filename_src="src_vocab.pkl", filename_tgt="tgt_vocab.pkl"):
        with open(os.path.join(self.tp, filename_src), "rb") as f:
            self.src_vocab = pickle.load(f)
        with open(os.path.join(self.tp, filename_tgt), "rb") as f:
            self.tgt_vocab = pickle.load(f)

    def preprocess_and_save(self, out_file="preprocessed_data.pkl"):
        if self.src_vocab is None or self.tgt_vocab is None:
            self.build_vocab()
        train_df, val_df, test_df = self._read()
        train_pairs = []
        for _, row in train_df.iterrows():
            src_indices = self.numericalize(row[self.src_lang], is_source=True)
            tgt_indices = self.numericalize(row[self.tgt_lang], is_source=False)
            train_pairs.append((src_indices, tgt_indices))
        val_pairs = []
        for _, row in val_df.iterrows():
            src_indices = self.numericalize(row[self.src_lang], is_source=True)
            tgt_indices = self.numericalize(row[self.tgt_lang], is_source=False)
            val_pairs.append((src_indices, tgt_indices))
        test_pairs = []
        for _, row in test_df.iterrows():
            src_indices = self.numericalize(row[self.src_lang], is_source=True)
            tgt_indices = self.numericalize(row[self.tgt_lang], is_source=False)
            test_pairs.append((src_indices, tgt_indices))
        data_dict = {
            "train": train_pairs,
            "validation": val_pairs,
            "test": test_pairs,
            "src_vocab": self.src_vocab,
            "tgt_vocab": self.tgt_vocab
        }
        with open(os.path.join(self.tp, out_file), "wb") as f:
            pickle.dump(data_dict, f)
        print(f"[INFO] Preprocessed data saved to {os.path.join(self.tp, out_file)}")
