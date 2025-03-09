import os
import re
import math
import unicodedata
import numpy as np
import pandas as pd
from collections import Counter
import sentencepiece as spm
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------
# SPECIAL TOKENS
# ------------------------------------------------------------------------
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
MAX_SENT_LEN = 107


# ------------------------------------------------------------------------
# Reading Data
# ------------------------------------------------------------------------
def get_max_lengths(df, name):
    max_min = df["min"].astype(str).str.split().str.len().max()
    max_eng = df["eng"].astype(str).str.split().str.len().max()
    print(f"{name} - Max 'min' sentence length: {max_min}")
    print(f"{name} - Max 'eng' sentence length: {max_eng}")


def _read(folder_path):
    train = pd.read_csv(os.path.join(folder_path, "train.csv"))
    val   = pd.read_csv(os.path.join(folder_path, "validation.csv"))
    test  = pd.read_csv(os.path.join(folder_path, "test.csv"))
    get_max_lengths(train,"TrainData")
    get_max_lengths(val,"TestData")
    get_max_lengths(test,"ValidData")
    return train, val, test

def get_data(base_path, src_lang, tgt_lang, max_sent_len=MAX_SENT_LEN):
    folder_path = os.path.join(base_path, f"{src_lang}_{tgt_lang}")
    train, val, test = _read(folder_path)
    return train, val, test, max_sent_len

# ------------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------------
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != 'Mn'
    )

# def normalizeString(s):
#     s = unicodeToAscii(s.lower().strip())
#     s = re.sub(r"[^a-zA-Z0-9\s.,!?]+", "", s)
#     s = re.sub(r"(\w)\s+([.,!?])", r"\1\2", s)
#     s = re.sub(r"\s+", " ", s).strip()
#     return s
# def normalizeString(s):
#     s = unicodeToAscii(s.lower().strip())
#     s = re.sub(r"[^a-zA-Z ]+", " ", s)  # Remove ALL punctuation
#     return s

def normalizeString(s):
    s = s.lower().strip()
    s = ''.join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != 'Mn'
    )
    s = re.sub(r"[^a-zA-Z'\s]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
# # ------------------------------------------------------------------------
# Preprocessing CSV data
# ------------------------------------------------------------------------
def preprocess_data(dataframe, source_lang, target_lang, max_sent_len, normalize_fn):
    source_sentences_all = dataframe[source_lang].tolist()
    target_sentences_all = dataframe[target_lang].tolist()
    source_normalized = list(map(normalize_fn, source_sentences_all))
    target_normalized = list(map(normalize_fn, target_sentences_all))
    source_sentences = []
    target_sentences = []
    for src, tgt in zip(source_normalized, target_normalized):
        src_tokens = src.split()
        tgt_tokens = tgt.split()
        if len(src_tokens) <= max_sent_len and len(tgt_tokens) <= max_sent_len:
            source_sentences.append(src)
            target_sentences.append(tgt)
    return source_sentences, target_sentences

# ------------------------------------------------------------------------
# Dictionary class
# ------------------------------------------------------------------------
class Dictionary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        # The next 4 indices are reserved for PAD, SOS, EOS, UNK
        self.index2word = {
            PAD_TOKEN: "PAD",
            SOS_TOKEN: "SOS",
            EOS_TOKEN: "EOS",
            UNK_TOKEN: "UNK"
        }
        self.n_count = 4  # so that the first 'real' word is index=4

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_count
            self.word2count[word] = 1
            self.index2word[self.n_count] = word
            self.n_count += 1
        else:
            self.word2count[word] += 1

# ------------------------------------------------------------------------
# Tokenize & Detokenize
# ------------------------------------------------------------------------
def tokenize(sentence, dictionary, max_length=MAX_SENT_LEN):
    split_sentence = sentence.split()
    token = [SOS_TOKEN]
    token += [dictionary.word2index.get(word, UNK_TOKEN) for word in split_sentence]
    token.append(EOS_TOKEN)
    num_pad = max_length - len(split_sentence)
    if num_pad > 0:
        token += [PAD_TOKEN] * num_pad
    return token

def detokenize(token_ids, vocab):
    words = []
    for tid in token_ids:
        w = vocab.index2word[tid]
        if w not in ["EOS", "SOS", "PAD"]:
            words.append(w)
    return " ".join(words)

# ------------------------------------------------------------------------
# BLEU-Score
# ------------------------------------------------------------------------
def bleu_stats(hypothesis, reference):
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )

        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats

def bleu(stats):
    if any(x == 0 for x in stats):
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.0
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)

def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.zeros(10,)
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)



def plot_loss(atl, avl, save_dir):
    """
    Plots training vs. validation loss and saves the figure to save_dir.

    Args:
        atl (dict): Dictionary with epoch numbers as keys and average training loss as values.
        avl (dict): Dictionary with epoch numbers as keys and average validation loss as values.
        save_dir (str): Directory where the plot image will be saved.
    """
    # Ensure the epochs are sorted
    epochs = sorted(atl.keys())
    train_losses = [atl[epoch] for epoch in epochs]
    val_losses = [avl[epoch] for epoch in epochs]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='o', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs. Validation Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot to SAVE_DIR
    plot_path = os.path.join(save_dir, 'loss_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss plot saved to {plot_path}")

# ------------------------------------------------------------------------
# SUBWORD-TOKENIZATION
# ------------------------------------------------------------------------

def create_spm_training_file(data_path, source_lang, target_lang, output_file="spm_train.txt"):
    train_df, _, _, max_sent_len = get_data(data_path, source_lang, target_lang)
    source_sentences, _ = preprocess_data(train_df, source_lang, target_lang, max_sent_len, normalizeString)
    with open(output_file, "w", encoding="utf8") as f:
        for sentence in source_sentences:
            f.write(sentence + "\n")
    print(f"SPM training file created at {output_file}")
    return output_file
def train_sentencepiece_model(training_file, model_prefix="spm_model", vocab_size=5000, model_type="bpe"):
    spm_args = f"--input={training_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type={model_type}"
    print("Training SentencePiece model with arguments:", spm_args)
    spm.SentencePieceTrainer.Train(spm_args)
    print(f"SentencePiece model trained and saved with prefix '{model_prefix}'.")

# if not os.path.exists("spm_model.model"):
#     training_file = create_spm_training_file(data_path, source_lang, target_lang, output_file="spm_train.txt")
#     train_sentencepiece_model(training_file, model_prefix="spm_model", vocab_size=5000, model_type="bpe")

# # Load the trained SentencePiece model.
# sp = spm.SentencePieceProcessor()
# sp.Load("spm_model.model")
# print("SentencePiece model loaded.")

def sp_tokenize(sentence):
    """
    Tokenize a sentence into subword token IDs using SentencePiece.
    Returns a list of token IDs.
    """
    return sp.EncodeAsIds(sentence)

def sp_detokenize(token_ids):

    return sp.DecodeIds(token_ids)

def sp_tokenize_with_specials(sentence, max_length=MAX_SENT_LEN):

    token_ids = sp_tokenize(sentence)
    token_ids = [SOS_TOKEN] + token_ids + [EOS_TOKEN]
    if len(token_ids) < max_length:
        token_ids += [PAD_TOKEN] * (max_length - len(token_ids))
    else:
        token_ids = token_ids[:max_length]
    return token_ids

def sp_detokenize_with_specials(token_ids):

    filtered_ids = [tid for tid in token_ids if tid not in (SOS_TOKEN, EOS_TOKEN, PAD_TOKEN)]
    return sp_detokenize(filtered_ids)
