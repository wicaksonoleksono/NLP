import pandas  as pd 
import os 
import unicodedata
import re
def _read(tp):
    train = pd.read_csv(os.path.join(tp, "train.csv"))
    val   = pd.read_csv(os.path.join(tp, "validation.csv"))
    test  = pd.read_csv(os.path.join(tp, "test.csv"))
    return train, val, test
def get_data(tp,src_lang,tgt_lang,max_sent_len=50):
    train,val,test= _read(f"{tp}/{src_lang}_{tgt_lang}/")
    return train,val,test,max_sent_len

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize("NFD",s)
        if unicodedata.category(c) != 'mn'
    )
def normalizeString(s):
    s =unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

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


PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
###########################
# DICT PREPROC
###########################
class Dictionary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_TOKEN: "PAD", SOS_TOKEN: "SOS", EOS_TOKEN: "EOS", UNK_TOKEN: "UNK" }
        self.n_count = 4
    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)
    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_count
            self.word2count[word] = 1
            self.index2word[self.n_count] = word
            self.n_count += 1
        else:
            self.word2count[word] += 1



def tokenize(sentence, dictionary, MAX_LENGTH=50):
    split_sentence = sentence.split(' ')
    token = [SOS_TOKEN]
    # Use dictionary.word2index.get() to return UNK_TOKEN if word not found.
    token += [dictionary.word2index.get(word, UNK_TOKEN) for word in split_sentence]
    token.append(EOS_TOKEN)
    token += [PAD_TOKEN] * (MAX_LENGTH - len(split_sentence))
    return token

# ########################################################################
# # Method to detokenize i.e. convert idx to words
# input - List of idx of a sentence, Vocabulary to convert from idx2word
# output - Sentence of words
# It ignores any special tokens (EOS,SOS,PAD)
# ########################################################################
def detokenize(x, vocab):
    words = []
    for i in x:
        word = vocab.index2word[i]
        if word != 'EOS' and word != 'SOS' and word != 'PAD':
            words.append(word)

    words = " ".join(words)
    return words
            
