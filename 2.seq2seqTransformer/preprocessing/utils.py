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