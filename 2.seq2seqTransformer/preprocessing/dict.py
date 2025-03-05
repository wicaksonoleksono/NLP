
import pickle
from . import utils

PAD_TOKEN = 0 
SOS_TOKEN = 1
EOS_TOKEN = 2

###########################
# DICT PREPROC
##########################
class Dictionary:
    def __init__(self,name):
        self.name=name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_TOKEN:"PAD",SOS_TOKEN:"SOS",EOS_TOKEN:"EOS"}
        self.n_count=3
    def add_sentence(self,sentence):
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

# ########################################################################
# # Preprocess sentences and create dictionaries for source and target
########################################################################
def create_dictionary(source_lang, target_lang,data_path):
    train, val, test, max_sent_len = utils.get_data(data_path, source_lang, target_lang)
    source_sentences_all = train[source_lang].tolist()
    target_sentences_all = train[target_lang].tolist()
    source_normalized = list(map(utils.normalizeString, source_sentences_all))
    target_normalized = list(map(utils.normalizeString, target_sentences_all))
    source_sentences = []
    target_sentences = []
    for src, tgt in zip(source_normalized, target_normalized):
        src_tokens = src.split()
        tgt_tokens = tgt.split()
        if len(src_tokens) <= max_sent_len and len(tgt_tokens) <= max_sent_len:
            source_sentences.append(src)
            target_sentences.append(tgt)
    input_dic = Dictionary(source_lang)
    output_dic = Dictionary(target_lang)
    for sentence in source_sentences:
        input_dic.add_sentence(sentence)
    for sentence in target_sentences:
        output_dic.add_sentence(sentence)
    save_dictionary(data_path,input_dic,source_lang,target_lang, input=True)
    save_dictionary(data_path,output_dic,source_lang,target_lang, input=False)
    return input_dic, output_dic, source_sentences, target_sentences
def save_dictionary(tp,dictionary,src,tgt, input=True):
    if input is True:
        with open(f'{tp}/{src}_{tgt}/input_dic.pkl', 'wb') as f:
            pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(f'{tp}/{src}_{tgt}/output_dic.pkl', 'wb') as f:
            pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)

