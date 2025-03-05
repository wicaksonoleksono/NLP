import pickle
import utils

# ########################################################################
# # Membuat dictionary
########################################################################
def create_dictionary(source_lang, target_lang, data_path):
    train, _, _, max_sent_len = utils.get_data(data_path, source_lang, target_lang)
    source_sentences, target_sentences = utils.preprocess_data(
        train, source_lang, target_lang, max_sent_len, utils.normalizeString
    )
    input_dic = utils.Dictionary(source_lang)
    output_dic = utils.Dictionary(target_lang)
    for sentence in source_sentences:
        input_dic.add_sentence(sentence)
    for sentence in target_sentences:
        output_dic.add_sentence(sentence)
    # Save the dictionaries.
    save_dictionary(data_path, input_dic, source_lang, target_lang, input=True)
    save_dictionary(data_path, output_dic, source_lang, target_lang, input=False)
    
    return input_dic, output_dic, source_sentences, target_sentences

def save_dictionary(tp, dictionary, src, tgt, input=True):
    if input:
        with open(f'{tp}/{src}_{tgt}/input_dic.pkl', 'wb') as f:
            pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(f'{tp}/{src}_{tgt}/output_dic.pkl', 'wb') as f:
            pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
