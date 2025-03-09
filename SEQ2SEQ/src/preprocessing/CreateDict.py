import pickle
import utils
import os
# ########################################################################
# # Membuat dictionary
########################################################################
def create_dictionary(source_lang, target_lang, data_path):

    train_df, _, _, max_sent_len = utils.get_data(data_path, source_lang, target_lang)
    source_sentences, target_sentences = utils.preprocess_data(
        train_df, source_lang, target_lang, max_sent_len, utils.normalizeString
    )
    input_dic = utils.Dictionary(source_lang)
    output_dic = utils.Dictionary(target_lang)
    for sentence in source_sentences:
        input_dic.add_sentence(sentence)
    for sentence in target_sentences:
        output_dic.add_sentence(sentence)
    save_dir = os.path.join(data_path, f"{source_lang}_{target_lang}")
    os.makedirs(save_dir, exist_ok=True)
    # Save the dictionaries
    save_dictionary(save_dir, input_dic, input=True)
    save_dictionary(save_dir, output_dic, input=False)

    return input_dic, output_dic, source_sentences, target_sentences
def save_dictionary(folder_path, dictionary, input=True):
    filename = "input_dic.pkl" if input else "output_dic.pkl"
    with open(os.path.join(folder_path, filename), 'wb') as f:
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)



