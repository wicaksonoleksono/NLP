import os
import pickle
from torch.utils.data import DataLoader
from ._dataset import CustomDataset
from ._collate_fn import collate_fn
import utils

def get_dataloaders(data_path, source_lang, target_lang, batch_size, device):
    # 1. Get raw data and max sentence length
    train_df, valid_df, test_df, max_sent_len = utils.get_data(data_path, source_lang, target_lang)
    
    # 2. Load the dictionaries (they should be stored as .pkl files)
    path_dic = os.path.join(data_path, f"{source_lang}_{target_lang}")
    input_dic_path = os.path.join(path_dic, "input_dic.pkl")
    output_dic_path = os.path.join(path_dic, "output_dic.pkl")
    
    with open(input_dic_path, "rb") as f:
        input_dic = pickle.load(f)
    with open(output_dic_path, "rb") as f:
        output_dic = pickle.load(f)
    
    # 3. Preprocess the raw sentences (normalize and filter by length)
    train_src_raw, train_tgt_raw = utils.preprocess_data(train_df, source_lang, target_lang, max_sent_len, utils.normalizeString)
    valid_src_raw, valid_tgt_raw = utils.preprocess_data(valid_df, source_lang, target_lang, max_sent_len, utils.normalizeString)
    test_src_raw, test_tgt_raw   = utils.preprocess_data(test_df, source_lang, target_lang, max_sent_len, utils.normalizeString)
    
    # 4. Tokenize each sentence using the appropriate dictionary
    train_src = [utils.tokenize(sentence, input_dic, max_sent_len) for sentence in train_src_raw]
    train_tgt = [utils.tokenize(sentence, output_dic, max_sent_len) for sentence in train_tgt_raw]
    
    valid_src = [utils.tokenize(sentence, input_dic, max_sent_len) for sentence in valid_src_raw]
    valid_tgt = [utils.tokenize(sentence, output_dic, max_sent_len) for sentence in valid_tgt_raw]
    
    test_src  = [utils.tokenize(sentence, input_dic, max_sent_len) for sentence in test_src_raw]
    test_tgt  = [utils.tokenize(sentence, output_dic, max_sent_len) for sentence in test_tgt_raw]
    
    # 5. Create datasets using tokenized data
    train_dataset = CustomDataset({'src': train_src, 'tgt': train_tgt})
    valid_dataset = CustomDataset({'src': valid_src, 'tgt': valid_tgt})
    test_dataset  = CustomDataset({'src': test_src, 'tgt': test_tgt})
    
    # 6. Create dataloaders with the custom collate_fn to handle padding
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_dataloader, valid_dataloader, test_dataset
