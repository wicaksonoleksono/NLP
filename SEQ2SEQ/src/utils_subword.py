# utils_subword.py
import os
import sentencepiece as spm
from utils import get_data, preprocess_data, normalizeString, MAX_SENT_LEN, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN

# ------------------------------------------------------------------------
# SUBWORD-TOKENIZATION FUNCTIONS
# ------------------------------------------------------------------------
def create_spm_training_file(data_path, src_lang, tgt_lang, save_dir, output_file="spm_train.txt"):
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, output_file)
    train_df, _, _, max_sent_len = get_data(data_path, src_lang, tgt_lang)
    source_sentences, _ = preprocess_data(train_df, src_lang, tgt_lang, max_sent_len, normalizeString)
    with open(output_path, "w", encoding="utf8") as f:
        for sentence in source_sentences:
            f.write(sentence + "\n")
    print(f"SentencePiece training file created at {output_path}")
    return output_path
def train_sentencepiece_model(training_file, save_dir, model_prefix="spm_model", vocab_size=5000, model_type="bpe"):
    os.makedirs(save_dir, exist_ok=True)
    model_prefix_path = os.path.join(save_dir, model_prefix)
    spm_args = f"--input={training_file} --model_prefix={model_prefix_path} --vocab_size={vocab_size} --model_type={model_type}"
    print("Training SentencePiece model with arguments:", spm_args)
    spm.SentencePieceTrainer.Train(spm_args)
    print(f"SentencePiece model trained and saved with prefix '{model_prefix_path}'.")
def load_sentencepiece_model(save_dir, model_prefix="spm_model"):
    sp = spm.SentencePieceProcessor()
    model_path = os.path.join(save_dir, f"{model_prefix}.model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please train the model first.")
    sp.Load(model_path)
    print("SentencePiece model loaded from", model_path)
    return sp

def sp_tokenize(sp, sentence):
    return sp.EncodeAsIds(sentence)
def sp_detokenize(sp, token_ids):
    return sp.DecodeIds(token_ids)

def sp_tokenize_with_specials(sp, sentence, max_length=MAX_SENT_LEN, sos_token=SOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN):
    token_ids = sp_tokenize(sp, sentence)
    token_ids = [sos_token] + token_ids + [eos_token]
    if len(token_ids) < max_length:
        token_ids += [pad_token] * (max_length - len(token_ids))
    else:
        token_ids = token_ids[:max_length]
    return token_ids
def sp_detokenize_with_specials(sp, token_ids, sos_token=SOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN):
    filtered_ids = [tid for tid in token_ids if tid not in (sos_token, eos_token, pad_token)]
    return sp_detokenize(sp, filtered_ids)
