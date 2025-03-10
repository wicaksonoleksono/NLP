import utils 
import torch
import heapq
import utils_subword
def translate_sentence_gru(token_ids, input_dic, output_dic, model, device, max_len=50):
    model.eval()
    # Encode the source sentence
    src_tensor = torch.LongTensor(token_ids).unsqueeze(0).to(device)
    with torch.no_grad():
        encoder_outputs, encoder_hidden = model.encoder(src_tensor)
    
    # Prepare the initial decoder hidden state (using bridge if available)
    if hasattr(model, "bridge") and model.bridge is not None:
        dec_hidden = torch.tanh(model.bridge(encoder_hidden))
    else:
        dec_hidden = encoder_hidden

    predicted_tokens = [utils.SOS_TOKEN]
    input_token = torch.LongTensor([utils.SOS_TOKEN]).unsqueeze(0).to(device)
    
    for _ in range(max_len):
        with torch.no_grad():
            output, dec_hidden = model.decoder(input_token, dec_hidden)
        if output.dim() == 3:
            logits = output.squeeze(0).squeeze(0)
        elif output.dim() == 2:
            logits = output.squeeze(0)
        else:
            logits = output
        sorted_indices = torch.argsort(logits, descending=True)
        
        next_token = None
        for candidate in sorted_indices:
            candidate = candidate.item()
            if candidate in predicted_tokens:
                continue
            if candidate == utils.UNK_TOKEN:
                continue
            if candidate == utils.SOS_TOKEN:
                continue
            next_token = candidate
            break
        # Fallback: if no valid candidate found, use the top candidate.
        if next_token is None:
            next_token = sorted_indices[0].item()
            
        predicted_tokens.append(next_token)
        
        if next_token == utils.EOS_TOKEN:
            break
        input_token = torch.LongTensor([next_token]).unsqueeze(0).to(device)
    
    # Remove special tokens and detokenize
    ignore_tokens = {utils.SOS_TOKEN, utils.EOS_TOKEN, utils.PAD_TOKEN}
    translation_tokens = [idx for idx in predicted_tokens if idx not in ignore_tokens]
    translation_str = ' '.join(output_dic.index2word[idx] for idx in translation_tokens)
    return translation_str, predicted_tokens


def translate_sentence(token_ids, input_dic, output_dic, model, device, max_len=utils.MAX_SENT_LEN):
    model.eval()
    input_tensor = torch.LongTensor(token_ids).unsqueeze(0).to(device)
    src_mask = model.make_input_mask(input_tensor)
    with torch.no_grad():
        memory = model.encoder(input_tensor, src_mask)
        
    predicted_tokens = [utils.SOS_TOKEN]
    for _ in range(max_len):
        tgt_tensor = torch.LongTensor(predicted_tokens).unsqueeze(0).to(device)
        tgt_mask = model.make_target_mask(tgt_tensor)
        with torch.no_grad():
            output, _ = model.decoder(tgt_tensor, memory, tgt_mask, src_mask)
        logits = output[0, -1] # take index
        sorted_indices = torch.argsort(logits, descending=True)
        next_token = None
        for candidate in sorted_indices:
            candidate = candidate.item()
            if candidate in predicted_tokens:
                continue
            if candidate == utils.UNK_TOKEN:
                continue
            if candidate == utils.SOS_TOKEN:
                continue
            next_token = candidate
            break
        if next_token is None:
            next_token = sorted_indices[0].item()
        predicted_tokens.append(next_token)
        if next_token == utils.EOS_TOKEN:
            break

    translation_str = utils.detokenize(predicted_tokens, output_dic)
    return translation_str, predicted_tokens


def translate_sentence_beam(token_ids, input_dic, output_dic, model, device, max_len=utils.MAX_SENT_LEN, beam_width=5):
    model.eval()
    input_tensor = torch.LongTensor(token_ids).unsqueeze(0).to(device)
    src_mask = model.make_input_mask(input_tensor)
    with torch.no_grad():
        memory = model.encoder(input_tensor, src_mask)
    beam = [([utils.SOS_TOKEN], 0.0)]
    for _ in range(max_len):
        candidates = []
        for seq, score in beam:
            if seq[-1] == utils.EOS_TOKEN:
                candidates.append((seq, score))
                continue
            tgt_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
            tgt_mask = model.make_target_mask(tgt_tensor)
            with torch.no_grad():
                output, _ = model.decoder(tgt_tensor, memory, tgt_mask, src_mask)
            log_probs = torch.log_softmax(output[0, -1], dim=-1)  # shape: [vocab_size]
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
            for k in range(beam_width):
                next_token = topk_indices[k].item()
                candidate_seq = seq + [next_token]
                candidate_score = score + topk_log_probs[k].item()
                candidates.append((candidate_seq, candidate_score))
        
        beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        if all(seq[-1] == utils.EOS_TOKEN for seq, _ in beam):
            break
    best_seq, best_score = beam[0]
    translation_str = utils.detokenize(best_seq, output_dic)
    return translation_str, best_seq

def translate_sentence_piece(token_ids, sp, model, device, max_len=utils.MAX_SENT_LEN):
    model.eval()
    input_tensor = torch.LongTensor(token_ids).unsqueeze(0).to(device)
    src_mask = model.make_input_mask(input_tensor)
    with torch.no_grad():
        memory = model.encoder(input_tensor, src_mask)
    predicted_tokens = [utils.SOS_TOKEN]
    for _ in range(max_len):
        tgt_tensor = torch.LongTensor(predicted_tokens).unsqueeze(0).to(device)
        tgt_mask = model.make_target_mask(tgt_tensor)
        with torch.no_grad():
            output, _ = model.decoder(tgt_tensor, memory, tgt_mask, src_mask)
        logits = output[0, -1]
        sorted_indices = torch.argsort(logits, descending=True)
        next_token = None
        for candidate in sorted_indices:
            candidate = candidate.item()
            if candidate == utils.SOS_TOKEN or candidate == utils.UNK_TOKEN:
                continue
            if candidate in predicted_tokens:
                continue
            next_token = candidate
            break
        if next_token is None:
            next_token = sorted_indices[0].item()
        predicted_tokens.append(next_token)
        if next_token == utils.EOS_TOKEN:
            break
    translation_str = utils_subword.sp_detokenize_with_specials(sp, predicted_tokens)
    return translation_str, predicted_tokens

