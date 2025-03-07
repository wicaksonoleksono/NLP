import utils  # For PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, etc.
import torch
import heapq

def translate_sentence(token_ids, input_dic, output_dic, model, device, max_len=50):
    """
    Greedy decoding (argmax at each step).
    Returns: (translation_str, predicted_token_list)
    """
    model.eval()
    # 1) Encode source
    input_tensor = torch.LongTensor(token_ids).unsqueeze(0).to(device)
    src_mask = model.make_input_mask(input_tensor)
    with torch.no_grad():
        memory = model.encoder(input_tensor, src_mask)

    # 2) Start decoding
    predicted_tokens = [utils.SOS_TOKEN]
    for _ in range(max_len):
        tgt_tensor = torch.LongTensor(predicted_tokens).unsqueeze(0).to(device)
        tgt_mask = model.make_target_mask(tgt_tensor)
        with torch.no_grad():
            output, _ = model.decoder(tgt_tensor, memory, tgt_mask, src_mask)
        # shape = [1, current_len, vocab_size]
        next_token = output[0, -1].argmax(dim=-1).item()
        predicted_tokens.append(next_token)
        if next_token == utils.EOS_TOKEN:
            break

    # 3) Convert predicted tokens to string
    ignore_tokens = {utils.SOS_TOKEN, utils.EOS_TOKEN, utils.PAD_TOKEN}
    translation_tokens = [idx for idx in predicted_tokens[1:] if idx not in ignore_tokens]
    translation_str = ' '.join(output_dic.index2word[idx] for idx in translation_tokens)
    return translation_str, predicted_tokens
def translate_sentence_gru(token_ids, input_dic, output_dic, model, device, max_len=50):

    model.eval()
    # 1) Convert source token ids into a tensor (shape: [1, src_len]) and encode
    src_tensor = torch.LongTensor(token_ids).unsqueeze(0).to(device)
    with torch.no_grad():
        encoder_outputs, encoder_hidden = model.encoder(src_tensor)
    
    # 2) Prepare the initial hidden state for the decoder.
    # If a bridge exists to match the dimension (from 2*enc_hidden to dec_hidden), apply it.
    if hasattr(model, "bridge") and model.bridge is not None:
        dec_hidden = torch.tanh(model.bridge(encoder_hidden))
    else:
        dec_hidden = encoder_hidden

    # 3) Start decoding with the SOS token.
    predicted_tokens = [utils.SOS_TOKEN]  # Replace with your actual SOS token index.
    input_token = torch.LongTensor([utils.SOS_TOKEN]).unsqueeze(0).to(device)  # shape: (1, 1)

    for _ in range(max_len):
        with torch.no_grad():
            # decoder takes the current token and hidden state and outputs a prediction and updated hidden state.
            output, dec_hidden = model.decoder(input_token, dec_hidden)
        # output shape: (1, 1, vocab_size) --> choose the token with highest probability.
        next_token = output.argmax(dim=-1).item()
        predicted_tokens.append(next_token)
        if next_token == utils.EOS_TOKEN:  # Replace with your EOS token index.
            break
        # Prepare the next input token for the decoder.
        input_token = torch.LongTensor([next_token]).unsqueeze(0).to(device)

    # 4) Convert predicted token indices into words.
    ignore_tokens = {utils.SOS_TOKEN, utils.EOS_TOKEN, utils.PAD_TOKEN}  # adjust as needed
    translation_tokens = [idx for idx in predicted_tokens if idx not in ignore_tokens]
    translation_str = ' '.join(output_dic.index2word[idx] for idx in translation_tokens)
    
    return translation_str, predicted_tokens


def translate_sentence_beam(
    token_ids,
    input_dic,
    output_dic,
    model,
    device,
    beam_size=5,
    max_len=50,
    alpha=0.8
):
    """
    Beam search decoding. Returns: (translation_str, best_sequence_token_ids)
    """
    model.eval()

    # 1) Encode the source
    src_tensor = torch.LongTensor(token_ids).unsqueeze(0).to(device)
    src_mask = model.make_input_mask(src_tensor)
    with torch.no_grad():
        memory = model.encoder(src_tensor, src_mask)

    # 2) Initialize beam with (log_prob=0.0, sequence=[SOS])
    beam = [(0.0, [utils.SOS_TOKEN])]
    # We'll store completed sequences here (if they end before max_len)
    completed = []

    # 3) Expand for up to max_len steps
    for _ in range(max_len):
        new_beam = []
        all_finished = True  # if all beams end with EOS, we can stop early

        for log_prob, seq in beam:
            # If this beam already ended, just carry it forward
            if seq[-1] == utils.EOS_TOKEN:
                new_beam.append((log_prob, seq))
                continue

            all_finished = False

            # 3a) Decode one step
            tgt_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
            tgt_mask = model.make_target_mask(tgt_tensor)
            with torch.no_grad():
                decoder_output, _ = model.decoder(tgt_tensor, memory, tgt_mask, src_mask)

            # 3b) Get log probs for the last step
            step_log_probs = torch.log_softmax(decoder_output[0, -1], dim=-1)
            # shape = [vocab_size]

            # 3c) Expand top beam_size
            topk = torch.topk(step_log_probs, beam_size)
            for next_token_id, token_log_prob in zip(topk.indices, topk.values):
                next_token_id = next_token_id.item()
                token_log_prob = token_log_prob.item()

                new_seq = seq + [next_token_id]
                new_log_prob = log_prob + token_log_prob
                new_beam.append((new_log_prob, new_seq))

        # 3d) Keep only beam_size best expansions
        beam = heapq.nlargest(beam_size, new_beam, key=lambda x: x[0])

        # 3e) If all beams are finished, break early
        if all_finished:
            break

    # 4) After max_len steps, maybe some beams are incomplete
    completed += beam

    # 5) Re-rank by length penalty
    def length_penalty_score(total_log_prob, seq):
        length = len(seq)
        # GNMT length penalty
        lp = (5 + length)**alpha / (5 + 1)**alpha
        return total_log_prob / lp

    best_logprob, best_seq = max(completed, key=lambda x: length_penalty_score(x[0], x[1]))

    # 6) Convert tokens to string
    ignore_tokens = {utils.SOS_TOKEN, utils.EOS_TOKEN, utils.PAD_TOKEN}
    translation_tokens = [idx for idx in best_seq[1:] if idx not in ignore_tokens]
    translation_str = " ".join(output_dic.index2word[idx] for idx in translation_tokens)

    return translation_str, best_seq
