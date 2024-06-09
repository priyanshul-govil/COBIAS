# import difflib
import torch


def get_log_prob_unigram(masked_token_ids, token_ids, mask_idx, lm):
    """
    Given a sequence of token ids, with one masked token, return the log probability of the masked token.
    """
    
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    # get model hidden states
    output = model(masked_token_ids)
    hidden_states = output[0].squeeze(0)
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)

    # we only need log_prob for the MASK tokens
    assert masked_token_ids[0][mask_idx] == mask_id

    hs = hidden_states[mask_idx]
    target_id = token_ids[0][mask_idx]
    log_probs = log_softmax(hs)[target_id]

    return log_probs


def mask_unigram(sent, lm):
    """
    Score each sentence by masking one word at a time.
    The score for a sentence is the sum of log probability of each word in
    the sentence.
    n = n-gram of token that is masked, if n > 1, we mask tokens with overlapping
    n-grams.
    """
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]
    device = next(model.parameters()).device

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if uncased:
        sent = sent.lower()

    # tokenize
    sent_token_ids = tokenizer.encode(sent, add_special_tokens=False, return_tensors='pt').to(device)

    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    
    sent_log_probs = 0.
    total_masked_tokens = 0

    for i in range(len(sent_token_ids[0])):
        sent_masked_token_ids = sent_token_ids.clone()
        sent_masked_token_ids[0][i] = mask_id
        total_masked_tokens += 1
        score = get_log_prob_unigram(sent_masked_token_ids, sent_token_ids, i, lm)
        sent_log_probs += score.item()

    return abs(sent_log_probs / total_masked_tokens)


