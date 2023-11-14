import difflib
import torch

from sentence_transformers import SentenceTransformer, util
sts_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')


def get_dissimiliarity(sentence_1, sentence_2):
    """
    Given two sentences, return a matrix of cosine similarity scores.
    """

    embeddings_1 = sts_model.encode(sentence_1, convert_to_tensor=True)
    embeddings_2 = sts_model.encode(sentence_2, convert_to_tensor=True)

    cosine_scores = util.cos_sim(embeddings_1, embeddings_2)
    dissimilarity = 1 - cosine_scores[0][0].cpu().item()
    return dissimilarity


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


def get_context(sent, target):
    """
    This function extracts context of `sent` sentence by removing the `target` span.
    """

    sent = [str(x) for x in sent.tolist()]
    target = [str(x) for x in target.tolist()]

    matcher = difflib.SequenceMatcher(None, sent, target)
    context_idxs = []
    for op in matcher.get_opcodes():
        # each op is a list of tuple: 
        # (operation, pro_idx_start, pro_idx_end, anti_idx_start, anti_idx_end)
        # possible operation: replace, insert, equal
        # https://docs.python.org/3/library/difflib.html
        if op[0] != 'equal':
            context_idxs += [x for x in range(op[1], op[2], 1)]

    return context_idxs


def mask_unigram(sent, target, lm, n=1):
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
    sent_token_ids = tokenizer.encode(sent, return_tensors='pt').to(device)
    target_token_ids = tokenizer.encode(target, return_tensors='pt').to(device)

    # get spans of non-changing tokens
    context = get_context(sent_token_ids[0], target_token_ids[0])

    N = len(context)  # num. of tokens that can be masked
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    
    sent_log_probs = 0.
    total_masked_tokens = 0

    # skipping CLS and SEP tokens, they'll never be masked
    for i in range(1, N-1):
        sent_masked_token_ids = sent_token_ids.clone()
        sent_masked_token_ids[0][context[i]] = mask_id
        total_masked_tokens += 1
        score = get_log_prob_unigram(sent_masked_token_ids, sent_token_ids, context[i], lm)
        sent_log_probs += score.item()

    return abs(sent_log_probs / total_masked_tokens)


