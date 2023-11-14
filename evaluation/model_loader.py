from transformers import BertTokenizer, BertForMaskedLM
from transformers import AlbertTokenizer, AlbertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM

import torch

def load_models_tokenizers():
    models_tokenizers = [(
        BertTokenizer.from_pretrained('bert-large-uncased'),
        BertForMaskedLM.from_pretrained('bert-large-uncased'),
        True,
        "bert-large-uncased"
    ), (
        RobertaTokenizer.from_pretrained('roberta-large'),
        RobertaForMaskedLM.from_pretrained('roberta-large'),
        False,
        "roberta-large"
    ), (
        AlbertTokenizer.from_pretrained('albert-xxlarge-v2'),
        AlbertForMaskedLM.from_pretrained('albert-xxlarge-v2'),
        True,
        "albert-xxlarge-v2"
    )]

    for _ in models_tokenizers:
        if torch.cuda.is_available():
            _[1].to(torch.device(f'cuda'))
        _[1].eval()

    return models_tokenizers