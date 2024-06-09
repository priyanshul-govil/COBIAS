from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

def load_models_tokenizers(model_names: list):

    models_tokenizers = []
    for model_name in model_names:
        models_tokenizers.append((
            AutoTokenizer.from_pretrained(model_name),
            AutoModelForMaskedLM.from_pretrained(model_name),
            "True",
            model_name
        ))

    for _ in models_tokenizers:
        if torch.cuda.is_available():
            _[1].to(torch.device(f'cuda'))
        _[1].eval()

    return models_tokenizers