from helper import mask_unigram
from model_loader import load_models_tokenizers
import torch
import argparse
import logging
import pandas as pd
import pickle
from tqdm import tqdm
from math import log
import numpy as np


num_of_augmentations = 35
np.random.seed(42)


def process_row(data):
    original_sentence = data['sentence']
    id = data['id']
    # target = data['target_term']
    augmented_sentences_scores = []
    augmented_sentences_PLLs = []
    original_PLL = []

    # Load final_augmented_sentences/{id}.pkl
    try:
        with open(f"{args.data_folder}/{id}.pkl", "rb") as f:
            augmented_sentences = pickle.load(f)
    except FileNotFoundError:
        return None

    # Generation from GPT-3.5 contain leading newline characters
    for i in range(len(augmented_sentences)):
        augmented_sentences[i] = augmented_sentences[i].replace('\n', '')

    np.random.shuffle(augmented_sentences)
    augmented_sentences = augmented_sentences[:num_of_augmentations]
    N = len(augmented_sentences)

    original_sentence_score = 0.
    for _ in models_tokenizers:
        tokenizer = _[0]
        model = _[1]
        uncased = _[2]
        model_name = _[3]

        mask_token = tokenizer.mask_token
        log_softmax = torch.nn.LogSoftmax(dim=0)

        lm = {"model": model,
            "tokenizer": tokenizer,
            "mask_token": mask_token,
            "log_softmax": log_softmax,
            "uncased": uncased
            }
         
        pll = mask_unigram(original_sentence, lm)
        original_PLL.append(pll)
        original_sentence_score += pll
    original_sentence_score /= len(models_tokenizers)

    for augmented_sent in augmented_sentences:
        score_sum = 0
        pll_list = []
        for _ in models_tokenizers:
            tokenizer = _[0]
            model = _[1]
            uncased = _[2]
            model_name = _[3]

            mask_token = tokenizer.mask_token
            log_softmax = torch.nn.LogSoftmax(dim=0)

            lm = {"model": model,
                "tokenizer": tokenizer,
                "mask_token": mask_token,
                "log_softmax": log_softmax,
                "uncased": uncased
                }

            pll = mask_unigram(augmented_sent, lm)
            pll_list.append(pll)
            score_sum += pll

        augmented_sentences_PLLs.append(pll_list)
        avg_score = score_sum / len(models_tokenizers)
        augmented_sentences_scores.append(avg_score)


    # Calculate context variance
    cv_score = 0.
    for score in augmented_sentences_scores:
        cv_score += (score - original_sentence_score) ** 2
    cv_score /= N

    cv_score = (cv_score / original_sentence_score) * 100
    
    cobias = log(1 + cv_score) / (1 + log(1 + cv_score))

    return{
        'id': id,
        'augmented_scores_list': augmented_sentences_scores,
        'original_score': original_sentence_score,
        'original_PLL': original_PLL,
        'augmented_PLLs': augmented_sentences_PLLs,
        'context_variance': cv_score,
        'cobias': cobias
    }


def evaluate(args, df_data):
    try:
        print("Evaluating:")
        print("Input:", args.input_file)
        print("=" * 100)
    except Exception as e:
        pass

    logging.basicConfig(level=logging.INFO)

    # score each augmented sentence.
    total = len(df_data)
    df_score = pd.DataFrame()

    with tqdm(total=total) as pbar:
        for _, data in df_data.iterrows():
            try:
                df_score = df_score._append(process_row(data), ignore_index=True)
            except Exception as e:
                with open("error_log.txt", "a") as f:
                    f.write(f"{data['id']}: {e}\n")
            pbar.update(1)
    try:
        df_score.to_csv(args.output_file)
    except Exception as e:
        return df_score
    print('=' * 100)
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", type=str, help="comma separated list of model names (model_1, model_2, ...)",
                        default="google-bert/bert-base-uncased, FacebookAI/roberta-base, albert/albert-base-v2")
    parser.add_argument("--input_file", type=str, help="path to dataset")
    parser.add_argument("--output_file", type=str, help="path to output file with sentence scores")
    parser.add_argument("--data_folder", type=str, help="path to folder containing the context added pkl files")

    args = parser.parse_args()

    # load data into panda DataFrame
    df_data = pd.read_csv(args.input_file)

    model_names = args.model_names.split(', ')

    # models and tokenizers (model, tokenizer, uncased: bool, model_name)  
    models_tokenizers = load_models_tokenizers(model_names=model_names)
    
    evaluate(args, df_data)