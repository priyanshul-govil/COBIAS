import pandas as pd
import json
from tqdm import tqdm
import torch

from sentence_transformers import SentenceTransformer, util

# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the SentenceTransformer model with the specified device
sts_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

def get_similarity(sentence_1, sentence_2):
    """
    Given two sentences, return a cosine similarity score.
    """
    embeddings_1 = sts_model.encode(sentence_1, convert_to_tensor=True, device=device)
    embeddings_2 = sts_model.encode(sentence_2, convert_to_tensor=True, device=device)

    cosine_scores = util.cos_sim(embeddings_1, embeddings_2)
    similarity = cosine_scores[0][0].cpu().item()  # Move tensor to CPU before getting the item
    return similarity

output = [
    "similarity/gemma-1.1-2b-it.json", 
    "similarity/gemma-1.1-7b-it.json", 
    "similarity/gpt-3.5-turbo-instruct-0914.json", 
    "similarity/Meta-Llama-3-8B-Instruct.json", 
    "similarity/Phi-3-mini-4k-instruct.json", 
    "similarity/Phi-3-mini-128k-instruct.json", 
    "similarity/Mistral-7B-Instruct-v0.2.json"
]

models = [
    "./data_generated/google_gemma_2b/gemma-1.1-2b-it_1.", 
    "./data_generated/google_gemma_7b/gemma-1.1-7b-it_1.", 
    "./data_generated/gpt_3.5/gpt-3.5-turbo-instruct-0914_1.", 
    "./data_generated/meta_llama_8b/Meta-Llama-3-8B-Instruct_1.", 
    "./data_generated/microsoft_phi_4k/Phi-3-mini-4k-instruct_1.", 
    "./data_generated/microsoft_phi_128k/Phi-3-mini-128k-instruct_1.", 
    "./data_generated/mistral_7b/Mistral-7B-Instruct-v0.2_1."
]

for idx, model in tqdm(enumerate(models)):
    if idx > -1:
        similarity_scores = {}
        average_similarity = {}
        for k in tqdm(range(6)):
            df = pd.read_csv(f"{model}{k}.csv")
            scores = []
            for j in tqdm(df['index']):
                # if j ==1055:
                similarity_sum = 0
                count =0
                for a in range(1, 11):
                    for b in range(a + 1, 11):
                        try:
                            similarity = get_similarity(df[f"Column_{int(a)}"][int(j)], df[f"Column_{int(b)}"][int(j)])
                            count = count+1
                        except:
                            similarity =0
                        similarity_sum += similarity
                avg_similarity_per_datapoint = similarity_sum / count
                scores.append(avg_similarity_per_datapoint)
            similarity_scores[f'1.{k}'] = scores
            average_similarity[f'1.{k}'] = sum(scores) / len(scores)
        
        with open(output[idx], 'w') as json_file:
            json.dump(similarity_scores, json_file)
        with open(f"avg_{output[idx]}", 'w') as json_file:
            json.dump(average_similarity, json_file)
