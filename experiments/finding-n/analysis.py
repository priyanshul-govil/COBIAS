
from tqdm import tqdm
import os
import numpy as np 
import pickle
import torch
from sys import argv

from sentence_transformers import SentenceTransformer, util

device = "cuda" if torch.cuda.is_available() else "cpu"

sts_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

def get_similarity(sentence_1, sentence_2):
    """
    Given two sentences, return a cosine similarity score.
    """
    embeddings_1 = sts_model.encode(sentence_1, convert_to_tensor=True, device=device)
    embeddings_2 = sts_model.encode(sentence_2, convert_to_tensor=True, device=device)

    cosine_scores = util.cos_sim(embeddings_1, embeddings_2)
    similarity = cosine_scores[0][0].cpu().item()  
    return similarity



file = argv[1]


with open(f'data/{file}', 'rb') as f:
    data = pickle.load(f)
data_id = file.split('.')[0]
sum = 0
count = 0
scores = []

os.makedirs('results', exist_ok=True)


for n in tqdm(range(1, 100), total=99):
    for a in range(0, n):
        sum += get_similarity(data[a].replace('\n' , ''), data[n].replace('\n' , ''))
        count += 1
    avg = sum/count
    scores.append(avg)

with open(f'results/{data_id}.pkl', 'wb') as f:
    pickle.dump(scores, f)