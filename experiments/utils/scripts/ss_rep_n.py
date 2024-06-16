import matplotlib.pyplot as plt
import pandas as pd
import json
from tqdm import tqdm
import torch

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

input_file = "abc.csv"
output_scores = "similarity.json"
output_avg = "average_similarity.json"

num_generations = 6
similarity_scores = {}
avg_similarity = {}

df = pd.read_csv(input_file)

for n in range(2 , num_generations +1):
    scores = []
    for j in tqdm(df['index']):
        similarity_sum = 0
        count =0
        for a in range(1, n):
            for b in range(a + 1, n+1):
                try:
                    similarity = get_similarity(df[f"Column_{int(a)}"][int(j)], df[f"Column_{int(b)}"][int(j)])
                    count = count+1
                except:
                    similarity =0
                similarity_sum += similarity
        avg_similarity_per_datapoint = similarity_sum / count
        scores.append(avg_similarity_per_datapoint)
    similarity_scores[n] = scores
    avg_similarity[n] = sum(scores) / len(scores)
    
with open(output_scores, 'w') as json_file:
    json.dump(similarity_scores, json_file)
with open(output_avg, 'w') as json_file:
    json.dump(avg_similarity, json_file)
    

# keys = list(avg_similarity.keys())
# values = list(avg_similarity.values())

# plt.figure(figsize=(10, 5))
# plt.plot(keys, values, marker='o', linestyle='-', color='b')

# plt.xlabel('Keys')
# plt.ylabel('Values')
# plt.title('Plot of Dictionary Keys vs. Values')

# plt.grid(True)
# plt.show()