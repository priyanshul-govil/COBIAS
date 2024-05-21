from transformers import pipeline, AutoTokenizer
import torch
import pandas as pd
from sys import argv
import json
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


MODEL = argv[1]
DATASET_PATH = argv[2]


tokenizer = AutoTokenizer.from_pretrained(MODEL)
data = pd.read_csv(DATASET_PATH)

pipe = pipeline(
        task="text-generation",
        model=MODEL,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

def get_prompt(sentence:str):
    message =  [
        {
            "role": "user",
            "content": f"Give a score in the range of (1-5) depending upon the contextual relevance of the given sentence. Just give the score and refrain from explanation. If the sentence is an error sentence then return a score of 0. ""\n\nSentence: {sentence}"
        }
    ]
    return message

def generate(prompt):
    sequences = pipe(
        prompt,
        max_new_tokens=128,
        do_sample=True,
        # top_k=10,
        num_return_sequences=1,
        temperature=1,
        top_p=1,
    )
    
    return sequences[0]['generated_text'][1]['content']

def process():
    data ={}
    for index, row in tqdm(data.iterrows()):
        sentence = row['context_points']
        id = row['id']
        prompt = get_prompt(sentence)
        output = generate(prompt)
        data[id] = output
        with open(f"../evals_generated/{MODEL}.json", "wb") as f:
            json.dump(data , f)

if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:
        futures = executor.submit(process )

        for future in as_completed(futures):
            pass
