from transformers import pipeline, AutoTokenizer
import torch
import pandas as pd
from sys import argv
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


MODEL = "google/gemma-1.1-7b-it"
DATASET_PATH = "data.csv"


tokenizer = AutoTokenizer.from_pretrained(MODEL)

df = pd.read_csv(DATASET_PATH)


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
            "content": f"Fill in the blanks with information. Do not modify anything else. There must not be any additional output.\n\n{sentence}"
        }
    ]
    return message


def generate(prompt, temp=1.5):
    sequences = pipe(
        prompt,
        max_new_tokens=128,
        do_sample=True,
        # top_k=10,
        num_return_sequences=1,
        temperature=temp,
        top_p=0.9,
    )
    
    return sequences[0]['generated_text'][1]['content'] 


def process_data_point(row, temp=1.5):
    generations = []
    sentence = row['context_points']
    id = row['id']
    prompt = get_prompt(sentence)
        
    for i in range(20):
        output = generate(prompt, temp)
        generations.append(output)

    with open(f"data/{id}.pkl", "wb") as f:
        pickle.dump(generations, f)



if __name__ == "__main__":

    with tqdm(total=len(df)) as pbar:
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(process_data_point, row) for _, row in df.iterrows()]

            for future in as_completed(futures):
                pbar.update(1)
