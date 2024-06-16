from transformers import pipeline, AutoTokenizer
import torch
import pandas as pd
from sys import argv
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


MODEL = argv[1]
DATASET_PATH = argv[2]
TEMPERATURES = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

DF_BATCH_COUNT = argv[3]
DF_BATCH_SEGMENT = argv[4]

tokenizer = AutoTokenizer.from_pretrained(MODEL)

df = pd.read_csv(DATASET_PATH)

batch_size = len(df) // int(DF_BATCH_COUNT)
df_batch = [df[i : i+batch_size] for i in range(0, len(df), batch_size)]
data = df_batch[int(DF_BATCH_SEGMENT)]


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
            "content": f"Fill in the blanks with information. Do not modify anything else. There must not be any additional output. Return the entire sentence with the filled-in blanks.\n\n{sentence}"
        }
    ]
    return message


def generate(prompt, temp):
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


def process_for_temperature(temp):
    generations = {}
    for index, row in tqdm(data.iterrows()):
        generations[row['id']] = []
        sentence = row['context_points']
        id = row['id']
        
        prompt = get_prompt(sentence)
        for i in range(10):
            output = generate(prompt, temp)
            generations[id].append(output)

    with open(f"/home/pgovil1/generations/generations_{MODEL}_{temp}_{DF_BATCH_SEGMENT}.pkl", "wb") as f:
        pickle.dump(generations, f)



if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_for_temperature, temp) for temp in TEMPERATURES]

        for future in as_completed(futures):
            pass
