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
        task="fill-mask",
        model=MODEL,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

def get_prompt(sentence):
    message =f'''You will be given a statement. Your task is to judge if there is sufficient context in the given sentence and return a value based on the amount of context present in the sentence. The rating should be done on a (1-5) Likert scale, where 1 is for 'almost no context present' and 5 is for 'almost perfectly sufficient context. For examples:\n
                Sentence: He quickly cancelled when he realized she was from Iraq.  Score: 2 
                In this statement, it is unclear what was cancelled. Depending on it (meeting, date, business transaction) different scenarios arise, and the semantics change accordingly.\n
                Sentence: I cannot stand this.  Score: 1
                Here, multiple scenarios are possible: weight of the backpack, a person's attitude, the winter's cold.\n
                Remember you need to rate the below statement on scale (1-5) depending on whether the context present is sufficient or not.Do not provide any kind of explanation and just return the score value.\n
                Sentence: {sentence}  Score: [MASK]
                '''
    
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
    
    return sequences

def process():
    
    for index, row in tqdm(data.iterrows()):
        sentence = row['context_points']
        id = row['id']
        prompt = get_prompt(sentence)
        output = generate(prompt)
        data[id] = output
        with open(f"./evals/{MODEL}.json", "wb") as f:
            json.dump(data , f)

if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:
        futures = executor.submit(process )

        for future in as_completed(futures):
            pass

