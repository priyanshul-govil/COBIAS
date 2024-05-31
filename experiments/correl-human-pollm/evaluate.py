from transformers import pipeline, AutoTokenizer
import torch
from sys import argv
from tqdm import tqdm
import pandas as pd


MODEL = argv[1]

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

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

def get_prompt(sentence):
    messages = [
        {
            "role": "user",
            "content": "Following is a statement. Your task is to judge if there is sufficient context in the given sentence. The scoring should be done on a (1-5) Likert scale, where 1 is for 'almost no context present' and 5 is for 'almost perfectly sufficient context.\n\nFor example:\nSentence: He quickly cancelled when he realized she was from Iraq.\nIn this statement, it is unclear what was cancelled. Depending on it (meeting, date, business transaction), different scenarios arise, and the semantics change accordingly.\n\nSentence: I cannot stand this.\nHere, multiple scenarios are possible: weight of the backpack, a person's attitude, the winter's cold.\n\nRemember you need to rate the below statement on scale (1-5) depending on whether the present context is sufficient. Do not provide any explanation and output only the score value.",
        },
        {
            "role": "assistant",
            "content": "I have understood the task. You want me to score a statement on a Likert scale of 1-5. Please give me the statement, and I will return a singular value which would be the score.",
        },
        {
            "role": "user",
            "content": sentence,
        }
    ]
    
    return messages

def generate(prompt):
    sequences = pipe(
        prompt,
        max_new_tokens=10,
        do_sample=True,
        num_return_sequences=1,
        temperature=0.1,
        top_p=1,
    )
    
    return sequences[0]['generated_text'][3]['content']


if __name__ == "__main__":
    
    data = pd.read_csv("/home/pgovil1/correl-human-pollm/data.csv")

    ratings = []

    sentences = data['sentence'].tolist()
    for sentence in tqdm(sentences):
        prompt = get_prompt(sentence)
        rating = generate(prompt)

        # Find numbers in the generated text
        rating = '\t'.join(filter(str.isdigit, rating))
        
        ratings.append(rating)  


    model_name = MODEL.replace("/", "_")
    with open(f"/home/pgovil1/correl-human-pollm/ratings/{model_name}.txt", "w") as f:
        for rating in ratings:
            f.write(f"{rating}\n")