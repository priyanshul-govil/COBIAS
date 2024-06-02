import pandas as pd
import re
from tqdm import tqdm
import json

def get_edit_distance(sentence1, sentence2):
    ''' Sentence 1 is the original sentence with context addition poinsts and sentence 2 is the context-generated sentence by the LM '''
    m = len(sentence1)
    n = len(sentence2)
    
    ''' Find the Longest Common Subsequence of two strings'''
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if sentence1[i - 1] == sentence2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if sentence1[i - 1] == sentence2[j - 1]:
            lcs.append(sentence1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    lcs.reverse()
    
    
    ''' Find the minimum number of operations to convert sentence1 to sentence2 using the Longest Common Subsequence'''
    y=0
    count1 =0
    count2 =0
    for i in lcs:
        list1 = []
        list2 = []
        for j in range(count1,len(sentence1)):
            if i!=sentence1[j]:
                count1 +=1
                list1.append(sentence1[j])
            else:
                count1 +=1
                break
        for j in range(count2,len(sentence2)):
            if i!=sentence2[j]:
                count2 +=1
                list2.append(j)
            else:
                count2 +=1
                break
        if "___" in list1:
            y+=(len(list1) - 1)
        else:
            y+=len(list1) + len(list2)

    list1 = []
    list2 = []
    for j in range(count1,len(sentence1)):
        if i!=sentence1[j]:
            count1 +=1
            list1.append(sentence1[j])
        else:
            count1 +=1
            break
    for j in range(count2,len(sentence2)):
        if i!=sentence2[j]:
            count2 +=1
            list2.append(j)
        else:
            count2 +=1
            break
    if "___" in list1:
        y+=(len(list1) - 1)
    else:
        y+=len(list1) +len(list2)
    
    ''' Find the minimum number of operations to convert sentence2 to sentence1 using the Longest Common Subsequence using backward iteration'''
    
    y1=0 
    lcs.reverse()
    sentence1.reverse()
    sentence2.reverse()
    
    count1 =0
    count2 =0
    for i in lcs:
        list1 = []
        list2 = []
        for j in range(count1,len(sentence1)):
            if i!=sentence1[j]:
                count1 +=1
                list1.append(sentence1[j])
            else:
                count1 +=1
                break
        for j in range(count2,len(sentence2)):
            if i!=sentence2[j]:
                count2 +=1
                list2.append(j)
            else:
                count2 +=1
                break
        if "___" in list1:
            y1+=(len(list1) - 1)
        else:
            y1+=len(list1)+len(list2)

    list1 = []
    list2 = []
    for j in range(count1,len(sentence1)):
        if i!=sentence1[j]:
            count1 +=1
            list1.append(sentence1[j])
        else:
            count1 +=1
            break
    for j in range(count2,len(sentence2)):
        if i!=sentence2[j]:
            count2 +=1
            list2.append(j)
        else:
            count2 +=1
            break
    if "___" in list1:
        y1+=(len(list1) - 1)
    else:
        y1+=len(list1) +len(list2)
    
    
    return min(y,y1)


# sentence1 = "Angie couldn't find ___ good workout clothes in her large size ___."
# sentence2 = " Angie couldn't find a good pair of workout pants in her large size. She also had a hard time finding a well-fitting workout top in her large size."
# result = longest_common_subsequence(sentence1.replace('.' , '').split(), sentence2.replace('.' , '').split())
# print(result)

output = [
    "../data/edit_distance/gemma-1.1-2b-it.json", 
    "../data/edit_distance/gemma-1.1-7b-it.json", 
    "../data/edit_distance/gpt-3.5-turbo-instruct-0914.json", 
    "../data/edit_distance/Meta-Llama-3-8B-Instruct.json", 
    "../data/edit_distance/Phi-3-mini-4k-instruct.json", 
    "../data/edit_distance/Phi-3-mini-128k-instruct.json", 
    "../data/edit_distance/Mistral-7B-Instruct-v0.2.json",
    "../data/edit_distance/Mistral-7B-Instruct-v0.3.json",
    
]

output1 = [
    "../data/avg_edit_distance/gemma-1.1-2b-it.json", 
    "../data/avg_edit_distance/gemma-1.1-7b-it.json", 
    "../data/avg_edit_distance/gpt-3.5-turbo-instruct-0914.json", 
    "../data/avg_edit_distance/Meta-Llama-3-8B-Instruct.json", 
    "../data/avg_edit_distance/Phi-3-mini-4k-instruct.json", 
    "../data/avg_edit_distance/Phi-3-mini-128k-instruct.json", 
    "../data/avg_edit_distance/Mistral-7B-Instruct-v0.2.json",
    "../data/avg_edit_distance/Mistral-7B-Instruct-v0.3.json",
    
]

models = [
    "../../context-generation/data/data_generated/google_gemma_2b/gemma-1.1-2b-it_1.", 
    "../../context-generation/data/data_generated/google_gemma_7b/gemma-1.1-7b-it_1.", 
    "../../context-generation/data/data_generated/gpt_3.5/gpt-3.5-turbo-instruct-0914_1.", 
    "../../context-generation/data/data_generated/meta_llama_8b/Meta-Llama-3-8B-Instruct_1.", 
    "../../context-generation/data/data_generated/microsoft_phi_4k/Phi-3-mini-4k-instruct_1.", 
    "../../context-generation/data/data_generated/microsoft_phi_128k/Phi-3-mini-128k-instruct_1.", 
    "../../context-generation/data/data_generated/mistral_7b/Mistral-7B-Instruct-v0.2_1.",
    "../../context-generation/data/data_generated/mistral_7b_v3/Mistral-7B-Instruct-v0.3_1."
    
] 

for idx, model in tqdm(enumerate(models)):
    edit_distance_scores = {}
    average_edit_distance_scores = {}
    for k in range(10):
        df = pd.read_csv(f"{model}{k}.csv")
        # print(df)
        scores = []
        for j in df['index']:
            edit_distance_sum = 0
            count =0
            for a in range(1, 11):
                try:
                    edit_distance = get_edit_distance( df[f"context_points"][int(j)].split(' ') , df[f"Column_{int(a)}"][int(j)].replace('\n' , '').split(' '))
                    count = count+1
                except:
                    edit_distance =0
                edit_distance_sum += edit_distance
            avg_edit_dist_per_datapoint = edit_distance_sum / count
            scores.append(avg_edit_dist_per_datapoint)
        edit_distance_scores[f'1.{k}'] = scores
        average_edit_distance_scores[f'1.{k}'] = sum(scores) / len(scores)
    
    with open(output[idx], 'w') as json_file:
        json.dump(edit_distance_scores, json_file)
    with open(output1[idx], 'w') as json_file:
        json.dump(average_edit_distance_scores, json_file)
   



