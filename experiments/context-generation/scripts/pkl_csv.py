
import pickle
import pandas as pd

csv_file_path = '../data/COBIAS.csv'
df2 = pd.read_csv(csv_file_path)

column_names = [f'Column_{i+1}' for i in range(10)]
df = pd.DataFrame(columns=['index' , 'context_points'] + column_names)

for j in range(10):
    file_path = f"../data/generations/generations_microsoft/Phi-3-mini-128k-instruct_1.3_{j}.pkl"
    with open(file_path, 'rb') as file:
        data= pickle.load(file)
    for key in data.keys():
        row = [key ,df2['context_points'][key]] + data[key]
        df = pd.concat([df, pd.DataFrame([row], columns=['index' , 'context_points'] + column_names)], ignore_index=True)

output_filename = f"../data/data_generated/microsoft_phi_128k/Phi-3-mini-128k-instruct_1.3.csv"
df.to_csv(output_filename, index=False)
print(f"CSV file '{output_filename}' created successfully.")