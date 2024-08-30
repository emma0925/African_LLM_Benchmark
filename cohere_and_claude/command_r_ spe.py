import cohere
import glob
import pandas as pd

co = cohere.Client(api_key="")

file = '/Users/emmazhuang/Documents/Codes/Masakhane/cohere/mgsm/amh.tsv'
lang = file.split('/')[-1].split('.')[0]
print(lang)



with open(file, 'r') as f:
    header = f.readline().strip().split('\t')
    print("Header:", header)



df = pd.read_csv(file, sep='\t', skiprows=1, names=header)
print("Columns in the DataFrame:", df.columns)

# Drop the 'llm' column if it exists
if 'llm' in df.columns:
    df = df.drop(columns=['llm'])

if 'command_r+' in df.columns:
    df = df.drop(columns=['command_r'])

if 'prompt' in df.columns:
    df = df.drop(columns=['prompt'])

# Ensure the 'command_r+' and 'prompt' columns exist
if 'command_r' not in df.columns:
    df['command_r'] = ''
if 'prompt' not in df.columns:
    df['prompt'] = ''

# Check if 'question' column exists
if 'question' in df.columns:
    for idx, row in df.iterrows():
        print(row)
        question = row['question']
        print(question)
        mes = 'Answer only and the answer should only contain integers. \n' + str(question) + "\n"
        
        response = co.chat(
            model="command-r",
            message=mes
        )
        
        # Update the DataFrame with the response text and prompt
        df.at[idx, 'command_r'] = response.text
        df.at[idx, 'prompt'] = mes
        
        print(response.text)
else:
    print("Error: 'question' column not found in the DataFrame")



# Save the updated DataFrame back to the file
df.to_csv(file, sep='\t', index=False)
