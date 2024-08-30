import ast
import glob
import os
import re
import pandas as pd
from sklearn.metrics import accuracy_score

def extract_letter_option(text,choices):
    choices = ast.literal_eval(choices)

    if pd.isnull(text):
        return 'unknown'
    
    # Match "the correct option is: A" format
    match = re.search(r'the correct option is[:\s]*([A-D])[).]', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Match "a) china" format
    match = re.search(r'^([a-d])\)', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    match = re.search(r'^([a-d]):', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    match = re.search(r'^([a-d])\s', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    match = re.search(r'^([a-d])\.', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    match = re.search(r'\s([a-d])\.\s', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r'<pad>\s*([A-D])\s*</s>', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Match "<pad> A) 4t</s>" or similar formats where the letter may be followed by other characters
    match = re.search(r'<pad>\s*([A-D])\)', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r'Model answers:\s*([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    match = re.search(r'text=\'([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    match = re.search(r'Correct answer:\s*([A-Da-d])[)]?', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r'correct answer:\s*([A-Da-d])[)]?', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    match = re.search(r'Correct answer is:\s*([A-Da-d])[)]?', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r'correct answer is\s*([A-Da-d])[)]?', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    match = re.search(r'answer is Option ([A-Da-d])[)]?', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    text_2 = text.strip()
    # Get the last character of the string
    last_letter = text_2[-1]
    # Check if the last letter is between A-D or a-d
    if last_letter in 'ABCDabcd':
        return last_letter.upper()  
     
    match = re.search(r'<pad>\s*(\d+)\s*<unk>', text, re.IGNORECASE)
    if match:
        number = match.group(1)
        if isinstance(choices, list):
            for i, choice in enumerate(choices):
                if choice == number:
                    return ['A', 'B', 'C', 'D'][i]
    
    match = re.search(r'Answer:\s*\*?\*?\s*([A-Da-d])[)]?', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    if isinstance(choices, list):
        for i, choice in enumerate(choices):
            if str(choice) in text:
                return ['A', 'B', 'C', 'D'][i]
    
    # Regex to find letter options (e.g., 'A)') in each part
    parts = text.split('<eos>')
    options = []
    for part in parts:
        matches = re.findall(r'\b([A-D])\)', part)
        if matches:
            options.extend(matches)
    
    # Return the last letter option found, if any
    if options:
        return options[-1]
    
    match = re.search(r'\s([a-d])\s', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    print(text)
    # Return 'unknown' if no match is found
    return 'unknown'

metric_gpt3 = {}
directory_path = '/Users/emmazhuang/Documents/Codes/Masakhane/afrimmlu/gpt_4o_few_shots_25'
files = glob.glob(os.path.join(directory_path, '*')) 
# Iterate over each file in the directory
for file in files:
    if os.path.isfile(file):
        lang = file.split("/")[-1].split('.')[0]
        df = pd.read_csv(file, delimiter=',')

        if 'verbalized' in df.columns:
            df = df.drop(columns=['verbalized'])

        print(df.columns)

        df['verbalized'] = df.apply(lambda row: extract_letter_option(row['output'], row['choices']), axis=1)

        # # Debugging output
        # print(df['answer'].dtype, df['verbalized'].dtype)
        # print(df['answer'].head(), df['verbalized'].head())  # Check sample data

        # Standardize types
        if df['answer'].dtype != df['verbalized'].dtype:
            df['answer'] = df['answer'].astype(str)
            df['verbalized'] = df['verbalized'].astype(str)

        # Calculate accuracy
        try:
            ac = accuracy_score(df['answer'], df['verbalized'])
            metric_gpt3[lang] = round(ac*100, 2)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

# Save results
sorted_metric = {k: metric_gpt3[k] for k in sorted(metric_gpt3)}
df_metric = pd.DataFrame([sorted_metric])
df_metric.to_csv('/Users/emmazhuang/Documents/Codes/Masakhane/scoring/score_result/accuracy_mmlu_few_4o_new.csv', index=False)