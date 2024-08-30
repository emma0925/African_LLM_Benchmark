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

df = pd.read_csv('/Users/emmazhuang/Documents/Codes/Masakhane/mmlu_tt_gpt/mmlu_tt_gpt3.5/swa.tsv', sep=',')
print(df.columns) 
if 'verbalized' in df.columns:
    df = df.drop(columns=['verbalized'])
df['verbalized'] = df.apply(lambda row: extract_letter_option(row['output'], row['choices']), axis=1)
output_file = "/Users/emmazhuang/Documents/Codes/Masakhane/mmlu_tt_gpt/mmlu_tt_gpt3.5/verbalized/swa.tsv"
# Save the updated DataFrame back to the file
df.to_csv(output_file, sep='\t', index=False)    
ac = accuracy_score(df['answer'], df['verbalized'])
print(round(ac*100, 2))
