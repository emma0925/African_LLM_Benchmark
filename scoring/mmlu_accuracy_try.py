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
        print(1)
        return match.group(1).upper()

    # Match "a) china" format
    match = re.search(r'^([a-d])\)', text, re.IGNORECASE)
    if match:
        print(2)
        return match.group(1).upper()
    
    match = re.search(r'^([a-d]):', text, re.IGNORECASE)
    if match:
        print(3)
        return match.group(1).upper()
    
    match = re.search(r'^([a-d])\s', text, re.IGNORECASE)
    if match:
        print(4)
        return match.group(1).upper()
    
    match = re.search(r'^([a-d])\.', text, re.IGNORECASE)
    if match:
        print(5)
        return match.group(1).upper()
    
    match = re.search(r'\s([a-d])\.\s', text, re.IGNORECASE)
    if match:
        print(6)
        return match.group(1).upper()
    
    if match:
        print(8)
        return match.group(1).upper()
    
    # Match "<pad> A) 4t</s>" or similar formats where the letter may be followed by other characters
    match = re.search(r'<pad>\s*([A-D])\)', text, re.IGNORECASE)
    if match:
        print(9)
        return match.group(1).upper()

    match = re.search(r'Model answers:\s*([A-D])', text, re.IGNORECASE)
    if match:
        print(10)
        return match.group(1).upper()
    
    match = re.search(r'Correct answer:\s*([A-Da-d])[)]?', text, re.IGNORECASE)
    if match:
        print(11)
        return match.group(1).upper()

    match = re.search(r'correct answer:\s*([A-Da-d])[)]?', text, re.IGNORECASE)
    if match:
        print(12)
        return match.group(1).upper()
    
    match = re.search(r'Correct answer is:\s*([A-Da-d])[)]?', text, re.IGNORECASE)
    if match:
        print(13)
        return match.group(1).upper()

    match = re.search(r'correct answer is\s*([A-Da-d])[)]?', text, re.IGNORECASE)
    if match:
        print(14)
        return match.group(1).upper()
    
    match = re.search(r'answer is Option ([A-Da-d])[)]?', text, re.IGNORECASE)
    if match:
        print(15)
        return match.group(1).upper()

    text_2 = text.strip()
    # Get the last character of the string
    last_letter = text_2[-1]
    # Check if the last letter is between A-D or a-d
    if last_letter in 'ABCDabcd':
        return last_letter.upper()  
     
    # match = re.search(r'<pad>\s*(\d+)\s*<unk>', text, re.IGNORECASE)
    # if match:
    #     number = match.group(1)
    #     if isinstance(choices, list):
    #         for i, choice in enumerate(choices):
    #             if choice == number:
    #                 return ['A', 'B', 'C', 'D'][i]
    
    # match = re.search(r'Answer:\s*\*?\*?\s*([A-Da-d])[)]?', text, re.IGNORECASE)
    # if match:
    #     return match.group(1).upper()

    # if isinstance(choices, list):
    #     for i, choice in enumerate(choices):
    #         if choice in text:
    #             return ['A', 'B', 'C', 'D'][i]
    
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
    

    print('1111', a[0])
    print('2222', text)
    # Return 'unknown' if no match is found
    return 'unknown'

metric_gpt3 = {}
file = '/Users/emmazhuang/Documents/Codes/Masakhane/afrimmlu/command_r+/verbalized/eng_try.csv'

df = pd.read_csv(file, sep='\t')
if 'verbalized' in df.columns:
    df = df.drop(columns=['verbalized'])
print('command_r+: ', df['command_r+'])
print('choices: ', df['choices'])
df['verbalized'] = df.apply(lambda row: extract_letter_option(row['command_r+'], row['choices']), axis=1)
print(df['verbalized'])