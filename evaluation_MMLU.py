import glob
import os
import re
import pandas as pd
from sklearn.metrics import accuracy_score

def extract_letter_option(text,choices):
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

    if isinstance(choices, list):
        for i, choice in enumerate(choices):
            if choice in text:
                return ['A', 'B', 'C', 'D'][i]


    # Return 'unknown' if no match is found
    return 'unknown'

metric_gpt3 = {}
directory_path = '/Users/emma.zhuang/dev/my_own_code/Masakhane/africanmmlu_result'
files = glob.glob(os.path.join(directory_path, '*')) 
for file in files:
    lang = file.split("/")[-1].split('.')[0]
    df = pd.read_csv(file, sep='\t')
    print(df['choices'])
    df['gpt-3.5_verbalized'] = df.apply(lambda row: extract_letter_option(row['gpt-3.5'], row['choices']), axis=1)
    df.to_csv(file, sep='\t', index=False)
    ac = accuracy_score(df['answer'], df['gpt-3.5_verbalized'])
    metric_gpt3[lang] = round(ac*100, 2)
    print(metric_gpt3)