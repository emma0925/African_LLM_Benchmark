import glob
import os
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import re
from evaluate import load

def verbalize_label(label):

    # Use regular expressions to extract 'Contradiction', 'Entailment', or 'Neutral' from the label
    match = re.findall(r'(contradiction|entailment|neutral)', label.lower())
    if len(match) == 1:
        return match[0]
    elif len(match) >= 2:
        # Split the text on the <eos> tag
        match = re.search(r'Answer:\s*(\w+)', label)
        # Check if a match is found
        if match:
            # Retrieve the matched word and convert it to lowercase
            found_label = match.group(1).lower()
            # Check if the found label is one of the expected options
            if found_label in ['contradiction', 'entailment', 'neutral']:
                return found_label
        
        parts = label.split('<eos>')
        # Focus on the second last part, as it should contain the label just before the last <eos>
        if len(parts) > 1:
            last_label_segment = parts[-2]  # Get the segment right before the last <eos>
            # Use regular expressions to extract 'Contradiction', 'Entailment', or 'Neutral' from this segment
            match = re.findall(r'(contradiction|entailment|neutral)', last_label_segment.lower())
            if match:
                return match[0]
            else:
                return 'Unknown'
        match = re.findall(r'(contradiction|entailment|neutral)', label.lower())
        return match[-1]
    else:
        return 'Unknown'


def calculate_accuracy(directory):
    xnli_metric = load("xnli")
    accuracy_scores = {}
    f1_scores = {}
    
    files = glob.glob(os.path.join(directory, '*'))
    print(files)
    for file in files:
        if os.path.isfile(file):
            print(file)
            # try:
            #     df = pd.read_csv(file)
            # except:
            df = pd.read_csv(file, sep=',')
            print("Column names:", list(df.columns))
            # Preprocess and encode labels
            df['label'] = df['label'].str.lower().fillna('empty')
            df['output'] = df['output'].str.lower().fillna('empty')
            df['verbalized'] = df['output'].apply(verbalize_label).str.lower().fillna('empty')
            output_directory = os.path.join(directory, 'verbalized')
            df.to_csv(os.path.join(output_directory, os.path.basename(file)), sep='\t', index=False)
            
            le = LabelEncoder()
            # Fit and transform labels
            le.fit(pd.concat([df['label'], df['verbalized']]))
            y_true = le.transform(df['label'])
            y_pred = le.transform(df['verbalized'])

            # Prepare data for xnli_metric
            pred_squad = y_pred.tolist()   # Predictions as integer list
            ref_squad = y_true.tolist()    # References as integer list

            # Compute results using the xnli metric
            results = xnli_metric.compute(predictions=pred_squad, references=ref_squad)
            accuracy = results['accuracy']
            accuracy_scores[os.path.basename(file)] = accuracy*100
            print(f"Accuracy for {file}: {accuracy*100:.2f}%")

    return accuracy_scores

def save_f1_scores_to_csv(f1_scores, output_csv_path, name):
    sorted_f1_scores = dict(sorted(f1_scores.items()))

    # Create a DataFrame from the sorted dictionary, orienting the data so it forms a single row
    f1_scores_df = pd.DataFrame([sorted_f1_scores])

    # Insert the 'Name' column at the first position if it does not already exist
    if 'Name' not in f1_scores_df.columns:
        f1_scores_df.insert(0, 'Name', name)
    else:
        f1_scores_df['Name'] = name

    # Check if the file exists and determine if headers should be written
    file_exists = os.path.exists(output_csv_path)
    header = not file_exists  # Write header only if file does not exist

    # Append to the CSV file or create it if it does not exist
    f1_scores_df.to_csv(output_csv_path, mode='a', header=header, index=False)


if __name__ == "__main__":
    directory_path = '/Users/emmazhuang/Documents/Codes/Masakhane/results_nli_tt'
    sub_dir_order = []
    sub_dir = glob.glob(os.path.join(directory_path, '*/'))
    sub_dir= ['/Users/emmazhuang/Documents/Codes/Masakhane/nli_gpt/nli_new_prompt_gpt3']
    output_csv = '/Users/emmazhuang/Documents/Codes/Masakhane/scoring/score_result/accuracy_results_nli_gpt3.csv'
    for dir in sub_dir:
        print(dir)
        f1_scores = dict(sorted(calculate_accuracy(dir).items()))
        print("1111111", f1_scores)
        save_f1_scores_to_csv(f1_scores, output_csv, dir.split('/')[-2])
        sub_dir_order.append(dir)
    print(output_csv)

    # directory = "/Users/emma.zhuang/dev/my_own_code/Masakhane/results_nli_llama3_translate_test"  # Update this path
    # output_csv = "f1_score_summary_david_xni.csv"  # Define your output CSV file name
    # f1_scores = calculate_f1_scores(directory)
    # 
    # print(f"F1 score summary saved to {output_csv}")