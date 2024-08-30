import glob
import os
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from datasets import load_metric
from evaluate import load


import pandas as pd
import re

def verbalizer(text_block):
    # Check if the text_block starts and ends with the expected format
    if text_block.startswith("[TextBlock(text=") and text_block.endswith(")]"):
        # Extract the text between 'text=' and the first non-digit character or the next single quote
        start = text_block.find("text='") + 6
        end = text_block.find("'", start)
        number_text = text_block[start:end]
        # Find all sequences of digits, split on non-digit characters
        numbers = re.findall(r'\d+', number_text)
        if numbers:
            # Convert the first sequence of digits found to an integer
            return int(numbers[0])
        else:
            print(text_block)
            return -1
    else:
        print(text_block)
        return -1

def to_numeric(value):
    try:
        if isinstance(value, str):
            # If it's a string, remove commas and convert to float
            return float(value.replace(',', ''))
    except ValueError:
        # Return a default value in case of conversion failure
        return -1


def calculate_f1_scores(directory):
    files = glob.glob(os.path.join(directory, '*')) 
    print(files)
    exact={}
    for file in files:
        if os.path.isfile(file):
            print(file)
            df = pd.read_csv(file, delimiter='\t')
            print(df.columns)
            if 'answer' in df.columns and 'verbalized' in df.columns:
                # le = LabelEncoder()
                # # Fit LabelEncoder on all possible labels from both 'label' and 'gpt_4'
                # le.fit(pd.concat([df['answer'], df['verbalized']]))
                # # Transform 'label' and 'gpt_4' to encoded numeric labels
                # y_true = le.transform(df['answer'])
                # y_pred = le.transform(df['verbalized'])
                
                # f1 = f1_score(y_true, y_pred, average='weighted')
                # filename = os.path.basename(file)
                # f1_scores[filename] = f1
                ref_squad, pred_squad = [], []
                for index, row in df.iterrows():
                    pred_dict = str(row['verbalized'])
                    ref_dict =  str(row['answer'])
                    pred_squad.append(pred_dict)
                    ref_squad.append(ref_dict)

                exact_match = load("exact_match")
                print(pred_squad)
                print(ref_squad)
                results= exact_match.compute(predictions=pred_squad, references=ref_squad, ignore_case=True, ignore_punctuation=True)
                print("results:", results)
                score = (round(results["exact_match"]*100, 2))
                filename = os.path.basename(file)
                exact[filename] = score
                print(f"Exact for {file}: {score:.2f}") 
    
    return exact

def save_f1_scores_to_csv(f1_scores, output_csv_path):
    f1_scores_df = pd.DataFrame.from_dict(f1_scores, orient='index', columns=['F1 Score'])
    
    # Sort the DataFrame by its index (language codes) in ascending order
    sorted_df = f1_scores_df.sort_index(ascending=True).T
    sorted_df.to_csv(output_csv_path)

if __name__ == "__main__":
    directory = "/Users/emmazhuang/Documents/Codes/Masakhane/results_mgsm/gpt3.5-turbo/verbalized"  # Update this path
    output_csv = "f1_score_summary_gpt_3.csv"  # Define your output CSV file name
    f1_scores = calculate_f1_scores(directory)
    save_f1_scores_to_csv(f1_scores, output_csv)
    print(f"F1 score summary saved to {output_csv}")
    # print(verbalizer('''[TextBlock(text='18', type='text')]'''))