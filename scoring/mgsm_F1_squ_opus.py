import glob
import os
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from datasets import load_metric


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
    f1_scores={}
    for file in files:
        if os.path.isfile(file):
            print(file)
            df = pd.read_csv(file, delimiter='\t')
            print(df.columns)
            if 'answer' in df.columns and 'opus' in df.columns:
                print(df['opus'])
                df['verbalized'] = df['opus'].apply(verbalizer)
                df['answer'] = df['answer'].apply(to_numeric).fillna(-1).astype(int)
                df['verbalized'] = df['verbalized'].astype(float).fillna(-1).astype(int)

                # Replace NaN
                df['answer'].fillna('empty', inplace=True)
                df['opus'].fillna('empty', inplace=True)
                df['verbalized'].fillna('empty', inplace=True)
                output_directory = os.path.join(directory, 'verbalized')
                df.to_csv(os.path.join(output_directory, os.path.basename(file)), sep='\t', index=False)
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
                    pred_dict = {'prediction_text': row['verbalized'], 'id': str(index)}
                    ref_dict = {'answers': {'answer_start': [1], 'text': [row['answer']]}, 'id': str(index)}
                    pred_squad.append(pred_dict)
                    ref_squad.append(ref_dict)

                squad_metric = load_metric("squad")
                results_squad = squad_metric.compute(predictions=pred_squad, references=ref_squad)
                score = round(results_squad['f1'], 2)
                filename = os.path.basename(file)
                f1_scores[filename] = score
                print(f"F1 Score for {file}: {score:.2f}") 
    
    return f1_scores

def save_f1_scores_to_csv(f1_scores, output_csv_path):
    f1_scores_df = pd.DataFrame.from_dict(f1_scores, orient='index', columns=['F1 Score'])
    
    # Sort the DataFrame by its index (language codes) in ascending order
    sorted_df = f1_scores_df.sort_index(ascending=True).T
    sorted_df.to_csv(output_csv_path)

if __name__ == "__main__":
    directory = "/Users/emmazhuang/Documents/Codes/Masakhane/afrimgsm_translate_test/opus"  # Update this path
    output_csv = "f1_score_summary_mgsm_tt_opus.csv"  # Define your output CSV file name
    f1_scores = calculate_f1_scores(directory)
    save_f1_scores_to_csv(f1_scores, output_csv)
    print(f"F1 score summary saved to {output_csv}")
    # print(verbalizer('''[TextBlock(text='18', type='text')]'''))