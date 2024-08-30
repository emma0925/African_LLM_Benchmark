import glob
import os
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from datasets import load_metric
from evaluate import load
from datasets import Dataset, Value, Features


import pandas as pd
import re


def verbalizer(text):
    # Find all numbers in the text, including those in decimal format
    numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', str(text).replace('$', '').replace(' ', ''))
    # numbers = numbers.replace(',', '')
    try:
    # Remove commas and convert to integers, taking the last number if multiple
        if numbers:
            cleaned_number = numbers[-1].replace(',', '')
            if cleaned_number.lower() == 'inf' or cleaned_number.lower() == '-inf':
                return -1 
            else:
                last_number = int(float(cleaned_number))
                return last_number
        else:
            # If no number is found, return None or raise an error as per your application's needs
            return '-1'
    except OverflowError:
        return -1 
    except ValueError:
        return -1

def to_numeric(value):
    try:
        if isinstance(value, str):
            # If it's a string, remove commas and convert to float
            return float(value.replace(',', ''))
    except ValueError:
        # Return a default value in case of conversion failure
        return -1

def list_of_dicts_to_dict_of_lists(list_of_dicts):
    # Initialize an empty dict with lists for each key
    dict_of_lists = {k: [] for k in list_of_dicts[0]}
    for item in list_of_dicts:
        for k, v in item.items():
            dict_of_lists[k].append(v)
    return dict_of_lists


def calculate_exact_match(directory):
    files = glob.glob(os.path.join(directory, '*')) 
    print(files)
    exact={}
    for file in files:
        if os.path.isfile(file):
            # print(file)
            try:
                df = pd.read_csv(file, delimiter=',')
            except:
                df = pd.read_csv(file, delimiter='\t')
            if 'answer' in df.columns and 'command_r+' in df.columns:
                df['verbalized'] = df['command_r+'].apply(verbalizer)
                # df['answer'] = df['answer'].apply(to_numeric).fillna(-1).astype(int)
                df['verbalized'] = df['verbalized'].astype(float).fillna(-1).astype(int)

                # Replace NaN
                df['answer'].fillna('empty')
                df['command_r+'].fillna('empty')
                df['verbalized'].fillna('empty')
                output_directory = os.path.join(directory, 'verbalized')
                if os.path.isdir(output_directory) == False:
                    os.mkdir(output_directory)
                df.to_csv(os.path.join(output_directory, os.path.basename(file)), sep='\t', index=False)
                # le = LabelEncoder()
                # # Fit LabelEncoder on all possible labels from both 'label' and 'gpt_4'
                # le.fit(pd.concat([df['answer'], df['verbalized']]))
                # # Transform 'label' and 'gpt_4' to encoded numeric labels
                # y_true = le.transform(df['answer'])
                # y_pred = le.transform(df['verbalized'])
                
                # f1 = f1_score(y_true, y_pred, average='weighted')
                # filename = os.path.basename(file)
                # exact[filename] = f1
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

def save_exact_to_csv(exact, output_csv_path, name):
    sorted_exact = dict(sorted(exact.items()))

    # Create a DataFrame from the sorted dictionary, orienting the data so it forms a single row
    exact_df = pd.DataFrame([sorted_exact])

    # Insert the 'Name' column at the first position if it does not already exist
    if 'Name' not in exact_df.columns:
        exact_df.insert(0, 'Name', name)
    else:
        exact_df['Name'] = name

    # Check if the file exists and determine if headers should be written
    file_exists = os.path.exists(output_csv_path)
    header = not file_exists  # Write header only if file does not exist

    # Append to the CSV file or create it if it does not exist
    exact_df.to_csv(output_csv_path, mode='a', header=header, index=False)


if __name__ == "__main__":
    directory_path = '/Users/emmazhuang/Documents/Codes/Masakhane/results_mgsm_tt'
    sub_dir_order = []
    sub_dir = glob.glob(os.path.join(directory_path, '*/'))
    output_csv = '/Users/emmazhuang/Documents/Codes/Masakhane/scoring/score_result/exact_results_mgsm_tt.csv'
    for dir in sub_dir:
        print(dir)
        exact = dict(sorted(calculate_exact_match(dir).items()))
        print("1111111", exact)
        save_exact_to_csv(exact, output_csv, dir.split('/')[-2])
        sub_dir_order.append(dir)
    print(output_csv)