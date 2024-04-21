import os
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

def calculate_f1_scores(directory):
    f1_scores = {}

    for subdir, _, _ in os.walk(directory):
        csv_path = os.path.join(subdir, 'output.csv')
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            if 'label' in df.columns and 'relation' in df.columns:
                # Convert 'label' and 'relation' to lowercase to ignore capitalization
                df['label'] = df['label'].str.lower()
                df['relation'] = df['relation'].str.lower()

                # Drop rows where either 'label' or 'relation' is NaN
                df.dropna(subset=['label', 'relation'], inplace=True)
                
                # Initialize LabelEncoder
                le = LabelEncoder()
                # Fit LabelEncoder on all possible labels from both 'label' and 'relation'
                le.fit(pd.concat([df['label'], df['relation']]))
                # Transform 'label' and 'relation' to encoded numeric labels
                y_true = le.transform(df['label'])
                y_pred = le.transform(df['relation'])
                
                # Calculate F1 score, using 'weighted' for multi-class scenario
                f1 = f1_score(y_true, y_pred, average='weighted')
                
                subdir_name = os.path.basename(subdir)
                f1_scores[subdir_name] = f1
                print(f"F1 Score for {subdir_name}: {f1:.2f}")  # Print F1 score for each category
    
    return f1_scores

def save_f1_scores_to_csv(f1_scores, output_csv_path):
    f1_scores_df = pd.DataFrame.from_dict(f1_scores, orient='index', columns=['gpt-3.5-turbo F1 Score'])
    f1_scores_df.to_csv(output_csv_path)

if __name__ == "__main__":
    directory = "/Users/emma.zhuang/dev/my_own_code/Masakhane"  # Update this path
    output_csv = "f1_score_summary.csv"  # Define your output CSV file name
    f1_scores = calculate_f1_scores(directory)
    save_f1_scores_to_csv(f1_scores, output_csv)
    print(f"F1 score summary saved to {output_csv}")
