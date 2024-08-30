import pandas as pd

file = '/Users/emmazhuang/Documents/Codes/Masakhane/mmlu-wahili.tsv'

df = pd.read_csv(file, on_bad_lines='warn', delimiter='\t')
df['choices'] = df[['choice_A', 'choice_B', 'choice_C', 'choice_D']].values.tolist()
df = df.drop(columns=['q_id', 'split', 'translate_question', 'translate_choice_A', 'translate_choice_D', 'translate_choice_B', 'translate_choice_C', 'translate_choice_D', 'choice_A', 'choice_B', 'choice_C'])

df.to_csv('/Users/emmazhuang/Documents/Codes/Masakhane/mmlu-swahili.tsv', sep='\t', index=False)