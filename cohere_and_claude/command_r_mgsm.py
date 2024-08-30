import cohere
import glob
import pandas as pd

co = cohere.Client(api_key="")

files = glob.glob('/Users/emmazhuang/Documents/Codes/Masakhane/afrimgsm/*/*.tsv', recursive=True)
print(files)
for file in files:
    lang = file.split('/')[-2]
    print(lang)
    df = pd.read_csv(file, sep='\t')
    
    # Ensure the 'command_r+' and 'prompt' columns exist
    if 'command_r' not in df.columns:
        df['command_r'] = ''
    if 'prompt' not in df.columns:
        df['prompt'] = ''
    
    for idx, row in df.iterrows():
        print(row)
        question = row['question']
        print(question)
        mes = 'Answer only and the answer should only contain integers. \n' + str(question) + "\n"
        
        response = co.chat(
            model="command-r",
            message=mes
        )
        
        # Update the DataFrame with the response text and prompt
        df.at[idx, 'command_r'] = response.text
        df.at[idx, 'prompt'] = mes
        
        print(response.text)
    
    # Drop the 'llm' column if it exists
    if 'llm' in df.columns:
        df = df.drop(columns=['llm'])
    
    output_file = "/Users/emmazhuang/Documents/Codes/Masakhane/afrimgsm/command_R/"+lang+'.csv'
    # Save the updated DataFrame back to the file
    df.to_csv(output_file, sep='\t', index=False)
