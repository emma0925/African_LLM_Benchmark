import cohere
import glob
import pandas as pd

co = cohere.Client(api_key="")

files = glob.glob('/Users/emmazhuang/Documents/Codes/Masakhane/afrimmlu/*/test.tsv', recursive=True)
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

        subject, question, choices = row['subject'], row['question'], row['choices']
        # Remove the surrounding brackets and split the string on commas
        choices_clean = choices.strip("[]")
        choices_list = [choice.strip().strip("'\"") for choice in choices_clean.split(',')]
        
        sys = "You are a highly knowledgeable and intelligent artificial intelligence "f"model answers multiple-choice questions about {subject}. Return the correct option only"

        mes=f"{question}\n A){choices_list[0]} B){choices_list[1]} C){choices_list[2]} D){choices_list[3]}"   
        response = co.chat(
            model="command-r",
            preamble=sys,    
            message=mes
        )
        
        # Update the DataFrame with the response text and prompt
        df.at[idx, 'command_r'] = response.text
        df.at[idx, 'prompt'] = 'presemble: '+sys +'\n' + 'user: '+mes
        
        print(response.text)
    
    # Drop the 'llm' column if it exists
    if 'llm' in df.columns:
        df = df.drop(columns=['llm'])
    
    output_file = "/Users/emmazhuang/Documents/Codes/Masakhane/afrimmlu/command_r/"+lang+'.csv'
    # Save the updated DataFrame back to the file
    df.to_csv(output_file, sep='\t', index=False)
