import cohere
import glob
import pandas as pd
import anthropic

client = anthropic.Anthropic(
    api_key="",
)



files = glob.glob('/Users/emmazhuang/Documents/Codes/Masakhane/afrimmlu/*/*.tsv', recursive=True)
print(files)
for file in files:
    lang = file.split('/')[-2]
    print(lang)
    df = pd.read_csv(file, sep='\t')
    
    # Ensure the 'opus+' and 'prompt' columns exist
    if 'opus' not in df.columns:
        df['opus'] = ''
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

        sub_mes=f"{question}\n A){choices_list[0]} B){choices_list[1]} C){choices_list[2]} D){choices_list[3]}"  
        mes=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": sub_mes
                    }
                ]
            }
        ]
        
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0,
            system=sys,
            messages=mes
        )
        
        # Update the DataFrame with the response text and prompt
        df.at[idx, 'opus'] = response.content
        df.at[idx, 'prompt'] = mes
        
        print(response.content)
    
    # Drop the 'llm' column if it exists
    if 'llm' in df.columns:
        df = df.drop(columns=['llm'])
    
    output_file = "/Users/emmazhuang/Documents/Codes/Masakhane/afrimmlu/opus/"+lang+'.csv'
    # Save the updated DataFrame back to the file
    df.to_csv(output_file, sep='\t', index=False)

