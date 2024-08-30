import cohere
import glob
import pandas as pd
import anthropic

client = anthropic.Anthropic(
    api_key="",
)



files = glob.glob('/Users/emmazhuang/Documents/Codes/Masakhane/afrixnli/*/*.tsv', recursive=True)
# print(files)
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
        premise, hypothesis = row['premise'], row['hypothesis']

        sub_mes=f"{{premise}}\nQuestion: {{hypothesis}} True, False, or Neither?\nAnswer:"
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
            messages=mes
        )
        
        # Update the DataFrame with the response text and prompt
        df.at[idx, 'opus'] = response.content
        df.at[idx, 'prompt'] = mes
        
        # print(response.content)
    
    # Drop the 'llm' column if it exists
    if 'llm' in df.columns:
        df = df.drop(columns=['llm'])
    
    output_file = "/Users/emmazhuang/Documents/Codes/Masakhane/afrixnli/opus_new_prompt/"+lang+'.csv'
    # Save the updated DataFrame back to the file
    df.to_csv(output_file, sep='\t', index=False)

