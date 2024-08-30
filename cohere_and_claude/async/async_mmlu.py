import asyncio
from io import StringIO
import aiofiles
import glob
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import os

# Assuming 'anthropic' library and API setup are similar to this
from anthropic import Anthropic

client = Anthropic(
    api_key="",
)
async def process_file(file_path, executor):
    lang = file_path.split('/')[-2]
    # Use aiofiles to asynchronously open and read the file
    async with aiofiles.open(file_path, mode='r') as file:
        content = await file.read()
    # Use StringIO to simulate a file object from a string of content
    df = pd.read_csv(StringIO(content), sep='\t')

    if 'opus' not in df.columns:
        df['opus'] = ''
    if 'prompt' not in df.columns:
        df['prompt'] = ''

    tasks = [process_row(df, idx, row, executor) for idx, row in df.iterrows()]
    await asyncio.gather(*tasks)

    if 'llm' in df.columns:
        df.drop(columns=['llm'], inplace=True)

    output_file = f"/Users/emmazhuang/Documents/Codes/Masakhane/afrimmlu_translate_test/opus/{lang}.csv"
    async with aiofiles.open(output_file, mode='w') as f:
        await f.write(df.to_csv(sep='\t', index=False))

async def process_row(df, idx, row, executor):
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

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        executor,
        lambda: client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0,
            system=sys,
            messages=mes
        )
    )
    df.at[idx, 'opus'] = response.content
    df.at[idx, 'prompt'] = mes


async def main():
    files = glob.glob('/Users/emmazhuang/Documents/Codes/Masakhane/afrimmlu_translate_test/*/test.tsv', recursive=True)
    print(files)
    executor = ThreadPoolExecutor(max_workers=10)

    tasks = [process_file(file, executor) for file in files]
    await asyncio.gather(*tasks)

    executor.shutdown()

if __name__ == '__main__':
    asyncio.run(main())