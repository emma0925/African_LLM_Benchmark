import os
import asyncio
import pandas as pd
from aiohttp import ClientSession
from typing import List, Dict, Any
import logging
import aiolimiter
import argparse

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

async def fetch_relation(session: ClientSession, api_key: str, message: Dict[str, Any], limiter: aiolimiter.AsyncLimiter) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    url = "https://api.openai.com/v1/chat/completions"
    async with limiter:
        try:
            # logging.debug(f"Request URL: {url}")
            # logging.debug(f"Request Headers: {headers}")
            # logging.debug(f"Request Payload: {message}")
            
            response = await session.post(url, headers=headers, json=message)
            
            # logging.debug(f"Response Status: {response.status}")
            # logging.debug(f"Response Headers: {response.headers}")
            
            if response.status == 200:
                result = await response.json()
                # logging.debug(f"Response Body: {result}")
                if 'choices' in result and result['choices']:  # Check if 'choices' exist and is not empty
                    return result['choices'][0]['message']['content']
                else:
                    return "No response text available"
            else:
                # logging.debug(f"Error Response: {await response.text()}")
                return f"Error: Response code {response.status}"
        except Exception as e:
            # logging.debug(f"Exception occurred: {str(e)}")
            return f"Exception: {str(e)}"


async def generate_from_openai_chat_completion(
    messages_list: List[Dict[str, Any]],
    model_name: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    requests_per_minute: int
) -> List[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    limiter = aiolimiter.AsyncLimiter(requests_per_minute, 1)  # Limit: requests_per_minute requests per second
    async with ClientSession() as session:
        tasks = [fetch_relation(session, api_key, {
                    "model": model_name,
                    "messages": messages,  # Pass the whole list here
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p
                }, limiter) for messages in messages_list]  # Iterate over messages_list
        return await asyncio.gather(*tasks)

async def main(input_file: str, output_file: str, model_name: str):
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable must be set.")

    if(input_file == '/Users/emma.zhuang/dev/my_own_code/Masakhane/afrimgsm/eng.tsv'):
        df = pd.read_csv(input_file, delimiter=',',index_col = 0)
    else:
        df = pd.read_csv(input_file, delimiter='\t')
    all_input_messages = []

    for _, row in df.iterrows():
        question = row['question']
        messages = [
            {"role": "user", "content": 'Answer only and the answer should only contains integers. \n' + str(question) + "\n"}
        ]
        all_input_messages.append(messages)

    responses = await generate_from_openai_chat_completion(
        all_input_messages, model_name, 0.3, 500, 1.0, 300)
    df['prompt'] = all_input_messages
    df['gpt_4o'] = responses
    if 'llm' in df.columns:
        df = df.drop(columns=['llm'])
    df.to_csv(output_file, index=False)
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and output files for OpenAI chat completions.")
    parser.add_argument("input_file", type=str, help="Input CSV file containing premise and hypothesis columns")
    parser.add_argument("output_file", type=str, help="Output CSV file to save the results")
    parser.add_argument("--model_name", type=str, default="gpt-4-turbo", help="Name of the GPT model to use (default: gpt-3.5-turbo)")
    args = parser.parse_args()

    asyncio.run(main(args.input_file, args.output_file, args.model_name))