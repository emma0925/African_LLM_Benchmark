import glob
import pandas as pd
import ast
import ast
import os
import asyncio
from typing import List, Dict, Any
import logging
import aiolimiter
import argparse
from aiohttp import ClientSession

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


def mmlu():
    files = glob.glob(f'/Users/emma.zhuang/dev/my_own_code/Masakhane/africammlu_2/*.tsv', recursive=True)
    for file in files:
        lang = file.split('/')[-1].split('.')[0]
        print(lang)
        if lang != 'wol':
            df = pd.read_csv(file, sep='\t')
            subjects = df['subject'].unique()
            new_df = pd.DataFrame()
            for subject in subjects:
                print(subject)
                sub_df = df[df['subject'] == subject]
                prompt_query = "Task Description: You are a highly knowledgeable and intelligent artificial intelligence " \
                               f"model answers multiple-choice questions about {subject}. Return the correct option only"
                all_input_messages = []
                for index in range(sub_df.shape[0]):  # df.shape[0]
                    question = sub_df['question'].iloc[index]
                    choices = sub_df['choices'].iloc[index]
                    try:
                        choices = [value.replace('\n', '') for value in ast.literal_eval(choices)]
                        print(choices)
                    except AttributeError:
                        print(0)
                        choices = ast.literal_eval(choices)
                        print(choices)
                    except ValueError:
                        stripped_string = choices.strip('[]')
                        split_times = [time.strip() + ')' for time in stripped_string.split('), ') if time.strip()]
                        choices = split_times

                    formatted_choices = "\n".join([f"{chr(65+i)}) {value}" for i, value in enumerate(choices)])

                    context = f"{question}\n{formatted_choices}"

                    message = prompt_query + '\n\n' + context
                    input_mes = [{"role": "system",
                                  "content": "You are a highly knowledgeable and intelligent artificial intelligence "
                                             f"model answers multiple-choice questions about {subject}"},
                                 {"role": "user", "content": message}]
                    all_input_messages.append(input_mes)

                responses = asyncio.run(
                    generate_from_openai_chat_completion(all_input_messages, "gpt-3.5-turbo", 0.3, 500, 1.0, 500))

                completions = []
                for completion_text in responses:
                    completions.append(completion_text.lower())

                sub_df['gpt-3.5'] = completions
                new_df = new_df._append(sub_df)
            new_df.to_csv("./africanmmlu_result/" + lang, sep='\t')

mmlu()