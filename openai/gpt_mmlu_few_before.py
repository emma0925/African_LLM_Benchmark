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
    api_key = ''
    limiter = aiolimiter.AsyncLimiter(requests_per_minute, 1) 
    async with ClientSession() as session:
        tasks = [fetch_relation(session, api_key, {
                    "model": model_name,
                    "messages": messages,  
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p
                }, limiter) for messages in messages_list]
        return await asyncio.gather(*tasks)

async def main(learn_file: str, test_file: str, output_file: str, model_name: str):
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable must be set.")

    df_test = pd.read_csv(test_file, delimiter='\t')
    df_learn = pd.read_csv(learn_file, delimiter='\t')
    all_input_messages = []
    
    learning_examples = {}
    
    for _, row in df_learn.iterrows():
        subject = row['subject']
        question = row['question']
        choices = row['choices']
        answer = row['answer']
        
        choices_clean = choices.strip("[]")
        choices_list = [choice.strip().strip("'\"") for choice in choices_clean.split(',')]
        
        formatted_question = f"{question}\nA) {choices_list[0]} B) {choices_list[1]} C) {choices_list[2]} D) {choices_list[3]}"
        
        learning_message = {
            "role": "assistant",
            "content": f"{formatted_question}\nCorrect Answer: {answer}"
        }
        
        if subject not in learning_examples:
            learning_examples[subject] = []
        learning_examples[subject].append(learning_message)
    
    all_input_messages = []
    
    for subject in learning_examples:
        all_input_messages.append(learning_examples[subject])
    
    for _, row in df_test.iterrows():
        subject = row['subject']
        question = row['question']
        choices = row['choices']
        choices_clean = choices.strip("[]")
        choices_list = [choice.strip().strip("'\"") for choice in choices_clean.split(',')]
        
        test_message = {
            "role": "user",
            "content": f"{question}\nA) {choices_list[0]} B) {choices_list[1]} C) {choices_list[2]} D) {choices_list[3]}"
        }
        
        if subject in learning_examples:
            all_input_messages.append([test_message])

    responses = await generate_from_openai_chat_completion(
        all_input_messages, model_name, 0.3, 500, 1.0, 300)
    
    # if len(responses) != len(df_test):
    #     print("Mismatch in the number of responses and test entries.")
    #     # For debugging: you might want to inspect a few responses
    #     print(responses[:5])  # Print first 5 responses to inspect
    #     # Handle mismatch, for example by truncating the list or raising an error
    #     responses = responses[:len(df_test)]  # Truncate if longer
    # else:
    #     df_test['output'] = responses

    print(all_input_messages[0])
    print(responses[0])
    df_test['output'] = responses[5:]
    df_test.to_csv(output_file, index=False)
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and output files for OpenAI chat completions.")
    parser.add_argument("learn_file", type=str, help="Input CSV file containing premise and hypothesis columns for learn")
    parser.add_argument("test_file", type=str, help="Input CSV file containing premise and hypothesis columns for test")
    parser.add_argument("output_file", type=str, help="Output CSV file to save the results")
    parser.add_argument("--model_name", type=str, default="gpt-4-turbo ", help="Name of the GPT model to use (default: gpt-3.5-turbo)")
    args = parser.parse_args()

    asyncio.run(main(args.learn_file, args.test_file, args.output_file, args.model_name))