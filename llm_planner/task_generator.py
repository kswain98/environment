import os
import time
import argparse
from groq import Groq

def get_llm_response(client, prompt, model):
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
    )
    return chat_completion.choices[0].message.content

def main(api_key, prompt, model, interval, output_file):
    client = Groq(api_key=api_key)

    while True:
        response = get_llm_response(client, prompt, model)
        if response:
            with open(output_file, "a") as file:
                file.write(f"Prompt: {prompt}\nResponse: {response}\n\n")
            print("Response appended to text file.")
        time.sleep(interval)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a loop to prompt an LLM model and save responses.")
    parser.add_argument("api_key", type=str, default=10, help="The API key for the Groq client.")
    parser.add_argument("prompt", type=str, help="The prompt to send to the language model.")
    parser.add_argument("--model", type=str, default="llama-3.1-70b-versatile", help="The language model to use.")
    parser.add_argument("--interval", type=int, default=10, help="Interval in seconds between each prompt (default: 10).")
    parser.add_argument("--output_file", type=str, default="llm_responses.txt", help="The file to save responses (default: llm_responses.txt).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args.api_key, args.prompt, args.model, args.interval, args.output_file)
