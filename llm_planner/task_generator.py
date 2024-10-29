import os
import time
import argparse
import logging
from groq import Groq
import requests


logging.basicConfig(
    filename="error_log.txt",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(message)s",
)


def get_llm_response(client, prompt, model):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )
        rate_limits = {
            "remaining_requests": chat_completion.headers.get(
                "x-ratelimit-remaining-requests"
            ),
            "remaining_tokens": chat_completion.headers.get(
                "x-ratelimit-remaining-tokens"
            ),
            "retry_after": chat_completion.headers.get("retry-after"),
        }
        return chat_completion.choices[0].message.content, rate_limits
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            retry_after = int(e.response.headers.get("retry-after", 10))
            logging.error(f"Rate limit hit. Retrying after {retry_after} seconds.")
            time.sleep(retry_after)
            return None, {}
        logging.error(f"Error fetching response: {e}")
        return None, {}


def main(api_key, prompt, model, interval, output_file):
    client = Groq(api_key=api_key)

    while True:
        response, rate_limits = get_llm_response(client, prompt, model)

        if response:
            with open(output_file, "a") as file:
                file.write(f"Prompt: {prompt}\nResponse: {response}\n\n")
            print("Response appended to text file.")

            if (
                rate_limits.get("remaining_requests")
                and int(rate_limits["remaining_requests"]) < 10
            ):
                print("Warning: Approaching daily request limit.")
            if (
                rate_limits.get("remaining_tokens")
                and int(rate_limits["remaining_tokens"]) < 50
            ):
                print("Warning: Approaching token rate limit.")
        else:
            print("No response received. Check error_log.txt for details.")

        retry_after = int(rate_limits.get("retry_after", interval))
        time.sleep(retry_after)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a loop to prompt an LLM model and save responses."
    )
    parser.add_argument("api_key", type=str, help="The API key for the Groq client.")
    parser.add_argument(
        "prompt", type=str, help="The prompt to send to the language model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.1-70b-versatile",
        help="The language model to use.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Interval in seconds between each prompt (default: 10).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="llm_responses.txt",
        help="The file to save responses (default: llm_responses.txt).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.api_key, args.prompt, args.model, args.interval, args.output_file)
