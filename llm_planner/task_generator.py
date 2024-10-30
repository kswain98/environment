import argparse
from groq import Groq


def get_llm_response(client, prompt, model):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )
    return chat_completion.choices[0].message.content


# Argument parsing
parser = argparse.ArgumentParser(description="Generate responses using Groq LLM API.")
parser.add_argument("--api_key", required=True, help="API key for Groq.")
parser.add_argument("--prompt", required=True, help="Prompt for the LLM.")
parser.add_argument(
    "--model", default="llama-3.1-70b-versatile", help="Model version for the LLM."
)
parser.add_argument(
    "--output_file", default="output.txt", help="File to save the responses."
)
args = parser.parse_args()

client = Groq(api_key=args.api_key)

ctr = 0

while True:
    try:
        response = get_llm_response(client, args.prompt, args.model)

        formatted_response = (
            f"Response #{ctr + 1}\n" f"{'-'*40}\n" f"{response}\n" f"{'='*60}\n\n"
        )

        with open(args.output_file, "a") as file:
            file.write(formatted_response)
            print(f"Response #{ctr + 1} saved to {args.output_file}.")

        ctr += 1
    except Exception as e:
        print("Error in getting response from Groq:", e)
