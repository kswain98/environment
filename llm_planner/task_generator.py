import argparse
from groq import Groq
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from pydantic import BaseModel


class GroqLLM(LLM, BaseModel):
    client: Groq
    model: str

    def _call(self, prompt, stop=None):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model,
        )
        return chat_completion.choices[0].message.content

    @property
    def _identifying_params(self):
        return {"model": self.model}

    @property
    def _llm_type(self):
        return "groq"


parser = argparse.ArgumentParser(
    description="Generate responses using Groq LLM API."
)
parser.add_argument("--api_key", required=True, help="API key for Groq.")
parser.add_argument("--prompt", required=True, help="Initial prompt for the LLM.")
parser.add_argument(
    "--model", default="llama-3.1-70b-versatile", help="Model version for the LLM."
)
parser.add_argument(
    "--output_file", default="output.txt", help="File to save the responses."
)
args = parser.parse_args()

client = Groq(api_key=args.api_key)
memory = ConversationBufferMemory()
groq_llm = GroqLLM(client=client, model=args.model)

prompt_template = ChatPromptTemplate.from_template("{history}\nUser: {input}\n")
llm_chain = LLMChain(
    prompt=prompt_template,
    llm=groq_llm,
    memory=memory,
)

ctr = 0

while True:
    try:
        response = llm_chain.run(input=args.prompt)

        formatted_response = (
            f"Response #{ctr + 1}\n" f"{'-'*40}\n" f"{response}\n" f"{'='*60}\n\n"
        )

        with open(args.output_file, "a") as file:
            file.write(formatted_response)
            print(f"Response #{ctr + 1} saved to {args.output_file}.")

        ctr += 1
    except Exception as e:
        print("Error in getting response from Groq:", e)
