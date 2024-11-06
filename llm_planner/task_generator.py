import argparse
from groq import Groq
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
    description="Generate tasks and subgoals using Groq LLM API."
)
parser.add_argument("--api_key", required=True, help="API key for Groq.")
parser.add_argument(
    "--prompt", required=True, help="Initial prompt to generate the task."
)
parser.add_argument(
    "--subgoal_prompt",
    default="What would all the intermediate steps required to {task}",
    help="Custom prompt for generating subgoals. Use '{task}' as a placeholder for the generated task.",
)
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

ctr = 0

while True:
    try:
        task_prompt = prompt_template.format(
            history=memory.load_memory_variables({})["history"], input=args.prompt
        )
        task_response = groq_llm(task_prompt)

        memory.save_context({"input": args.prompt}, {"output": task_response})

        subgoal_prompt = args.subgoal_prompt.format(task=task_response)

        subgoal_input = prompt_template.format(
            history=memory.load_memory_variables({})["history"], input=subgoal_prompt
        )
        subgoal_response = groq_llm(subgoal_input)

        memory.save_context({"input": subgoal_prompt}, {"output": subgoal_response})

        formatted_response = (
            f"Task #{ctr + 1}\n"
            f"{'-'*40}\n"
            f"{task_response}\n\n"
            f"Subgoals:\n{subgoal_response}\n"
            f"{'='*60}\n\n"
        )

        with open(args.output_file, "a") as file:
            file.write(formatted_response)
            print(f"Task and subgoals #{ctr + 1} saved to {args.output_file}.")

        ctr += 1
    except Exception as e:
        print("Error in getting response from Groq:", e)
