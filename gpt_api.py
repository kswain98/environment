from openai import OpenAI



class OpenAIBot():

    def __init__(self):
        """
        Define openai agent and give prompt examples.
        """
        self.client = OpenAI(
            organization="your_org?",
            api_key="your_key?",
        ) 
        self.sys_msg = """You're an assistant to answer questions.
        """
 
   
    def __call__(self, text):
        """
        Given the paragraph, parse it into the steps.
        Args:
            texts: str, paragraph of subgoal <details>
        Return:
            output: str, step-by-step instruction based on <details>
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": self.sys_msg},
                {"role": "user", "content": text},
                ]
        )
        return response



## >>> Test >>>
openai_bot = OpenAIBot()
text = "Hi, how are you?"

resp = openai_bot(text)
resp_text = resp.choices[0].message.content
print(f"User: {text}\nGPT: {resp_text}\n")
## <<< Test <<<

