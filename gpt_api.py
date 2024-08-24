from openai import OpenAI



class OpenAIBot():

    def __init__(self):
        """
        Define openai agent and give prompt examples.
        Make this specific to the escape room challenge
        """
        self.client = OpenAI(
            organization="your_org?",
            api_key="your_key?",
        ) 
        self.initalization = "You're an embodied agent trapped in a simulated environment game popularly known as an escape room. \
            You will be provided with a background story about the escape room which includes hints to what needs to be done within the room such that you can escape. \
            You are able to see, move around, and interact with objects around you. However, you cannot be violent or break things. \
            The format of this interactive game is iterative, where you are provided information about the objects around and you need to send back a single \
            line of instruction of what you would do in the situation."
        self.sys_msg = """You are an agent trapped in an escape room, use the observation and history clues wot figure out your next move. You can do the following:
            1. Move to <location name>
            2. Pick up <nearby object name>
            3. ....
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

