from openai import OpenAI

class OpenAIBot():

    def __init__(self, 
        base_url='localhost',
        model="gpt-3.5-turbo-0125", 
        use_openai=False,
        api_key="000",
        llm_config={}
        ):

        """
        Define openai agent and give prompt examples.
        Make this specific to the escape room challenge

        Args:
            base_url (str): used when running local llm server, i.e. localhost:8000

            model (str): either local llm path or the name for openai models

            use_openai (bool): if True, use openai models, otherwise, use llm servered locally through base_url
        """

        if use_openai:
            self.client = OpenAI(
                # organization="your-org",
                # project="your-project",
                api_key=api_key,
            ) 
        else:
            openai_api_base = f"http://{base_url}/v1"
            self.client = OpenAI(
                api_key=api_key,
                base_url=openai_api_base,
            )

        self.initalization = "You're an embodied agent trapped in a simulated environment game popularly known as an escape room. \
            You will be provided with a background story about the escape room which includes hints to what needs to be done within the room such that you can escape. \
            You are able to see, move around, and interact with objects around you. However, you cannot be violent or break things. \
            The format of this interactive game is iterative, where you are provided information about the objects around and you need to send back a single \
            line of instruction of what you would do in the situation."
            
        self.sys_msg = """You are an agent trapped in an escape room, use the observation and history clues to figure out your next move. You can do the following:
            1. Move to <location name>
            2. Pick up <nearby object name>
            3. ....
            """
        
        self.sampling_params = {
            "max_tokens": 16384, # max_tokens for gpt-4o-mini
            "temperature": 0.4,
            "top_p": 0.9
        }
 
        if llm_config:
            for k, v in self.sampling_params.items():
                if k in llm_config:
                    self.sampling_params[k] = llm_config[k]

        self.use_openai = use_openai
        self.model = model
   
    def __call__(self, text, image=None, img_format=None, sys_msg=""):
        """
        Given the paragraph, parse it into the steps.
        Args:
            texts: str, paragraph of subgoal <details>
        Return:
            output: str, step-by-step instruction based on <details>
        """
        if sys_msg:
            self.sys_msg = sys_msg

        user_query = {"role": "user", "content": [{"type": "text", "text": text}]}
        if image:
            user_query["content"].append(
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/{img_format};base64,{image}"}
                }
            )
        
        if self.use_openai:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.sys_msg},
                    user_query,
                ],
                max_tokens=self.sampling_params['max_tokens'],
                temperature=self.sampling_params['temperature'],
                top_p=self.sampling_params['top_p']
            )
        
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages= [
                    {"role": "user", "content": self.sys_msg},
                    {"role": "assistant", "content": "Sure."},
                    {"role": "user", "content": text}
                ],
                max_completion_tokens=self.sampling_params['max_tokens'],
                temperature=self.sampling_params['temperature'],
                top_p=self.sampling_params['top_p']
            )

        response_text = response.choices[0].message.content
        return response, response_text


    def update_sampling_params(self, sampling_params: dict={}):
        """
        Update the sampling parameters for the openaiapi call.
        If using locally served llm, please make sure the keys are compatible.

        Args:
            sampling_params (dict): the keys should be compatible with the openai-api chat completion call
        """
        if sampling_params:
            for k, v in sampling_params.items():
                if k in self.sampling_params:
                    self.sampling_params[k] = v





if __name__ == "__main__":

    base_url = "localhost:8000"

    local_model = "/local_path/models/Mistral-7B-Instruct-v0.2"
    openai_model = "gpt-4o-mini"

    api_key = "000"

    api_option = "openai" # "local"
    log_output = True

    test_case = 3
            
    if test_case == 1 or test_case == "all":
        ## >>> Test 1 >>>
        if api_option == "openai":
            openai_bot = OpenAIBot(model=openai_model, use_openai=True, api_key=api_key)
        else:
            openai_bot = OpenAIBot(base_url=base_url, model=local_model, llm_config={'max_token': 2048})
        
        prompt = "Hi, how are you?"

        resp, resp_text = openai_bot(prompt, sys_msg="")
        print(f"User: {prompt}\nLLM: {resp_text}\n")
        ## <<< Test 1 <<<


    elif test_case == 2 or test_case == "all":
        ## >>> Test 2 >>>
        import json
        with open("../observation.json", 'r') as f:
            graph = json.load(f)

        prompts = [
            f"Can you understand the json data as an observation graph?\nThe graph:\n{graph}",
            "Please describe briefly what's in the data?",
            """Can you modify the json data to reflect that an apple is added on top of the table, please keep the same json format. 
            Please present the new json data to me, and title it as: new_observation.json .""",
            """If there is an agent in the environment, can you guide the agent to pick up the apple?
            Please represent the corresponding actions in the same json format.
            Please present the new json data to me, and title it as: new_observation_pickup_apple.json .""",
        ]

        sys_msg = "You're a knowledgable assistant who can well understand the virtual home simulation environment and good at problem solving, please answer the requests from the us."
        
        if api_option == "openai":
            openai_bot = OpenAIBot(model=openai_model, use_openai=True, api_key=api_key)
        else:
            openai_bot = OpenAIBot(base_url=base_url, model=local_model, llm_config={'max_token': 2048})
        
        ret = []
        for p in prompts:
            resp, resp_text = openai_bot(p, sys_msg=sys_msg)
            print_text = f"User: {p}\nLLM: {resp_text}\n\n"
            print(print_text)
            ret.append(print_text)

        if log_output:
            with open(f"{api_option}_llm_test_case{test_case}.txt", 'a') as wf:
                wf.write(f"\n{'#' * 80}\n".join(ret))
        ## <<< Test 2 <<<

    elif test_case == 3 or test_case == "all":
        ## >>> Test 3 >>>
        import json
        with open("../observation.json", 'r') as f:
            graph = json.load(f)

        fixed_action_ids = {
            "stand": 1, "walk": 2, "run": 3, "drive": 4, "grab": 5, 
            "place": 6, "open": 7, "close": 8, "lookat": 9, "switch on": 10, 
            "switch off": 11, "sit": 12, "sleep": 13, "eat": 14, "drink": 15, 
            "clean": 16, "point": 17, "pull": 18, "push": 19, "cook": 20, 
            "cut": 21, "pour": 22, "shower": 23, "dry": 24, "lock": 25, 
            "unlock": 26, "fill": 27, "talk": 28, "laugh": 29, "angry": 30, 
            "cry": 31, "call": 32, "interact": 33, "step_forward": 34, "step_backwards": 35, 
            "turn_left": 36, "turn_right": 37}


        sys_msg = f"""You're an embodied agent trapped in a simulated environment game popularly known as an escape room.
You will be provided with an obervation graph in json format representing the objects and their corresponding relationships within the room.
You'll be given a set of allowed actions. However, you cannot be violent or break things.
The allowed actions are represented in the format of 'action_name: action_index', a complete list of allowed actions is as follows: {fixed_action_ids}\n.
The format of this interactive game is iterative, where you are provided information about the objects around and you need to send back a single line of instruction of what you would do in the situation.
While trying to escape, please report your every action in the following format:
    Move to <location or object>
    Pick up <object name>
    , etc.
After all your actions are executed, please update the observation graph in the same json format, and save it out to new_observation.json file."""
    

        prompt = f"The observation graph of current environment is as follows:\n{graph}\n \
        Please instruct the agent actions in the above action format and \
        output the final observation graph in json format."

        if api_option == "openai":
            openai_bot = OpenAIBot(model=openai_model, use_openai=True, api_key=api_key)
        else:
            openai_bot = OpenAIBot(base_url=base_url, model=local_model, llm_config={'max_token': 2048})

        resp, resp_text = openai_bot(prompt, sys_msg=sys_msg)
        print_text = f"User: {prompt}\nLLM: {resp_text}\n\n"
        print(print_text)
       
        if log_output:
            with open(f"{api_option}_llm_test_case{test_case}.txt", 'w') as wf:
                wf.write(f"System Prompt:\n{sys_msg}\n\nUser Prompt:\n{prompt}\n\nLLM Output:\n{resp_text}\n\n")
       
        ## <<< Test 3 <<<

    elif test_case == 4 or test_case == "all":
        ## >>> Test 4 >>>
        import os
        import json
        from image_utils import get_image_encode
        obs_file = "../observation.json"
        obs_file = os.path.abspath(obs_file)
        with open(obs_file, 'r') as f:
            graph = json.load(f)


        image_dir = "../../test_images"
        image_dir = os.path.abspath(image_dir)

        images, img_format = get_image_encode(image_dir, start_idx=0, count=1)

        sys_msg = f"""You're an embodied agent trapped in a simulated environment game popularly known as an escape room.
Your goal is to observe and interact with the environment to escape the room.
You'll be given a set of allowed actions to control the movement of the agents. However, you cannot be violent or break things.
Everytime you control the agent to take an action, the observation of the environment will be returned to you by an image, you need to translate the image into an observation graph in json format. An example of the observation graph in json format is as follows:\n{graph}\n
While trying to escape, please report your every action in the following format:
Move to <location or object>
    Turn left
    Pick up <object name>
    , etc."""
        

        prompt = "Can you tell what objects are in the image and what are the relationships between those objects? Please refer to the above example of observation graph in json format and generate the json format of observation graph for this image."
       
        if api_option == "openai":
            openai_bot = OpenAIBot(model=openai_model, use_openai=True, api_key=api_key)
        else:
            openai_bot = OpenAIBot(base_url=base_url, model=local_model, llm_config={'max_token': 2048})
        
        for img in images:
            resp, resp_text = openai_bot(prompt, img, img_format, sys_msg=sys_msg)
            print_text = f"User: {prompt}\nLLM: {resp_text}\n\n"
            print(print_text)      

            if log_output:
                with open(f"{api_option}_llm_test_case{test_case}", 'a') as wf:
                    wf.write(f"System Prompt:\n{sys_msg}\n\nUser Prompt:\n{prompt}\n\nLLM Output:\n{resp_text}\n\n")

        ## <<< Test 4 <<<