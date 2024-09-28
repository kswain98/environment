# Some test examples: #

# Use UI:
#   python llm_escaper.py --ui

# Don't use UI:
#  * Use openai api with key:
#     python llm_escaper.py --api-key your_openai_key --api-option openai --model gpt-4o-mini
#
#  * Use local llm served through vllm server:
#     python llm_escaper.py --model my_local_llm_model_path --llm-url my_vllm_server_url


from typing import Literal, Union

from argparse import ArgumentParser

import re
import json

from llm import OpenAIBot
from utils import fixed_action_ids, sequence
from client import *

import gradio as gr


# Escape Room Configs #
available_escaperooms = ["KswainEscapeRoom4", "SIEEscapeRoom"]
environment = {}
llm_response = ""

# LLM prompts #
allowed_actions = list(fixed_action_ids.keys())

obs_start_phrase = "Start of New Observation Graph"
obs_end_phrase = "End of New Observation Graph"

sys_msg = f"""You're playing a game as an embodied agent trapped in an escape room, your goal is to escape from it. 
Once the room is successfully created, you can call the "func <observe_environment>" function anytime to obtain the observation graph in json format representing the objects and their corresponding relationships in the room.

Once you obtain the observation graph in json format, you need to analyze it to understand what objects are in the room and what their attributes are. After you understand the observation graph for the current environment, you can instruct the agent to interact with the environment in order to escape the room. You're only allowed to use the following {len(allowed_actions)} actions: {' '.join(allowed_actions)}. Please try different action instructions if you noticed no changes in the observation graph since the previous action instruction.

Whenever you instruct some actions to the agent, you need to observe the updated environment again by calling "func <observe_environment>", and you'll need to analyze the updated observation graph presented in json format to instruct the agent to take either a new action or a new series of actions.

The action instructions to the agent should follow the format as below:
Example 1:
There is only one agent in the environment, you ask the agent to run to object with id 1, instruction will be:
    func <action: agent_1 run to object_1>

Example 2:
There is only one agent in the environment, you ask the agent to grab object with id 2, if the object is far from the agent, instruction will be:
    func <action: agent_1 run to object_2>
    func <action: agent_1 grab object_2>

If the object is next to the agent, instruction will just be:
    func <action: agent_1 grab object_2>

Please note that in order to successfully grab an object, the agent must be next to that object, otherwise, ask the agent to move to the object first.

Example 3:
There are two agents in the environment, you ask agent_1 to run to object with id 3, and you ask agent_2 to grab object with id 4, if object 4 is far from agent_2, instruction will be:
    func <action: agent_1 run to object_3>
    func <action: agent_2 run to object_4>
    func <action: agent_2 grab object_4>

If object 4 is next to agent_2, the instruction will be:
    func <action: agent_1 run to object_3>
    func <action: agent_2 grab object_4>

You only call "func <observe_environment>" after you instruct the agent to take some actions.

After all your actions are executed, please update the observation graph in the same json format and return it in the following struction:

{obs_start_phrase}
    the_updated_json_data
{obs_end_phrase}

And then call "func <update_observation>".

If you think the agent has successfully escaped from the room, or you don't know what to do anymore, end the program by sending "func <end>"."""

init_prompt = f"The name of the escaperoom is"

send_obs_prompt = f"You have successfully created the escape room. Now here is the observation graph in json format:\n"


log_output = True

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--ui", action='store_false', 
        help="set to use UI, if True, the following arguments are ignored and set through UI") 

    parser.add_argument("--api-key", default='000', 
        help="if using OpenAI API with key, specify the key") 
    parser.add_argument("--api-option", default='other', 
        help="[other | openai], set to 'openai' if using OpenAI API with key")
    parser.add_argument("--model", default='gpt-4o-mini', 
        help="model used for llm call")
    parser.add_argument("--llm-url", default='localhost:8000', 
        help="Ignore if using OpenAI API with key, otherwise specify url")

    parser.add_argument("--available-rooms", nargs="*", 
        help="available escaperoom names, separated by space") 
    
    return parser.parse_args()

def load_json(file):
    with open(file, 'r') as rf:
        data = json.load(rf)
    return data

def save_json(data, file):
    with open(file, 'w') as wf:
        json.dump(data, wf, indent=2)

def call_llm(sys_msg, query, llm):
    _, resp_text = llm(query, sys_msg=sys_msg)
    print_text = f"Start of Conversation.\nUser: {query}\nLLM: {resp_text}\nEnd of Conversation.\n\n"
    # print(print_text)

    if log_output:
        with open(f"{api_option}_llm_escaper.txt", 'a') as wf:
            wf.write(
                f"System Prompt:\n{sys_msg}\n{print_text}\n\n"
            )
    return resp_text

def create_environment(escape_room_name):
    data = {
        "environment": escape_room_name
    }

    make(data)

def get_observation():
    data = {
        "type": "graph", 
    }

    observation(data)
    environment = load_json('observation.json')
    return environment

def set_agent_actions(func_call: str):
    action_list = [func_call]
    data_list = sequence(action_list)

    for data in data_list:
        set_action(data)

def update_observation(llm_resp):
    assert obs_start_phrase in llm_resp and obs_end_phrase in llm_resp
    obs_start_idx = llm_resp.find(obs_start_phrase) + len(obs_start_phrase)
    obs_end_idx = llm_resp.find(obs_end_phrase)

    environment = json.loads(llm_resp[obs_start_idx : obs_end_idx])
    save_json(environment, 'new_observation.json')
    return environment

def end_program():
    exit(0)

def extract_func(llm_response):
    # Extract Response #
    func_pattern = r"func\s<[^>\n]+>?"
    function_calls = re.findall(func_pattern, llm_response)
    return function_calls

def parse_func(
    function_calls: list=[], 
    llm_resp: str=""):
    
    global environment

    for func in function_calls:
        func_call = func.replace("func", '').\
            replace('<', '').replace('>', '').replace("\\", "")\
            .strip()
        
        print(func_call)

        if "observe_environment" in func_call:
            environment = get_observation()

        elif "action" in func_call:
            set_agent_actions(func_call)

        elif "update" in func_call:
            environment = update_observation(llm_resp=llm_resp)

        elif "end" in func_call:
            end_program()

    return environment

def get_llm_response(
    api_key: str, api_option: str, llm_model: str, base_url: str,
    available_rooms: Union[str, list], user_input: str):

    global llm_response, send_obs_prompt, environment, available_escaperooms

    if api_option == "openai":
        openai_bot = OpenAIBot(
            model=llm_model, use_openai=True, api_key=api_key)
    else:
        openai_bot = OpenAIBot(
            base_url=base_url, model=llm_model, llm_config={'max_token': 2048})
        
    if isinstance(available_rooms, str):
        available_escaperooms = re.split(r'[ ,;.]', available_rooms)
    else:
        available_escaperooms = available_rooms

    if user_input in available_escaperooms:
        create_environment(user_input)
        environment = get_observation()
        llm_response = f"Succesfully created {user_input}!"

    else:
        send_obs_prompt += f"{environment}\n"

        llm_response = call_llm(
            sys_msg=sys_msg, 
            query=f"{user_input}\n{send_obs_prompt}",
            llm=openai_bot)
        
        function_calls = extract_func(llm_response)
        environment = parse_func(function_calls=function_calls, llm_resp=llm_response)

        
    return llm_response
    

def ui():
    with gr.Blocks() as exp:
        with gr.Row():
            api_key_box = gr.Textbox(
                label="api_key (Required if using OpenAI API with key, otherwise ignored)", 
                value='000', 
                interactive=True
            )
            api_option_box = gr.Textbox(
                label="api_option (Set to 'openai' if using OpenAI API with key)", 
                value='other', 
                interactive=True
            )
            llm_model_box = gr.Textbox(
                label="llm_model", 
                value='gpt-4o-mini',
                interactive=True
            )
            base_url_box = gr.Textbox(
                label="llm_url (Required if not using OpenAI API with key, otherwise set the url)", 
                value='localhost:8000', 
                interactive=True
            )    

        available_rooms = gr.Textbox(
            label="Available Escape Room Names (separated by ',', ';', '.' or space):", 
            value=' '.join(available_escaperooms), 
            interactive=True
        )
        user_input = gr.Textbox(
            label="User Input (start by giving escape room name):", 
            value="KswainEscapeRoom4", 
            interactive=True
        )
        llm_output = gr.Markdown(label="LLM Output:", value="") 

        send_request_btn = gr.Button("Get Response")
        send_request_btn.click(
            get_llm_response,
            inputs = [
                api_key_box, api_option_box, 
                llm_model_box, base_url_box,
                available_rooms, user_input],
            outputs = llm_output
        )
        exp.launch(share=True, debug=False)


if __name__ == "__main__":

    args = get_args()

    user_interact = args.ui

    api_key = args.api_key
    api_option = args.api_option
    llm_model = args.model
    base_url = args.llm_url

    available_rooms = args.available_rooms


    if user_interact:
        ui()
    else:
        if not available_rooms:
            available_rooms = available_escaperooms
        
        get_llm_response(
            api_key, api_option, llm_model, base_url,
            available_rooms, "KswainEscapeRoom4")

        try_steps = 5 # TODO: set to 2 for quick debug
        while try_steps:
            get_llm_response(
                api_key, api_option, llm_model, base_url,
                available_rooms, "")
            try_steps -= 1