'''snipped of code taken from LLM-Planner paper. https://github.com/OSU-NLP-Group/LLM-Planner/blob/v1/hlp_planner.py'''

import openai

#NOTE Add your openai API key here
openai.api_key= ""

fixed_action_ids = {
            "stand": 1, 
            "walk": 2, 
            "run": 3, 
            "drive": 4, 
            "grab": 5, 
            "place": 6, 
            "open": 7, 
            "close": 8, 
            "lookat": 9, 
            "switch on": 10, 
            "switch off": 11, 
            "sit": 12, 
            "sleep": 13, 
            "eat": 14, 
            "drink": 15, 
            "clean": 16, 
            "point": 17, 
            "pull": 18, 
            "push": 19, 
            "cook": 20, 
            "cut": 21, 
            "pour": 22, 
            "shower": 23, 
            "dry": 24, 
            "lock": 25, 
            "unlock": 26, 
            "fill": 27, 
            "talk": 28, 
            "laugh": 29, 
            "angry": 30, 
            "cry": 31, 
            "call": 32, 
            "interact": 33, 
            "step_forward": 34, 
            "step_backwards": 35, 
            "turn_left": 36, 
            "turn_right": 37}
    

def infer_action_mapping(graph, response):
    """this function processes LLM response to map onto feasible actions. Additionally check for feasibility from environment."""
    return action



