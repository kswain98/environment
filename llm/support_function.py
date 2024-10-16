'''snipped of code taken from LLM-Planner paper. https://github.com/OSU-NLP-Group/LLM-Planner/blob/v1/hlp_planner.py'''

'''reference response:
'''
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

def check_feasibility(action, graph):
    """check if the action is feasible in the current environment"""

    ##check if the object / parameter exists in the current graph (can be used specific to environment scene graph)
    # graph_objects = [graph['KswainEscapeRoom2'][i]['name'] for i in range(len(graph['KswainEscapeRoom2']))]
    # if action.split(' ')[1] not in graph_objects:
    #     return False

    ##call communication API to check feasibility in unreal engine

    return True

def infer_action_mapping(graph, response):
    """this function processes LLM response to map onto feasible actions.
    Exact format from the response is expected, we only process the first n line of the response.
    Additionally check for feasibility from environment."""

    assert "LLM Output" in response, "Response does not contain LLM output required for actions"

    #only process llm's output related to actions
    message_begining = "LLM Output:"
    message_end = "\n\n" #alternatively, "{" for beginning of json
    filtered_response = (data.split(message_begining)[1]).split(message_end)[0]
    intended_actions = filtered_response.split("\n")

    #build action set
    #and check for feasibility
    action_list = [action.replace('  ', '').split(' ') for action in intended_actions \
        if action.split(' ')[0].lower() in fixed_action_ids.keys() and check_feasibility(action, graph=None)]

    return action_list



