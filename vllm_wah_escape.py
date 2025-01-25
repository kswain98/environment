from typing import Dict, List, Tuple, Union
import json
import os
from llm import OpenAIBot
from utils import sequence
from client import *
import time
import openai
import base64

class Agent:
    """Base class for LLM-based agents in Watch and Help environment"""
    
    def __init__(
        self,
        agent_id: int,
        agent_name: str,
        api_key: str,
        api_option: str = "openai",
        model: str = "gpt-4v",
        base_url: str = "localhost:5000",
        max_steps: int = 20,
        debug: bool = False,
        environment: str = "WatchAndHelp1"
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.max_steps = max_steps
        self.debug = debug
        self.environment = environment
        
        # Set up screenshot directory and clear existing screenshots
        self.screenshot_dir = r"E:\environment-sourcecode\environment\Saved\Screenshots\WindowsEditor"
        self._clear_screenshots()
        
        self.screenshot_counter = 0
        self.current_state = self.get_observation()
        
        # Initialize two LLMs - one for Alice and one for Bob
        if api_option == "openai":
            self.alice_llm = OpenAIBot(model=model, use_openai=True, api_key=api_key, llm_config={'max_token': 2048})
            self.bob_llm = OpenAIBot(model=model, use_openai=True, api_key=api_key, llm_config={'max_token': 2048})
        else:
            self.alice_llm = OpenAIBot(base_url=base_url, model=model, llm_config={'max_token': 2048})
            self.bob_llm = OpenAIBot(base_url=base_url, model=model, llm_config={'max_token': 2048})
            
        # Load object info
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(f'{dir_path}/dataset/object_info_small.json', 'r') as f:
            self.object_info = json.load(f)
        #make the initial observation
        self.get_observation()
        self.alice_system_prompt = self._create_alice_prompt()
        self.bob_system_prompt = self._create_bob_prompt()
        self.environment = environment
        # Add render configuration
        self.render_config_alice = {
            "render_pipeline": 'raytracing',
            "camera_index": [0],
            "image_width": [1920],
            "image_height": [1080],
            "fps": [60],
            "fov": [90],
        }
        self.render_config_bob = {
            "render_pipeline": 'raytracing',
            "camera_index": [0],
            "image_width": [1920],
            "image_height": [1080],
            "fps": [60],
            "fov": [90],
        }
        self.allowed_relations = ['on', 'inside']
        self.allowed_state = ['open', 'closed', "switchon", "switchoff"]
        # Add tracking for current goals
        self.alice_current_goals = {}
        self.bob_current_goals = {}
        self.alice_completed_goals = set()
        self.bob_completed_goals = set()

    def _clear_screenshots(self):
        """Clear all existing screenshots in the directory"""
        if os.path.exists(self.screenshot_dir):
            for file in os.listdir(self.screenshot_dir):
                if file.startswith("HighresScreenshot"):
                    try:
                        os.remove(os.path.join(self.screenshot_dir, file))
                    except Exception as e:
                        print(f"Failed to remove {file}: {e}")

    def get_observation(self) -> Dict:
        """Gets the current environment observation"""
        observation({"type": "full"})
        
        with open('graph.json', 'r') as f:
            data = json.load(f)
        self.current_state = data[self.environment]
        return self.current_state

    def execute_action(self, action: str) -> None:
        """Executes a single action in the environment"""
        if self.debug:
            print(f"{self.agent_name} executing: {action}")
        
        action = action.replace('.0', '')
        action_sequence = sequence([action])
        for action_dict in action_sequence:
            set_action(action_dict)
            if 'grab' in action :
                time.sleep(5)
            else:
                time.sleep(2)

    def get_screenshot(self, render_config: Dict) -> str:
        """Captures and saves a screenshot of the current state and returns base64 encoded image"""
        # Render new screenshot
        render(render_config)
        time.sleep(0.5)  # Give a small delay for the screenshot to be saved
        
        # Find the new screenshot
        files = [f for f in os.listdir(self.screenshot_dir) if f.startswith("HighresScreenshot")]
        if not files:
            raise ValueError("No screenshot found after rendering")
        
        latest_screenshot = max(files, key=lambda x: os.path.getctime(os.path.join(self.screenshot_dir, x)))
        screenshot_path = os.path.join(self.screenshot_dir, latest_screenshot)
        
        # Read and encode the image as base64
        with open(screenshot_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        return encoded_string
    def get_all_objects(self, obs: List[Dict]) -> Dict:
        """Gets all the objects in the scene"""
        # List of objects to exclude
        self.invalid_objects = [
            "Wall", 
            "SpotLight",
            "SM_DownLight",
            "RugsRectanglex",
            "RectLight",
            "CameraActor",
            "RecastNavMeshDefault",
            "NavMeshBoundsVolume",
            "PostProcessVolume",
            "PlayerStart",
        ]
        
        return [obj['name'] for obj in obs if not any(invalid.lower() in obj['name'].lower() for invalid in self.invalid_objects)]
        
    def format_observation(self, obs: List[Dict], goal_spec: Dict, render_config: Dict) -> Dict:
        """Formats the observation for both agents"""
        # Validate goal state objects exist in observation
        all_objects = self.get_all_objects(obs)
        for (subject, relation, target), count in goal_spec.items():
            if relation != 'state':
                if subject not in all_objects:
                    raise ValueError(f"Goal state contains invalid subject: {subject}")
                if target not in all_objects:
                    raise ValueError(f"Goal state contains invalid target: {target}")
            else:
                # only look for subject node if this is a state goal
                subject_node = next((node for node in obs if subject.lower() in node['name'].lower()), None)
                formatted = {"subject": subject_node}
        subject, relation, target = list(goal_spec.keys())[0]
       
        # Find nodes for all relevant objects and agents
        subject_node = next((node for node in obs if subject.lower() in node['name'].lower()), None)
        # Only look for target node if this is a relation goal, not a state goal
        target_node = next((node for node in obs if target.lower() in node['name'].lower()), None) if relation != 'state' else {'name': target, 'id': -1, 'transform': ['X=0.000 Y=0.000 Z=0.000'], 'relation': [], 'state': []}
        
        # Find both BP_ThirdPersonCharacter instances
        character_nodes = [node for node in obs if 'BP_ThirdPersonCharacter' in node['name']]
        if len(character_nodes) >= 2:
            alice_node = character_nodes[0]  # First instance is Alice
            bob_node = character_nodes[1]    # Second instance is Bob
        else:
            alice_node = None
            bob_node = None
        
        # Handle missing nodes
        missing = []
        if not subject_node: missing.append(f"subject ({subject})")
        if not target_node and relation != 'state': missing.append(f"target ({target})")
        if not alice_node: missing.append("Alice (BP_ThirdPersonCharacter)")
        if not bob_node: missing.append("Bob (BP_ThirdPersonCharacter)")
        if missing:
            raise ValueError(f"Missing objects in observation: {', '.join(missing)}")
        
        # Only keep position from transform
        formatted = {
            'target': {
                'name': target_node['name'],
                'id': target_node['id'],
                'position': target_node['transform'][0],  # Only keep position
                'relations': target_node.get('relation', []),
                'state': target_node['state']
            },
            'subject': {
                'name': subject_node['name'],
                'id': subject_node['id'],
                'position': subject_node['transform'][0],  # Only keep position
                'relations': subject_node.get('relation', []),
                'state': subject_node['state']
            },
            'alice': {
                'name': alice_node['name'],
                'id': alice_node['id'],
                'position': alice_node['transform'][0],  # Only keep position
                'relations': alice_node.get('relation', []),
                'state': alice_node['state']
            },
            'bob': {
                'name': bob_node['name'],
                'id': bob_node['id'],
                'position': bob_node['transform'][0],  # Only keep position
                'relations': bob_node.get('relation', []),
                'state': bob_node['state']
            }
        }

        # Create prompt string
        formatted['prompt'] = f"""Current state:
Target: {target_node['name']} (id: {target_node['id']}) at {target_node['transform'][0]} relations: {', '.join(target_node.get('relation', []))} state: {target_node['state']}
Subject: {subject_node['name']} (id: {subject_node['id']}) at {subject_node['transform'][0]} relations: {', '.join(subject_node.get('relation', []))} state: {subject_node['state']}
Alice: {alice_node['name']} (id: {alice_node['id']}) at {alice_node['transform'][0]} relations: {', '.join(alice_node.get('relation', []))} state: {alice_node['state']}
Bob: {bob_node['name']} (id: {bob_node['id']}) at {bob_node['transform'][0]} relations: {', '.join(bob_node.get('relation', []))} state: {bob_node['state']}"""

        # Add screenshot
        formatted['image'] = self.get_screenshot(render_config)
        
        return formatted
    
    def _create_goal_state_prompt(self, all_objects: List[str], user_prompt: str) -> str:
        return f"""Based on the image and the information of the whole graph feeding to you, determine the goal state of the scene.
        The goal state is the state of the scene that you want to achieve based on the user prompt: {user_prompt}

Goal states must be in one of these two formats:
1. (subject, relation, target): count
   Example: ('milk', 'on', 'table'): 1
   - Relations can be: {self.allowed_relations}

2. (subject, 'state', state_type): count
   Example: ('refrigerator', 'state', 'open'): 1
   - State types can be: {self.allowed_state}

You can generate multiple goal states and put them in a dictionary with key as the goal state and value as the count.
All the objects in the scene are: {all_objects} 
IMPORTANT: Only use objects from this list - do not hallucinate new objects.
After you have generated the goal state, check if the goal state makes logical sense to you. If it does not, get rid of it.
Generate at most 3 goal states at a time.
Respond with only the goal state as a python dictionary, no explanation needed."""

    def _create_alice_prompt(self) -> str:
        allowed_actions = ['walk to', 'grab', 'put', 'putin', 'open', 'close']
        return f"""You are Alice, the primary AI assistant completing household tasks. You control agent_0 
using these actions: {', '.join(allowed_actions)}. 
Use format: agent_0 <action> object_<id>

Rules:
1. Don't repeat "walk to" if already near target (distance < 2.0)
2. Can only hold one object at a time
3. Must grab before putting
4. Open containers before putting things inside
5. Use "put" for 'on' relations, "putin" for 'inside' relations, but only {allowed_actions} are allowed
6. Object IDs must be integers

When given a goal:
- Break it down into steps
- Execute one step at a time
- Verify each action's success
- Coordinate with Bob (Agent1) who will help you

Respond with only the next action, no explanation needed."""

    def _create_bob_prompt(self) -> str:
        allowed_actions = ['walk to', 'grab', 'put', 'putin', 'open', 'close']
        return f"""You are Bob, a helper AI assistant collaborating with Alice (Agent0) to complete tasks efficiently. 
You control agent_1 using these actions: {', '.join(allowed_actions)}.
Use format: agent_1 <action> object_<id>

Collaboration Rules:
1. Observe Alice's actions and infer her objective
2. Choose complementary tasks - don't target the same object as Alice
3. If Alice is handling one goal object, focus on another
4. Coordinate movements to avoid collisions 
5. Help if Alice seems stuck 
6. Prioritize tasks furthest from Alice's position

Core Rules:
1. Don't repeat "walk to" if already close to target
2. Can only hold one object at a time
3. Must grab before putting
4. Open containers before putting things inside
5. Use "put" for 'on' relations, "putin" for 'inside' relations
6. Object IDs must be integers


Respond with only the next action, no explanation needed."""

    def check_goal_completion(self, obs: List[Dict], goal_state: Dict) -> List[Tuple]:
        """Checks which goals have been completed"""
        completed_goals = []
        
        for (subject, relation, target), count in goal_state.items():
            if relation == 'state':
                # Check state goals
                subject_node = next((node for node in obs if subject.lower() in node['name'].lower()), None)
                if subject_node and target in subject_node['state']:
                    completed_goals.append((subject, relation, target))
            else:
                # Check relation goals
                subject_node = next((node for node in obs if subject.lower() in node['name'].lower()), None)
                if subject_node:
                    relations = subject_node.get('relation', [])
                    if f"{relation}({target})" in relations:
                        completed_goals.append((subject, relation, target))
        
        return completed_goals

    def run(self, user_prompt: str) -> float:
        """Runs both agents simultaneously to achieve the task specified in the prompt"""
        steps_taken = 0
        
        # Get initial goal states
        obs = self.get_observation()
        all_objects = self.get_all_objects(obs)
        goal_state_prompt = self._create_goal_state_prompt(all_objects, user_prompt)
        _, alice_analysis = self.alice_llm(goal_state_prompt, image=self.get_screenshot(self.render_config_alice))
        _, bob_analysis = self.bob_llm(goal_state_prompt, image=self.get_screenshot(self.render_config_bob))
        
        try:
            self.alice_current_goals = eval(alice_analysis.strip().strip('`python').strip('`'))
            self.bob_current_goals = eval(bob_analysis.strip().strip('`python').strip('`'))
        except Exception as e:
            print(f"Error parsing initial goal states: {e}")
            return 0.0

        while steps_taken < self.max_steps:
            obs = self.get_observation()
            time.sleep(1)
            
            # Check for completed goals
            alice_completed = self.check_goal_completion(obs, self.alice_current_goals)
            bob_completed = self.check_goal_completion(obs, self.bob_current_goals)
            
            # Remove completed goals
            for goal in alice_completed:
                if goal in self.alice_current_goals:
                    self.alice_completed_goals.add(goal)
                    del self.alice_current_goals[goal]
                    
            for goal in bob_completed:
                if goal in self.bob_current_goals:
                    self.bob_completed_goals.add(goal)
                    del self.bob_current_goals[goal]
            
            # If all goals are completed, get new goals
            if not self.alice_current_goals:
                _, alice_analysis = self.alice_llm(goal_state_prompt, image=self.get_screenshot(self.render_config_alice))
                try:
                    new_goals = eval(alice_analysis.strip().strip('`python').strip('`'))
                    self.alice_current_goals.update({k: v for k, v in new_goals.items() 
                                                  if k not in self.alice_completed_goals})
                except Exception:
                    pass
                    
            if not self.bob_current_goals:
                _, bob_analysis = self.bob_llm(goal_state_prompt, image=self.get_screenshot(self.render_config_bob))
                try:
                    new_goals = eval(bob_analysis.strip().strip('`python').strip('`'))
                    self.bob_current_goals.update({k: v for k, v in new_goals.items() 
                                                if k not in self.bob_completed_goals})
                except Exception:
                    pass
            
            # Format observations with current goals
            formatted_obs_alice = self.format_observation(obs, self.alice_current_goals, self.render_config_alice)
            formatted_obs_bob = self.format_observation(obs, self.bob_current_goals, self.render_config_bob)
            
            prompt_alice = (formatted_obs_alice['prompt'] + 
                          f"\nTask: {user_prompt}" + 
                          f"\nCurrent goals: {self.alice_current_goals}" +
                          f"\nCompleted goals: {self.alice_completed_goals}")
            
            prompt_bob = (formatted_obs_bob['prompt'] + 
                         f"\nTask: {user_prompt}" + 
                         f"\nCurrent goals: {self.bob_current_goals}" +
                         f"\nCompleted goals: {self.bob_completed_goals}")

            # Get actions from both agents simultaneously
            _, alice_response = self.alice_llm(prompt_alice, sys_msg=self.alice_system_prompt)
            _, bob_response = self.bob_llm(prompt_bob, sys_msg=self.bob_system_prompt)
            print(alice_response)
            print(bob_response)
            # Execute Alice's action
            alice_action = alice_response.strip()
            if alice_action.startswith("agent_0"):
                if self.debug:
                    print(f"Alice executing: {alice_action}")
                self.execute_action(alice_action)
            
            # Execute Bob's action
            bob_action = bob_response.strip()
            if bob_action.startswith("agent_1"):
                if self.debug:
                    print(f"Bob executing: {bob_action}")
                self.execute_action(bob_action)
            
            steps_taken += 1
            time.sleep(1)
            
        if self.debug:
            print("Max steps reached")
        return 0.0

def run_experiment(user_prompt: str, api_key: str, api_option="openai", model="gpt-4o", debug=False, environment: str = "WatchAndHelp1"):
    """Runs the collaborative agents with a user-defined prompt"""
    agent = Agent(
        agent_id=0,
        agent_name="Coordinator",
        api_key=api_key,
        api_option=api_option,
        model=model,
        debug=debug,
        environment = environment
    )
    
    success = agent.run(user_prompt)
    print(f"Task completed with {success} success rate")
    return success

if __name__ == "__main__":
    render_config = {
        "render_pipeline": 'raytracing',
        "camera_index": [0],
        "image_width": [1920],
        "image_height": [1080],
        "fps": [60],
        "fov": [90],
    }
    data = {
        "environment": "WatchAndHelp1", 
    }
    observation({"type": "full"})
    time.sleep(1)

    make(data)
    set_action({'agent_index': [1], 'task': [6, 326, 0, 0]})

    user_prompt = "Clean up the floor and put the objects in the right place"
    render(render_config)
    run_experiment(
        user_prompt=user_prompt,
        api_key="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",
        debug=True,
        environment="WatchAndHelp1"
    ) 