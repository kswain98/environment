from typing import Dict, List, Tuple, Union
import json
import os
from llm import OpenAIBot
from llm_planner.utils import sequence
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
        model: str = "gpt-4o",
        base_url: str = "localhost:5000",
        max_steps: int = 20,
        debug: bool = False
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.max_steps = max_steps
        self.debug = debug
        self.current_state = self.get_observation()
        
        # Set up screenshot directory and clear existing screenshots
        self.screenshot_dir = r"E:\environment-sourcecode\environment\Saved\Screenshots\WindowsEditor"
        self._clear_screenshots()
        
        self.screenshot_counter = 0
        
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
            
        self.alice_system_prompt = self._create_alice_prompt()
        self.bob_system_prompt = self._create_bob_prompt()

        # Add render configuration
        self.render_config = {
            "render_pipeline": 'raytracing',
            "camera_index": [0],
            "image_width": [1920],
            "image_height": [1080],
            "fps": [60],
            "fov": [90],
        }

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
        self.current_state = data['WatchAndHelp1']
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

    def get_screenshot(self) -> str:
        """Captures and saves a screenshot of the current state and returns base64 encoded image"""
        # Clear any existing screenshots before rendering
        self._clear_screenshots()
        
        # Render new screenshot
        render(self.render_config)
        time.sleep(0.5)  # Give a small delay for the screenshot to be saved
        
        # Find the new screenshot
        files = [f for f in os.listdir(self.screenshot_dir) if f.startswith("HighresScreenshot")]
        if not files:
            raise ValueError("No screenshot found after rendering")
        
        latest_screenshot = max(files, key=lambda x: os.path.getctime(os.path.join(self.screenshot_dir, x)))
        screenshot_path = os.path.join(self.screenshot_dir, latest_screenshot)
        
        # Convert to base64
        with open(screenshot_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        return encoded_string

    def format_observation(self, obs: List[Dict], goal_spec: Dict) -> Dict:
        """Formats the observation for both agents"""
        subject, relation, target = list(goal_spec.keys())[0]
        
        # Find nodes for all relevant objects and agents
        subject_node = next((node for node in obs if subject.lower() in node['name'].lower()), None)
        target_node = next((node for node in obs if target.lower() in node['name'].lower()), None)
        
        # Find both BP_ThirdPersonCharacter instances
        character_nodes = [node for node in obs if 'BP_ThirdPersonCharacter' in node['name']]
        if len(character_nodes) >= 2:
            alice_node = character_nodes[0]  # First instance is Alice
            bob_node = character_nodes[1]    # Second instance is Bob
        else:
            alice_node = None
            bob_node = None
        
        # Handle missing nodes
        if not all([subject_node, target_node, alice_node, bob_node]):
            missing = []
            if not subject_node: missing.append(f"subject ({subject})")
            if not target_node: missing.append(f"target ({target})")
            if not alice_node: missing.append("Alice (BP_ThirdPersonCharacter)")
            if not bob_node: missing.append("Bob (BP_ThirdPersonCharacter)")
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
        formatted['image'] = self.get_screenshot()
        
        return formatted

    def check_goal_reached(self, goal_spec: Dict) -> bool:
        """Checks if the current goal has been reached"""
        obs = self.get_observation()
        
        for (subject, relation, target), _ in goal_spec.items():
            # Find the relevant nodes
            subject_node = next((node for node in obs if subject.lower() in node['name'].lower()), None)
            target_node = next((node for node in obs if target.lower() in node['name'].lower()), None)
            
            if not subject_node or not target_node:
                return False
            
            # Check if the relation exists in the relations array
            relations = subject_node.get('relation', [])
            expected_relation = f"{subject_node['name']} {relation} {target_node['name']}"
            
            if expected_relation not in relations:
                return False
                
        return True

    def format_goal(self, goal_spec: Dict) -> str:
        """Formats the goal specification into a string"""
        goal_str = "Goal:\n"
        for (subject, relation, target), _ in goal_spec.items():
            goal_str += f"Place {subject} {relation} {target}\n"
        return goal_str

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
4. Coordinate movements to avoid collisions (maintain distance > 2.0)
5. Help if Alice seems stuck
6. Prioritize tasks furthest from Alice's position

Core Rules:
1. Don't repeat "walk to" if already near target (distance < 2.0)
2. Can only hold one object at a time
3. Must grab before putting
4. Open containers before putting things inside
5. Use "put" for 'on' relations, "putin" for 'inside' relations
6. Object IDs must be integers

Respond with only the next action, no explanation needed."""

    def run(self, goal_spec: Dict) -> float:
        """Runs both agents simultaneously to achieve goals"""
        steps_taken = 0
        
        while steps_taken < self.max_steps:
            obs = self.get_observation()
            # wait for graph to update
            time.sleep(1)
            formatted_obs = self.format_observation(obs, goal_spec)
            
            if self.check_goal_reached(goal_spec):
                if self.debug:
                    print(f"Goal reached in {steps_taken} steps!")
                return 1.0
            
            prompt = formatted_obs['prompt'] + "\n" + self.format_goal(goal_spec)
            
            # Get actions from both agents simultaneously, including image in prompt
            _, alice_response = self.alice_llm(prompt, image=formatted_obs['image'], sys_msg=self.alice_system_prompt)
            _, bob_response = self.bob_llm(prompt, image=formatted_obs['image'], sys_msg=self.bob_system_prompt)
            
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
            time.sleep(1)  # Small delay between iterations
            
        if self.debug:
            print("Max steps reached without achieving goal")
        return 0.0

def run_experiment(goal_specs, api_key, api_option="openai", model="gpt-4o", debug=False):
    """Runs the collaborative agents to achieve goals"""
    # Initialize single agent that controls both Alice and Bob
    agent = Agent(
        agent_id=0,  # Main agent ID
        agent_name="Coordinator",
        api_key=api_key,
        api_option=api_option,
        model=model,
        debug=debug
    )
    
    # Standardize goal_specs format
    if isinstance(goal_specs, tuple):
        goal_specs = {goal_specs: 1}
    elif isinstance(goal_specs, dict):
        goal_specs = {tuple(k) if isinstance(k, tuple) else k: v for k, v in goal_specs.items()}
    
    # Run both agents together
    success = agent.run(goal_specs)
    
    print(f"Task completed successfully: {success}")
    return success

if __name__ == "__main__":
    # add a new agent
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

    make(data)
    set_action({'agent_index': [1], 'task': [6, 326, 0, 0]})

    goal_spec = {
        ('milk', 'on', 'table'): 1,
        ('apple', 'on', 'table2'): 1,
    }
    render(render_config)
    run_experiment(
        goal_specs=goal_spec,
        api_key="your_api_key",
        debug=True
    ) 