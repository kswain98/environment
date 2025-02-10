from typing import Dict, List, Tuple, Union
import json
import os


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import OpenAIBot
from llm_planner.utils import sequence 
from interface.client import *
import time
import openai

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
        debug: bool = False,
        alice_prompt: str = None,
        bob_prompt: str = None
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.max_steps = max_steps
        self.debug = debug
        self.current_state = self.get_observation()
        
        # Initialize two LLMs - one for Alice and one for Bob
        if api_option == "openai":
            self.alice_llm = OpenAIBot(model=model, use_openai=True, api_key=api_key, llm_config={'max_token': 2048})
            self.bob_llm = OpenAIBot(model=model, use_openai=True, api_key=api_key, llm_config={'max_token': 2048})
        else:
            self.alice_llm = OpenAIBot(base_url=base_url, model=model, llm_config={'max_token': 2048})
            self.bob_llm = OpenAIBot(base_url=base_url, model=model, llm_config={'max_token': 2048})
            
        self.alice_system_prompt = alice_prompt if alice_prompt else self._create_alice_prompt()
        self.bob_system_prompt = bob_prompt if bob_prompt else self._create_bob_prompt()

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


    def format_observation(self, obs: List[Dict], goal_spec: Dict) -> Dict:
        """Formats the observation for both agents"""
        subject, relation, target = list(goal_spec.keys())[0]
        
        # Find nodes for all relevant objects and agents
        subject_node = next((node for node in obs if subject.lower() in node['name'].lower()), None)
        target_node = next((node for node in obs if target.lower() in node['name'].lower()), None)
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
            if not alice_node: missing.append("Alice (agent0)")
            if not bob_node: missing.append("Bob (agent1)")
            raise ValueError(f"Missing objects in observation: {', '.join(missing)}")
        
        formatted = {
            'target': {
                'name': target_node['name'],
                'id': target_node['id'],
                'position': target_node['transform'][0],
                'relations': target_node.get('relation', []),
                'state': target_node['state']
            },
            'subject': {
                'name': subject_node['name'],
                'id': subject_node['id'],
                'position': subject_node['transform'][0],
                'relations': subject_node.get('relation', []),
                'state': subject_node['state']
            },
            'alice': {
                'name': alice_node['name'],
                'id': alice_node['id'],
                'position': alice_node['transform'][0],
                'relations': alice_node.get('relation', []),
                'state': alice_node['state']
            },
            'bob': {
                'name': bob_node['name'],
                'id': bob_node['id'],
                'position': bob_node['transform'][0],
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
            formatted_obs = self.format_observation(obs, goal_spec)
            
            if self.check_goal_reached(goal_spec):
                if self.debug:
                    print(f"Goal reached in {steps_taken} steps!")
                return 1.0
            
            prompt = formatted_obs['prompt'] + "\n" + self.format_goal(goal_spec)
            
            # Get actions from both agents simultaneously
            _, alice_response = self.alice_llm(prompt, sys_msg=self.alice_system_prompt)
            _, bob_response = self.bob_llm(prompt, sys_msg=self.bob_system_prompt)
            
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

def run_experiment(
    goal_specs, 
    api_key, 
    api_option="openai", 
    model="gpt-4o", 
    debug=False,
    alice_prompt=None,
    bob_prompt=None
):
    """Runs the collaborative agents to achieve goals"""
    # Initialize single agent that controls both Alice and Bob
    agent = Agent(
        agent_id=0,  # Main agent ID
        agent_name="Coordinator",
        api_key=api_key,
        api_option=api_option,
        model=model,
        debug=debug,
        alice_prompt=alice_prompt,
        bob_prompt=bob_prompt
    )
    observation({"type": "full"})
    time.sleep(0.5)
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

    data = {
        "environment": "WatchAndHelp1", 
    }

    make(data)
    set_action({'agent_index': [1], 'task': [6, 326, 0, 0]})

    goal_spec = {
        ('milk', 'on', 'table'): 1,
        ('apple', 'on', 'table2'): 1,
    }
    
    run_experiment(
        goal_specs=goal_spec,
        api_key="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",
        debug=True,
        alice_prompt=None,
        bob_prompt=None
    ) 