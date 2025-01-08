from typing import Dict, List, Tuple, Union
import json
import os
from llm import OpenAIBot
from utils import sequence
from client import *
import time

class LLMWatchAndHelp:
    """LLM-based agent for Watch and Help environment"""
    
    def __init__(
        self,
        agent_id: int,
        api_key: str = None,
        api_option: str = "other",
        model: str = "gpt-4",
        base_url: str = "localhost:5000",
        max_steps: int = 20,
        debug: bool = False
    ):
        self.agent_id = agent_id
        self.max_steps = max_steps
        self.debug = debug
        
        # Initialize LLM
        if api_option == "openai":
            self.llm = OpenAIBot(model=model, use_openai=True, api_key=api_key)
        else:
            self.llm = OpenAIBot(base_url=base_url, model=model, llm_config={'max_token': 2048})
            
        # Load object info
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(f'{dir_path}/dataset/object_info_small.json', 'r') as f:
            self.object_info = json.load(f)
        # System prompt for the LLM
        self.system_prompt = self._create_system_prompt()
        
    def _create_system_prompt(self) -> str:
        """Creates the system prompt for the LLM"""
        allowed_actions = ['walk to', 'grab', 'put', 'putin', 'open', 'close']
        
        prompt = f"""You are an AI assistant helping to complete household tasks. You can control an agent with ID {self.agent_id} 
using the following actions: {', '.join(allowed_actions)}.

For each action, use the format:
agent_{self.agent_id} <action> object_<id>

For example:
- agent_{self.agent_id} walk to object_1
- agent_{self.agent_id} grab object_2
- agent_{self.agent_id} put object_2 object_3

Important rules:
1. The agent must be next to an object to interact with it
2. The agent can only hold one object at a time
3. To put an object somewhere, the agent must be holding it first
4. Containers must be opened before putting objects inside

When given a goal, break it down into steps and execute them one at a time.
Analyze the environment after each action to confirm it worked as expected.

Please respond with only the next action to take, no additional explanation needed."""

        return prompt

    def get_observation(self) -> Dict:
        """Gets the current environment observation"""
        observation({"type": "full"})
        with open('graph.json', 'r') as f:
            data = json.load(f)
        return data['WatchAndHelp1']  # Using default environment name

    def execute_action(self, action: str) -> None:
        """Executes a single action in the environment"""
        if self.debug:
            print(f"Executing action: {action}")
        action_sequence = sequence([action])
        for action_dict in action_sequence:
            set_action(action_dict)
            time.sleep(5)  # Wait for action to complete

    def check_goal_reached(self, goal_spec: Dict) -> bool:
        """Checks if the goal has been reached"""
        current_state = self.get_observation()
        
        for (subject, relation_type, target), count in goal_spec.items():
            found = 0
            for node in current_state:
                if subject.lower() in node['name'].lower():
                    if relation_type == 'state':
                        if target in node['state']:
                            found += 1
                    else:
                        for relation in node.get('relation', []):
                            if f"{relation_type} {target}" in relation:
                                found += 1
            if found < count:
                return False
        return True

    def format_observation(self, obs: List[Dict]) -> str:
        """Formats the observation for the LLM"""
        formatted = "Current environment state:\n"
        for node in obs:
            formatted += f"Object {node['id']}: {node['name']}\n"
            if 'state' in node:
                formatted += f"  State: {', '.join(node['state'])}\n"
            if 'relation' in node:
                formatted += f"  Relations: {', '.join(node['relation'])}\n"
        return formatted

    def format_goal(self, goal_spec: Dict) -> str:
        """Formats the goal specification for the LLM"""
        formatted = "Goal:\n"
        for (subject, relation_type, target), count in goal_spec.items():
            formatted += f"Place {count}x {subject} {relation_type} {target}\n"
        return formatted

    def run(self, goal_spec: Dict) -> float:
        """
        Runs the LLM planner to achieve the specified goal
        
        Args:
            goal_spec: Dictionary mapping (subject, relation_type, target) to count
            
        Returns:
            float: 1.0 if goal achieved, 0.0 otherwise
        """
        steps_taken = 0
        
        while steps_taken < self.max_steps:
            # Get current observation
            obs = self.get_observation()
            
            # Check if goal is reached
            if self.check_goal_reached(goal_spec):
                if self.debug:
                    print(f"Goal reached in {steps_taken} steps!")
                return 1.0
                
            # Format current state and goal for LLM
            prompt = self.format_observation(obs) + "\n" + self.format_goal(goal_spec)
            
            # Get next action from LLM
            _, response = self.llm(prompt, sys_msg=self.system_prompt)
            
            # Extract and execute action
            action = response.strip()
            if not action.startswith(f"agent_{self.agent_id}"):
                if self.debug:
                    print(f"Invalid action format: {action}")
                continue
                
            self.execute_action(action)
            steps_taken += 1
            
        if self.debug:
            print("Max steps reached without achieving goal")
        return 0.0

def main():
    # Example usage
    agent = LLMWatchAndHelp(
        agent_id=0,
        api_key="your-api-key",  # If using OpenAI
        api_option="other",  # or "openai"
        model="gpt-4",
        debug=True
    )
    
    # Example goal: put a book inside a cabinet
    goal_spec = {("book", "inside", "cabinet"): 1}
    
    success = agent.run(goal_spec)
    print(f"Task completed successfully: {success}")

if __name__ == "__main__":
    main() 