from typing import Dict, List, Tuple, Any
import json
import os
import sys
import time
import base64

# Append parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interface.client import observation, render, set_action, make
from llm import OpenAIBot
from llm_planner.utils import sequence

# -------------------------------
# Constants and Helper Functions
# -------------------------------

ALLOWED_ACTIONS = ['walk to', 'grab', 'put', 'putin', 'open', 'close']
ALLOWED_RELATIONS = ['on', 'inside']
ALLOWED_STATES = ['open', 'closed', 'switchon', 'switchoff']

ALICE_PROMPT = f"""You are Alice, the primary AI assistant completing household tasks. You control agent_0 using these actions: {', '.join(ALLOWED_ACTIONS)}.
Use format: agent_0 <action> object_<id>

Rules:
1. Don't repeat "walk to" if already near target (close to the target).
2. Can only hold one object at a time.
3. Must grab before putting.
4. Open containers before putting things inside.
5. Use "put" for 'on' relations and "putin" for 'inside' relations.
6. Object IDs must be integers.

When given a goal:
- Break it down into steps.
- Execute one step at a time.
- Verify each action's success.
- Coordinate with Bob (Agent1) who will help you.

Respond with only the next action, no explanation needed."""

BOB_PROMPT = f"""You are Bob, a helper AI assistant collaborating with Alice (Agent0) to complete tasks efficiently. You control agent_1 using these actions: {', '.join(ALLOWED_ACTIONS)}.
Use format: agent_1 <action> object_<id>

Collaboration Rules:
1. Observe Alice's actions and infer her objective.
2. Choose complementary tasks – don't target the same object as Alice.
3. If Alice is handling one goal object, focus on another.
4. Coordinate movements to avoid collisions.
5. Help if Alice seems stuck.
6. Prioritize tasks furthest from Alice's position.

Core Rules:
1. Don't repeat "walk to" if already close to target.
2. Can only hold one object at a time.
3. Must grab before putting.
4. Open containers before putting things inside.
5. Use "put" for 'on' relations and "putin" for 'inside' relations.
6. Object IDs must be integers.

Respond with only the next action, no explanation needed."""

def capture_screenshot(render_config: Dict, screenshot_dir: str) -> str:
    """
    Render the scene, wait a short time, and return the latest screenshot as a base64 string.
    """
    render(render_config)
    time.sleep(0.5)
    files = [f for f in os.listdir(screenshot_dir) if f.startswith("HighresScreenshot")]
    if not files:
        raise ValueError("No screenshot found after rendering")
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(screenshot_dir, x)))
    screenshot_path = os.path.join(screenshot_dir, latest_file)
    with open(screenshot_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def clean_llm_response(response: str) -> str:
    """
    Clean up LLM responses by removing markdown and variable assignments.
    """
    cleaned = response.strip()
    if '```' in cleaned:
        # Take content from first code block.
        cleaned = cleaned.split('```')[1]
    lines = cleaned.splitlines()
    # Remove any potential language identifiers and variable assignments.
    if lines and lines[0].lower().startswith('python'):
        lines = lines[1:]
    cleaned = "\n".join(lines)
    if 'agent_config =' in cleaned:
        cleaned = cleaned.split('agent_config =')[-1]
    return cleaned.strip()

# -------------------------------
# Environment Interface
# -------------------------------

class EnvironmentInterface:
    def __init__(self, environment: str, screenshot_dir: str):
        self.environment = environment
        self.screenshot_dir = screenshot_dir
        self.clear_screenshots()

    def clear_screenshots(self) -> None:
        if os.path.exists(self.screenshot_dir):
            for file in os.listdir(self.screenshot_dir):
                if file.startswith("HighresScreenshot"):
                    try:
                        os.remove(os.path.join(self.screenshot_dir, file))
                    except Exception as e:
                        print(f"Failed to remove {file}: {e}")

    def get_observation(self) -> Dict:
        observation({"type": "full"})
        with open('graph.json', 'r') as f:
            data = json.load(f)
        return data[self.environment]

    def get_all_objects(self, obs: List[Dict]) -> List[str]:
        excluded_objects = [
            "Wall", "SpotLight", "SM_DownLight", "RugsRectanglex", "RectLight",
            "CameraActor", "RecastNavMeshDefault", "NavMeshBoundsVolume",
            "PostProcessVolume", "PlayerStart", "Floor_x",
            # Adding system/debug/structural objects
            "WorldSettings", "Brush", "DefaultPhysicsVolume_",
            "GameplayDebuggerPlayerManager_", "ChaosDebugDrawActor",
            "BP_ThirdPersonGameMode_C_", "GameSession_", "ParticleEventManager_",
            "GameNetworkManager_", "GameStateBase_", "AbstractNavDataDefault",
            "BP_ThirdPersonPlayerController_C_", "PlayerState_", "PlayerCameraManager_",
            "HUD_", "GameplayDebuggerCategoryReplicator_", "API",
            "GroupActor_"  # Grouping actor
        ]
        return [
            obj['name'] for obj in obs
            if not any(excluded.lower() in obj['name'].lower() for excluded in excluded_objects)
        ]

    def get_latest_screenshot(self, render_config: Dict) -> str:
        return capture_screenshot(render_config, self.screenshot_dir)

# -------------------------------
# Agent Class
# -------------------------------

class Agent:
    """Represents an individual agent responsible for executing actions."""

    def __init__(
        self,
        agent_id: int,
        agent_name: str,
        api_key: str,
        api_option: str = "openai",
        model: str = "gpt-4v",
        base_url: str = "localhost:5000",
        debug: bool = False,
        environment: str = "WatchAndHelp1",
        screenshot_dir: str = r"E:\environment-sourcecode\environment\Saved\Screenshots\WindowsEditor",
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.api_key = api_key
        self.api_option = api_option
        self.model = model
        self.base_url = base_url
        self.debug = debug
        self.environment = environment
        self.screenshot_dir = screenshot_dir

        # Instantiate the LLM instance for this agent.
        if self.api_option == "openai":
            self.llm = OpenAIBot(model=self.model, use_openai=True, api_key=self.api_key, 
                                  llm_config={'max_token': 2048})
        else:
            self.llm = OpenAIBot(base_url=self.base_url, model=self.model, 
                                  llm_config={'max_token': 2048})

        # The system prompt will be set externally (e.g., via configuration).
        self.system_prompt = ""
        self.current_goals: Dict[Any, Any] = {}
        self.completed_goals: set = set()

    def execute_action(self, action: str) -> None:
        if self.debug:
            print(f"{self.agent_name} executing: {action}")
        # Preprocess action string if needed.
        action = action.replace('.0', '')
        action_sequence = sequence([action])
        for action_dict in action_sequence:
            set_action(action_dict)
            # Wait extra time for "grab" actions.
            #time.sleep(5 if 'grab' in action else 2)
            time.sleep(1)

    def get_screenshot(self, render_config: Dict) -> str:
        return capture_screenshot(render_config, self.screenshot_dir)

    def format_observation(self, obs: List[Dict], goal_spec: Dict, render_config: Dict) -> Dict:
        env_interface = EnvironmentInterface(self.environment, self.screenshot_dir)
        all_objects = env_interface.get_all_objects(obs)
        # Validate that each goal object exists
        for (subject, relation, target), _ in goal_spec.items():
            if relation != 'state':
                if subject not in all_objects:
                    raise ValueError(f"Goal state contains invalid subject: {subject}")
                if target not in all_objects:
                    raise ValueError(f"Goal state contains invalid target: {target}")

        # Find relevant nodes
        subject, relation, target = list(goal_spec.keys())[0]
        subject_node = next((node for node in obs if subject.lower() in node['name'].lower()), None)
        if relation != 'state':
            target_node = next((node for node in obs if target.lower() in node['name'].lower()), None)
        else:
            target_node = {'name': target, 'id': -1, 'transform': ['X=0.000 Y=0.000 Z=0.000'], 'relation': [], 'state': []}

        # Find character nodes
        characters = [node for node in obs if 'BP_ThirdPersonCharacter' in node['name']]
        num_agents = len(characters)
        
        # Create the formatted dictionary
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
            }
        }
        
        # Add character information
        for i in range(num_agents):
            formatted[f"{characters[i]['name']}"] = {
                'name': characters[i]['name'],
                'id': characters[i]['id'],
                'position': characters[i]['transform'][0],
                'relations': characters[i].get('relation', []),
                'state': characters[i]['state']
            }

        # Create character prompt
        character_prompt = ""
        for i in range(num_agents):
            character_prompt += (
                f"{characters[i]['name']}: {characters[i]['name']} "
                f"(id: {characters[i]['id']}) at {characters[i]['transform'][0]} "
                f"relations: {', '.join(characters[i].get('relation', []))} "
                f"state: {characters[i]['state']}\n"
            )

        formatted['prompt'] = (
            f"Current state:\n"
            f"Target: {formatted['target']['name']} (id: {formatted['target']['id']}) at {formatted['target']['position']} "
            f"relations: {', '.join(formatted['target']['relations'])} state: {formatted['target']['state']}\n"
            f"Subject: {formatted['subject']['name']} (id: {formatted['subject']['id']}) at {formatted['subject']['position']} "
            f"relations: {', '.join(formatted['subject']['relations'])} state: {formatted['subject']['state']}\n"
            f"{character_prompt}"
        )

        # Attach screenshot
        formatted['image'] = self.get_screenshot(render_config)
        return formatted

    def check_goal_completion(self, obs: List[Dict], goal_state: Dict) -> List[Tuple]:
        completed_goals = []
        for (subject, relation, target), _ in goal_state.items():
            if relation == 'state':
                subject_node = next((node for node in obs if subject.lower() in node['name'].lower()), None)
                if subject_node and target in subject_node['state']:
                    completed_goals.append((subject, relation, target))
            else:
                subject_node = next((node for node in obs if subject.lower() in node['name'].lower()), None)
                if subject_node:
                    rels = subject_node.get('relation', [])
                    if f"{relation}({target})" in rels:
                        completed_goals.append((subject, relation, target))
        return completed_goals

    def get_goal_state_prompt(self, all_objects: List[str], user_prompt: str) -> str:
        return f"""Based on the image and the entire scene graph, determine the goal state you want to achieve given the user prompt: {user_prompt}

Goal states must be in one of these two formats:
1. (subject, relation, target): count
   Example: ('milk', 'on', 'table'): 1
   - Allowed relations: {ALLOWED_RELATIONS}

2. (subject, 'state', state_type): count
   Example: ('refrigerator', 'state', 'open'): 1
   - Allowed states: {ALLOWED_STATES}

You can generate up to 3 goal states. Only use objects from this list:
{all_objects}

Respond with only the goal state as a Python dictionary, no explanation needed.
"""

# -------------------------------
# Multi-Agent Coordination
# -------------------------------

class MultiAgentController:
    def __init__(self, user_prompt: str, agents: Dict[int, Agent],
                 render_configs: Dict[int, Dict], max_steps: int = 20, debug: bool = False):
        self.user_prompt = user_prompt
        self.agents = agents
        self.render_configs = render_configs
        self.max_steps = max_steps
        self.debug = debug
        # Use the first agent's environment settings (assumes all agents share the same environment)
        any_agent = next(iter(agents.values()))
        self.env_interface = EnvironmentInterface(any_agent.environment, any_agent.screenshot_dir)
        self.goal_timeouts = {}  # Track time spent on each goal
        self.max_goal_time = 30  # Maximum steps to spend on a goal before moving on

    def initialize_goals(self) -> None:
        obs = self.env_interface.get_observation()
        all_objects = self.env_interface.get_all_objects(obs)
        for agent_id, agent in self.agents.items():
            goal_prompt = agent.get_goal_state_prompt(all_objects, self.user_prompt)
            screenshot = self.env_interface.get_latest_screenshot(self.render_configs[agent_id])
            _, analysis = agent.llm(goal_prompt, image=screenshot)
            print("analysis:", analysis)
            try:
                agent.current_goals = eval(clean_llm_response(analysis))
                agent.completed_goals = set()
            except Exception as e:
                print(f"Error parsing initial goal states for agent {agent_id}: {e}")
                agent.current_goals = {}
                agent.completed_goals = set()

    def run(self) -> float:
        self.initialize_goals()
        steps_taken = 0

        while steps_taken < self.max_steps:
            obs = self.env_interface.get_observation()
            time.sleep(0.1)

            for agent_id, agent in self.agents.items():
                # Initialize timeout tracking for new goals
                for goal in agent.current_goals:
                    if (agent_id, goal) not in self.goal_timeouts:
                        self.goal_timeouts[(agent_id, goal)] = 0

                # Check which goals have been completed
                completed = agent.check_goal_completion(obs, agent.current_goals)
                for goal in completed:
                    if goal in agent.current_goals:
                        agent.completed_goals.add(goal)
                        del agent.current_goals[goal]
                        if (agent_id, goal) in self.goal_timeouts:
                            del self.goal_timeouts[(agent_id, goal)]

                # Check for stuck goals and remove them
                stuck_goals = []
                for goal in agent.current_goals:
                    self.goal_timeouts[(agent_id, goal)] += 1
                    if self.goal_timeouts[(agent_id, goal)] >= self.max_goal_time:
                        stuck_goals.append(goal)
                        if self.debug:
                            print(f"Agent {agent_id} stuck on goal {goal}, moving to next goal")

                # Remove stuck goals and their timeouts
                for goal in stuck_goals:
                    del agent.current_goals[goal]
                    del self.goal_timeouts[(agent_id, goal)]

                # If no current goals, request new ones.
                if not agent.current_goals:
                    all_objects = self.env_interface.get_all_objects(obs)
                    goal_prompt = agent.get_goal_state_prompt(all_objects, self.user_prompt)
                    screenshot = self.env_interface.get_latest_screenshot(self.render_configs[agent_id])
                    _, analysis = agent.llm(goal_prompt, image=screenshot)
                    try:
                        new_goals = eval(clean_llm_response(analysis))
                        # Only add goals not already completed.
                        agent.current_goals.update({
                            k: v for k, v in new_goals.items() if k not in agent.completed_goals
                        })
                    except Exception as e:
                        print(f"Error updating goals for agent {agent_id}: {e}")

                formatted_obs = agent.format_observation(obs, agent.current_goals, self.render_configs[agent_id])
                prompt = (
                    formatted_obs['prompt'] +
                    f"\nTask: {self.user_prompt}" +
                    f"\nCurrent goals: {agent.current_goals}" +
                    f"\nCompleted goals: {agent.completed_goals}"
                )
                _, response = agent.llm(prompt, sys_msg=agent.system_prompt)
                action = response.strip()
                if action.startswith(f"agent_{agent_id}"):
                    if self.debug:
                        
                        print(f"Agent {agent_id} executing: {action}")
                    agent.execute_action(action)
            steps_taken += 1
            time.sleep(0.1)

        if self.debug:
            print("Max steps reached")
        return 0.0

# -------------------------------
# Agent Configuration
# -------------------------------

def configure_agents(planner_llm: OpenAIBot, user_prompt: str, render_config: Dict) -> Dict:
    """
    Use a planner LLM to determine the optimal agent configuration.
    Returns a configuration dictionary with the number of agents and
    system prompts for each agent.
    """
    system_prompt = f"""You are a high-level task orchestrator. Given a task, determine:
1. The optimal number of agents needed (1-4 agents)
2. Each agent's role and responsibilities
3. The specific prompt/instructions for each agent

Rules:
- Each agent must have a unique role and purpose.
- Agents should complement each other's abilities.
- Consider task complexity and spatial requirements.
- Maximum 4 agents allowed.
- Each agent requires:
  * A unique name.
  * A role description.
  * Allowed actions: {ALLOWED_ACTIONS}.
  * A system prompt similar to the following examples:

Example for Alice:
{ALICE_PROMPT}

Example for Bob:
{BOB_PROMPT}


Respond with only a Python dictionary containing the configuration.
Example output:

agent_config = {{
    'num_agents': 2,
    'agents': {{
        0: {{
            'name': 'Alice',
            'role': 'Primary task executor',
            'allowed_actions': {ALLOWED_ACTIONS},
            'system_prompt': "{ALICE_PROMPT}"
        }},
        1: {{
            'name': 'Bob',
            'role': 'Support assistant',
            'allowed_actions': {ALLOWED_ACTIONS},
            'system_prompt': "{BOB_PROMPT}"
        }}
    }}
}}
"""

    # Capture an initial screenshot.
    observation({"type": "full"})
    render(render_config)
    time.sleep(0.1)
    screenshot_dir = render_config.get('screenshot_dir', r"E:\environment-sourcecode\environment\Saved\Screenshots\WindowsEditor")
    screenshot = capture_screenshot(render_config, screenshot_dir)

    # Get configuration from the planner LLM.
    _, config_response = planner_llm(
        f"Task: {user_prompt}\n\nDetermine the optimal agent configuration for this task.",
        sys_msg=system_prompt,
        image=screenshot
    )
    print("config_response:", config_response)
    try:
        cleaned = clean_llm_response(config_response)
        agent_config = eval(cleaned)
    except Exception as e:
        print(f"Error parsing agent configuration: {e}")
        # Fallback default configuration.
        agent_config = {
            'num_agents': 2,
            'agents': {
                0: {
                    'name': 'Alice',
                    'role': 'Primary task executor',
                    'allowed_actions': ALLOWED_ACTIONS,
                    'system_prompt': ALICE_PROMPT
                },
                1: {
                    'name': 'Bob',
                    'role': 'Support assistant',
                    'allowed_actions': ALLOWED_ACTIONS,
                    'system_prompt': BOB_PROMPT
                }
            }
        }
    return agent_config

# -------------------------------
# Multi-Agent Experiment Runner
# -------------------------------

def run_multi_agent_experiment(
    user_prompt: str,
    api_key: str,
    api_option: str = "openai",
    model: str = "gpt-4o",
    debug: bool = False,
    environment: str = "WatchAndHelp1",
    screenshot_dir: str = r"E:\environment-sourcecode\environment\Saved\Screenshots\WindowsEditor"
) -> float:
    """
    Runs a configurable multi-agent system using LLM-based planning.
    """
    # Initialize the planner LLM.
    planner_llm = OpenAIBot(model=model, use_openai=True, api_key=api_key, 
                              llm_config={'max_token': 2048})

    render_config = {
        "render_pipeline": 'raytracing',
        "camera_index": [0],
        "image_width": [1920],
        "image_height": [1080],
        "fps": [60],
        "fov": [90],
        "screenshot_dir": screenshot_dir
    }

    agent_config = configure_agents(planner_llm, user_prompt, render_config)

    # Set an initial action for each agent (if needed).
    for i in range(agent_config['num_agents']):
        set_action({'agent_index': [i], 'task': [6, 326, 0, 0]})

    # Instantiate agents.
    agents: Dict[int, Agent] = {}
    render_configs: Dict[int, Dict] = {}
    for agent_id, config in agent_config['agents'].items():
        agent = Agent(
            agent_id=agent_id,
            agent_name=config['name'],
            api_key=api_key,
            api_option=api_option,
            model=model,
            debug=debug,
            environment=environment,
            screenshot_dir=screenshot_dir
        )
        # Use the configured system prompt.
        agent.system_prompt = config['system_prompt']
        agents[agent_id] = agent

        # Each agent gets its own render configuration (e.g., different camera index).
        render_configs[agent_id] = {
            "render_pipeline": 'raytracing',
            "camera_index": [agent_id],
            "image_width": [1920],
            "image_height": [1080],
            "fps": [60],
            "fov": [90],
            "screenshot_dir": screenshot_dir
        }

    # Create the multi-agent controller and run the experiment.
    controller = MultiAgentController(user_prompt, agents, render_configs, max_steps=20, debug=debug)
    success = controller.run()
    print(f"Task completed with {success} success rate")
    return success

# -------------------------------
# Main Entry Point
# -------------------------------

if __name__ == "__main__":
    render_config = {
        "render_pipeline": 'raytracing',
        "camera_index": [0],
        "image_width": [1920],
        "image_height": [1080],
        "fps": [60],
        "fov": [90],
        "screenshot_dir": r"E:\environment-sourcecode\environment\Saved\Screenshots\WindowsEditor"
    }
    data = {"environment": "EscapeRoom1"}
    observation({"type": "full"})
    time.sleep(0.1)
    make(data)

    user_prompt = """I want 3 agents to collaborate to escape from the room, you need to look around the room to find the key to open the door,
        the key could be anywhere, if its in the container, you need to open the container first, the output of the three agents should strictly follow the format of the example
        agent_<agent_id> <action> object_<id>, the action should be one of the allowed actions, the object_<id> should be id(integer) of one of the objects in the room"""
    render(render_config)
    run_multi_agent_experiment(
        user_prompt=user_prompt,
        api_key="your_api_key",
        debug=True,
        environment="EscapeRoom1",
        screenshot_dir=r"E:\environment-sourcecode\environment\Saved\Screenshots\WindowsEditor"
    )
