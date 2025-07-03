# ==========================================
# 0.  Shared utilities – **no more repeats**
# ==========================================
from typing import Dict, List, Any, Tuple
import os, time, json, base64, sys

# Append parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interface.client import observation, render, set_action, make
from llm import OpenAIBot
from llm_planner.utils import sequence

ALLOWED_ACTIONS   = ['walk to', 'grab', 'put', 'putin', 'open', 'close']
ALLOWED_RELATIONS = ['on', 'inside']
ALLOWED_STATES    = ['open', 'closed', 'switchon', 'switchoff']

# ---------- Agent Prompts ----------
ALICE_PROMPT = f"""You are Alice, the primary AI assistant completing household tasks. You control agent_0 using these actions: {', '.join(ALLOWED_ACTIONS)}.
Use format: agent_0 <action> object_<id>

Rules:
1. Don't repeat "walk to" if already near target (close to the target).
2. Can only hold one object at a time.
3. Must grab before putting.
4. Open containers before putting things inside.
5. Use "put" for 'on' relations and "putin" for 'inside' relations.
6. Object IDs must be integers.
7. Avoid repeating the same action on the same object multiple times.
8. When searching for items, methodically check different containers rather than checking the same one repeatedly.
9. If you don't find what you're looking for in one container, move on to a different one.
10. Coordinate with Bob (agent_1) by focusing on different areas than him.

When given a goal:
- Break it down into steps.
- Execute one step at a time.
- Verify each action's success.
- Remember what containers you've already searched and don't waste time rechecking them.

In the escape room scenario:
- Look for the keycard in various containers by opening them
- Once you find the keycard, grab it and use it on the card reader
- Stay methodical and avoid repetitive actions

Respond with only the next action, no explanation needed."""

BOB_PROMPT = f"""You are Bob, a helper AI assistant collaborating with Alice (Agent0) to complete tasks efficiently. You control agent_1 using these actions: {', '.join(ALLOWED_ACTIONS)}.
Use format: agent_1 <action> object_<id>

Collaboration Rules:
1. Observe Alice's actions and infer her objective.
2. Choose complementary tasks – don't target the same object as Alice.
3. When searching for items, focus on containers that Alice hasn't checked yet.
4. Coordinate movements to avoid collisions.
5. Help if Alice seems stuck.
6. Prioritize tasks furthest from Alice's position.
7. If you don't find what you're looking for in one container, move on to a different one.
8. Avoid repeating the same action on the same object multiple times.

Core Rules:
1. Don't repeat "walk to" if already close to target.
2. Can only hold one object at a time.
3. Must grab before putting.
4. Open containers before putting things inside.
5. Use "put" for 'on' relations and "putin" for 'inside' relations.
6. Object IDs must be integers.

In the escape room scenario:
- Focus on searching containers that Alice isn't checking
- When you find the keycard, grab it and use it on the card reader
- Stay methodical and explore new areas
- Remember what containers you've already searched

Respond with only the next action, no explanation needed."""

# ---------- single source of truth for model providers ----------
def get_llm(model_provider:str, model:str, api_key:str, base_url:str|None=None) -> OpenAIBot:
    provider_table = {
        "openai"   : dict(use_openai=True , base_url=None),
        "anthropic": dict(use_openai=False, base_url="https://api.anthropic.com/v1",
                          default="claude-3-opus-20240229"),
        "qwen"     : dict(use_openai=False, base_url="localhost:5000", default="qwen-max"),
        "deepseek" : dict(use_openai=False, base_url="localhost:5000", default="deepseek-coder"),
    }
    entry      = provider_table.get(model_provider, provider_table["openai"])
    model      = model or entry.get("default", "gpt-4o")
    base_url   = base_url or entry["base_url"]
    return OpenAIBot(model=model,
                     use_openai=entry["use_openai"],
                     api_key=api_key,
                     base_url=base_url,
                     llm_config={'max_token': 2048})

def safe_mkdir(path:str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def capture_screenshot(render_cfg:Dict, shot_dir:str)->str:
    render(render_cfg); time.sleep(1)
    shots=[f for f in os.listdir(shot_dir) if f.startswith("HighresScreenshot")]
    if not shots: return ""
    latest=max(shots,key=lambda f: os.path.getctime(os.path.join(shot_dir,f)))
    with open(os.path.join(shot_dir,latest),"rb") as im:
        return base64.b64encode(im.read()).decode()

def clean_llm_response(response: str) -> str:
    cleaned = response.strip()
    if '```' in cleaned:
        cleaned = cleaned.split('```')[1]
    lines = cleaned.splitlines()
    if lines and lines[0].lower().startswith('python'):
        lines = lines[1:]
    cleaned = "\n".join(lines)
    if 'agent_config =' in cleaned:
        cleaned = cleaned.split('agent_config =')[-1]
    return cleaned.strip()

# ---------- Action parsing helpers ----------
def parse_action(action: str):
    parts = action.strip().split()
    if len(parts) < 3:
        return None, None, None
    agent = parts[0]
    action_type = ' '.join(parts[1:-1])
    object_id = parts[-1]
    return agent, action_type, object_id

def action_signature(action: str):
    _, action_type, object_id = parse_action(action)
    return f"{action_type}_{object_id}" if action_type and object_id else None

def find_agent_node(obs: List[Dict], agent_id: int):
    return next(
        (node for node in obs 
         if node['name'] == "BP_ThirdPersonCharacter" and 
         any(f"BP_ThirdPersonCharacter{agent_id}" in rel for rel in node.get('relation', []))),
        None
    )

# ==========================================
# 1.  Environment Interface
# ==========================================
class EnvironmentInterface:
    def __init__(self, environment: str, screenshot_dir: str):
        self.environment = environment
        self.screenshot_dir = screenshot_dir
        self.clear_screenshots()

    def clear_screenshots(self) -> None:
        if os.path.exists(self.screenshot_dir):
            for file in os.listdir(self.screenshot_dir):
                if file.startswith("HighresScreenshot"):
                    os.remove(os.path.join(self.screenshot_dir, file))

    def get_observation(self) -> Dict:
        observation({"type": "full"})
        with open('graph.json', 'r') as f:
            data = json.load(f)
        return data[self.environment]

    def get_all_objects(self, obs: List[Dict]) -> List[Dict]:
        excluded_objects = [
            "Wall", "SpotLight", "SM_DownLight", "RugsRectanglex", "RectLight",
            "CameraActor", "RecastNavMeshDefault", "NavMeshBoundsVolume",
            "PostProcessVolume", "PlayerStart", "Floor_x",
            "WorldSettings", "Brush", "DefaultPhysicsVolume_",
            "GameplayDebuggerPlayerManager_", "ChaosDebugDrawActor",
            "BP_ThirdPersonGameMode_C_", "GameSession_", "ParticleEventManager_",
            "GameNetworkManager_", "GameStateBase_", "AbstractNavDataDefault",
            "BP_ThirdPersonPlayerController_C_", "PlayerState_", "PlayerCameraManager_",
            "HUD_", "GameplayDebuggerCategoryReplicator_", "API",
            "GroupActor_"
        ]
        return [
            {"name": obj['name'], "id": obj['id']} for obj in obs
            if not any(excluded.lower() in obj['name'].lower() for excluded in excluded_objects)
        ]

    def get_latest_screenshot(self, render_config: Dict) -> str:
        return capture_screenshot(render_config, self.screenshot_dir)

# ==========================================
# 2.  Agent Class
# ==========================================
class Agent:
    def __init__(self, agent_id:int, name:str, api_key:str,
                 model_provider:str="openai", model:str="gpt-4o",
                 debug:bool=False, environment:str="EscapeRoom1",
                 screenshot_dir:str=r"E:\environment-sourcecode\environment\Saved\Screenshots\WindowsEditor"):
        self.agent_id = agent_id
        self.agent_name = name
        self.debug = debug
        self.environment = environment
        self.screenshot_dir = screenshot_dir
        self.llm = get_llm(model_provider, model, api_key)
        self.system_prompt = ""
        self.current_goals: Dict[Any, Any] = {}
        self.completed_goals: set = set()

    def execute_action(self, action: str) -> None:
        if self.debug:
            print(f"{self.agent_name} executing: {action}")
        action = action.replace('.0', '')
        action_sequence = sequence([action])
        for action_dict in action_sequence:
            set_action(action_dict)
            time.sleep(1)

    def get_screenshot(self, render_config: Dict) -> str:
        return capture_screenshot(render_config, self.screenshot_dir)

    def format_observation(self, obs: List[Dict], goal_spec: Dict, render_config: Dict) -> Dict:
        default_observation = {
            'prompt': f"You are agent_{self.agent_id}. Your goal is to find the keycard and use it on the card reader to unlock the door. Look around the room carefully.",
            'image': ""
        }
        
        if not goal_spec:
            default_observation['image'] = self.get_screenshot(render_config)
            return default_observation
            
        env_interface = EnvironmentInterface(self.environment, self.screenshot_dir)
        all_objects = env_interface.get_all_objects(obs)
        
        subject, relation, target = list(goal_spec.keys())[0]
        
        subject_nodes = [node for node in obs if subject.lower() in node['name'].lower()]
        subject_node = subject_nodes[0] if subject_nodes else None
        
        if subject_node is None:
            subject_node = {'name': subject, 'id': -1, 'transform': ['X=0.000 Y=0.000 Z=0.000'], 'relation': [], 'state': []}
            
        if relation != 'state':
            target_nodes = [node for node in obs if target.lower() in node['name'].lower()]
            target_node = target_nodes[0] if target_nodes else None
            
            if target_node is None:
                target_node = {'name': target, 'id': -1, 'transform': ['X=0.000 Y=0.000 Z=0.000'], 'relation': [], 'state': []}
        else:
            target_node = {'name': target, 'id': -1, 'transform': ['X=0.000 Y=0.000 Z=0.000'], 'relation': [], 'state': []}

        characters = [node for node in obs if 'BP_ThirdPersonCharacter' in node['name']]
        num_agents = len(characters)
        
        if not characters:
            characters = [{'name': f'BP_ThirdPersonCharacter{i}', 'id': 100+i, 'transform': ['X=0.000 Y=0.000 Z=0.000'], 'relation': [], 'state': []} for i in range(2)]
            num_agents = len(characters)
        
        formatted = {
            'target': {
                'name': target_node['name'],
                'id': target_node['id'],
                'position': target_node['transform'][0] if target_node.get('transform') else 'X=0.000 Y=0.000 Z=0.000',
                'relations': target_node.get('relation', []),
                'state': target_node.get('state', [])
            },
            'subject': {
                'name': subject_node['name'],
                'id': subject_node['id'],
                'position': subject_node['transform'][0] if subject_node.get('transform') else 'X=0.000 Y=0.000 Z=0.000',
                'relations': subject_node.get('relation', []),
                'state': subject_node.get('state', [])
            }
        }
        
        count = goal_spec.get((subject, relation, target), 1)
        if count > 1:
            if subject_nodes and len(subject_nodes) > 1:
                formatted['multiple_subjects'] = [
                    {
                        'name': node['name'],
                        'id': node['id'],
                        'position': node['transform'][0] if node.get('transform') else 'X=0.000 Y=0.000 Z=0.000',
                        'relations': node.get('relation', []),
                        'state': node.get('state', [])
                    } for node in subject_nodes
                ]
            
            if relation != 'state' and target_nodes and len(target_nodes) > 1:
                formatted['multiple_targets'] = [
                    {
                        'name': node['name'],
                        'id': node['id'],
                        'position': node['transform'][0] if node.get('transform') else 'X=0.000 Y=0.000 Z=0.000',
                        'relations': node.get('relation', []),
                        'state': node.get('state', [])
                    } for node in target_nodes
                ]
        
        for i in range(num_agents):
            char = characters[i]
            formatted[f"{char['name']}"] = {
                'name': char['name'],
                'id': char['id'],
                'position': char['transform'][0] if char.get('transform') else 'X=0.000 Y=0.000 Z=0.000',
                'relations': char.get('relation', []),
                'state': char.get('state', [])
            }

        character_prompt = ""
        for i in range(num_agents):
            char = characters[i]
            character_prompt += (
                f"{char['name']}: {char['name']} "
                f"(id: {char['id']}) at {char['transform'][0] if char.get('transform') else 'X=0.000 Y=0.000 Z=0.000'} "
                f"relations: {', '.join(char.get('relation', []))} "
                f"state: {char.get('state', [])}\n"
            )

        multiple_objects_prompt = ""
        if count > 1:
            if 'multiple_subjects' in formatted:
                multiple_objects_prompt += f"\nMultiple {subject} objects found:\n"
                for idx, obj in enumerate(formatted['multiple_subjects']):
                    multiple_objects_prompt += (
                        f"{idx+1}. {obj['name']} (id: {obj['id']}) at {obj['position']} "
                        f"relations: {', '.join(obj['relations'])} state: {obj['state']}\n"
                    )
            
            if 'multiple_targets' in formatted:
                multiple_objects_prompt += f"\nMultiple {target} objects found:\n"
                for idx, obj in enumerate(formatted['multiple_targets']):
                    multiple_objects_prompt += (
                        f"{idx+1}. {obj['name']} (id: {obj['id']}) at {obj['position']} "
                        f"relations: {', '.join(obj['relations'])} state: {obj['state']}\n"
                    )

        formatted['prompt'] = (
            f"Current state:\n"
            f"Target: {formatted['target']['name']} (id: {formatted['target']['id']}) at {formatted['target']['position']} "
            f"relations: {', '.join(formatted['target']['relations'])} state: {formatted['target']['state']}\n"
            f"Subject: {formatted['subject']['name']} (id: {formatted['subject']['id']}) at {formatted['subject']['position']} "
            f"relations: {', '.join(formatted['subject']['relations'])} state: {formatted['subject']['state']}\n"
            f"{character_prompt}"
            f"{multiple_objects_prompt}"
        )

        formatted['image'] = self.get_screenshot(render_config)
        return formatted

    def check_goal_completion(self, obs: List[Dict], goal_state: Dict) -> List[Tuple]:
        print(f"\n=== Checking goal completion for agent {self.agent_id} ===")
        print(f"Goal state to check: {goal_state}")
        
        if not goal_state:
            print("No goals to check")
            return []
            
        completed_goals = []
        for (subject, relation, target), count in goal_state.items():
            print(f"Checking goal: ({subject}, {relation}, {target}) with count {count}")
            
            if relation == 'state':
                print(f"State goal - checking if {subject} is in state {target}")
                subject_nodes = [node for node in obs if subject.lower() in node['name'].lower()]
                print(f"Found {len(subject_nodes)} matching subject nodes")
                
                if not subject_nodes:
                    print(f"WARNING: No matching nodes found for subject '{subject}'")
                    continue
                    
                completed_count = sum(1 for node in subject_nodes if target in node['state'])
                print(f"Found {completed_count}/{count} instances in desired state")
                
                if completed_count >= count:
                    print(f"Goal COMPLETE: {subject} is in state {target}")
                    completed_goals.append((subject, relation, target))
                else:
                    print(f"Goal NOT complete: {subject} is not in state {target}")
                    for node in subject_nodes:
                        print(f"  Node {node['name']} (id: {node['id']}) states: {node['state']}")
            else:
                print(f"Relation goal - checking if {subject} is {relation} {target}")
                subject_nodes = [node for node in obs if subject.lower() in node['name'].lower()]
                print(f"Found {len(subject_nodes)} matching subject nodes")
                
                if not subject_nodes:
                    print(f"WARNING: No matching nodes found for subject '{subject}'")
                    continue
                    
                completed_count = 0
                for node in subject_nodes:
                    rels = node.get('relation', [])
                    relation_match = f"{relation}({target})"
                    if relation_match in rels:
                        completed_count += 1
                        print(f"  Found matching relation in node {node['name']} (id: {node['id']}): {relation_match}")
                    else:
                        print(f"  No matching relation in node {node['name']} (id: {node['id']}). Relations: {rels}")
                
                print(f"Found {completed_count}/{count} instances with desired relation")
                if completed_count >= count:
                    print(f"Goal COMPLETE: {subject} is {relation} {target}")
                    completed_goals.append((subject, relation, target))
                else:
                    print(f"Goal NOT complete: {subject} is not {relation} {target}")
        
        print(f"Completed goals: {completed_goals}")
        return completed_goals

    def get_goal_state_prompt(self, all_objects: List[Dict], user_prompt: str) -> str:
        return f"""Based on the image and the entire scene graph, determine the goal state you want to achieve given the user prompt: {user_prompt}

Goal states must be in one of these two formats:
1. (subject, relation, target): count
   Example: ('milk', 'on', 'table'): 1
   - Allowed relations: {ALLOWED_RELATIONS}
   - Use count > 1 when multiple instances of the same object need to be placed

2. (subject, 'state', state_type): count
   Example: ('refrigerator', 'state', 'open'): 1
   - Allowed states: {ALLOWED_STATES}

You can generate up to 3 goal states. Only use objects from this list:
{all_objects}

In the escape room scenario, consider these potential goals:
1. Find the keycard that is hidden in one of the containers
2. Use the keycard on the card reader to unlock the door
3. Open the door to escape

Ensure your goals are progressive and complementary. Explore multiple containers but don't get stuck checking the same container repeatedly.

Respond with only the goal state as a Python dictionary, no explanation needed.
"""

# ==========================================
# 3.  Multi-Agent Controller
# ==========================================
class MultiAgentController:
    def __init__(self, user_prompt: str, agents: Dict[int, Agent],
                 render_configs: Dict[int, Dict], max_steps: int = 20, debug: bool = False):
        self.user_prompt = user_prompt
        self.agents = agents
        self.render_configs = render_configs
        self.max_steps = max_steps
        self.debug = debug
        any_agent = next(iter(agents.values()))
        self.env_interface = EnvironmentInterface(any_agent.environment, any_agent.screenshot_dir)
        self.goal_timeouts = {}
        self.max_goal_time = 30
        
        self.metrics = {
            'total_instructions': 0,
            'correct_instructions': 0,
            'wrong_instructions': 0,
            'effective_instructions': 0,
            'missing_instructions': 0
        }
        
        self.agent_action_history = {agent_id: [] for agent_id in agents.keys()}
        self.max_history_len = 5
        self.max_repeats = 2

    def initialize_goals(self) -> None:
        print("\n=== Initializing goals for all agents ===")
        obs = self.env_interface.get_observation()
        all_objects = self.env_interface.get_all_objects(obs)
        
        for agent_id, agent in self.agents.items():
            print(f"\nInitializing goals for agent {agent_id} ({agent.agent_name})")
            try:
                goal_prompt = agent.get_goal_state_prompt(all_objects, self.user_prompt)
                print(f"[LLM Prompt] Goal generation prompt:\n{goal_prompt[:200]}...")
                
                screenshot = self.env_interface.get_latest_screenshot(self.render_configs[agent_id])
                _, analysis = agent.llm(goal_prompt, image=screenshot)
                print(f"[LLM Response] Goal analysis:\n{analysis}")
                
                cleaned_analysis = clean_llm_response(analysis)
                agent.current_goals = eval(cleaned_analysis)
                agent.completed_goals = set()
                print(f"Agent {agent_id} goals: {agent.current_goals}")
            except Exception as e:
                print(f"Error initializing goals for agent {agent_id}: {e}")
                agent.current_goals = {('BP_Card', 'on', 'BP_CardReader'): 1}
                agent.completed_goals = set()
                print(f"Set fallback goal for agent {agent_id}: {agent.current_goals}")

    def is_action_repetitive(self, agent_id: int, action: str) -> bool:
        if not action:
            return False
        action_sig = action_signature(action)
        if not action_sig:
            return False
        
        history = self.agent_action_history.get(agent_id, [])
        if len(history) < self.max_repeats:
            return False
            
        recent_actions = history[-self.max_repeats:]
        if all(a == action_sig for a in recent_actions):
            print(f"Agent {agent_id} is repeating action: {action_sig}")
            return True
        return False
        
    def get_alternative_action(self, agent_id: int, obs: List[Dict]) -> str:
        storage_objects = [obj for obj in obs if 'Storage' in obj.get('name', '')]
        door_objects = [obj for obj in obs if 'Door' in obj.get('name', '')]
        card_objects = [obj for obj in obs if 'Card' in obj.get('name', '')]
        
        recent_objects = []
        for act in self.agent_action_history.get(agent_id, []):
            if '_object_' in act:
                obj_id = act.split('_object_')[1]
                recent_objects.append(obj_id)
        
        if storage_objects:
            fresh_storages = [obj for obj in storage_objects 
                             if str(obj.get('id')) not in recent_objects]
            
            if fresh_storages:
                storage = fresh_storages[0]
                return f"agent_{agent_id} walk to object_{int(storage['id'])}"
            
        if 'open' not in self.agent_action_history.get(agent_id, [])[-3:]:
            if storage_objects:
                storage = storage_objects[0]
                return f"agent_{agent_id} open object_{int(storage['id'])}"
        
        if card_objects:
            card = card_objects[0]
            return f"agent_{agent_id} grab object_{int(card['id'])}"
            
        if door_objects:
            door = door_objects[0]
            return f"agent_{agent_id} walk to object_{int(door['id'])}"
            
        return f"agent_{agent_id} walk to object_107"

    def update_action_history(self, agent_id: int, action: str) -> None:
        if not action:
            return
        action_sig = action_signature(action)
        if not action_sig:
            return
        
        history = self.agent_action_history.get(agent_id, [])
        history.append(action_sig)
        
        if len(history) > self.max_history_len:
            history = history[-self.max_history_len:]
            
        self.agent_action_history[agent_id] = history

    def validate_instruction(self, action: str, obs: List[Dict]) -> Tuple[bool, bool]:
        agent, action_type, object_id = parse_action(action)
        if not all([agent, action_type, object_id]):
            return False, False
        
        if not agent.startswith('agent_') or not agent[6:].isdigit():
            return False, False
            
        agent_id = int(agent[6:])
        
        if not object_id.startswith('object_') or not object_id[7:].isdigit():
            return False, False
            
        object_id = float(object_id[7:])
        
        if action_type not in ALLOWED_ACTIONS:
            return False, False
            
        object_exists = any(obj.get('id') == object_id for obj in obs)
        if not object_exists:
            return False, False
            
        agent_node = find_agent_node(obs, agent_id)
        if not agent_node:
            return False, False
            
        if action_type == 'grab':
            if any('hand' in rel.lower() and 'empty' not in rel.lower() for rel in agent_node.get('relation', [])):
                return False, False
                
        return True, True

    def update_metrics(self, action: str, obs: List[Dict]) -> None:
        self.metrics['total_instructions'] += 1
        
        is_correct, is_effective = self.validate_instruction(action, obs)
        
        if is_correct:
            self.metrics['correct_instructions'] += 1
        else:
            self.metrics['wrong_instructions'] += 1
            
        if is_effective:
            self.metrics['effective_instructions'] += 1

    def get_metrics_report(self) -> Dict:
        total = max(1, self.metrics['total_instructions'])
        return {
            'instruction_correct_rate': self.metrics['correct_instructions'] / total,
            'instruction_error_rate': self.metrics['wrong_instructions'] / total,
            'instruction_effective_rate': self.metrics['effective_instructions'] / total,
            'raw_counts': self.metrics
        }

    def run(self) -> float:
        print("\n=== Starting MultiAgentController.run() ===")
        self.initialize_goals()
        steps_taken = 0
        agent_avoid_objects = {agent_id: set() for agent_id in self.agents.keys()}

        while steps_taken < self.max_steps:
            print(f"\n=== Step {steps_taken+1}/{self.max_steps} ===")
            obs = self.env_interface.get_observation()
            
            for agent_id, agent in self.agents.items():
                print(f"\n>>> Processing agent {agent_id} ({agent.agent_name})")
                
                for goal in agent.current_goals:
                    if (agent_id, goal) not in self.goal_timeouts:
                        self.goal_timeouts[(agent_id, goal)] = 0

                completed = agent.check_goal_completion(obs, agent.current_goals)
                if completed:
                    print(f"Completed goals: {completed}")
                
                for goal in completed:
                    if goal in agent.current_goals:
                        agent.completed_goals.add(goal)
                        del agent.current_goals[goal]
                        if (agent_id, goal) in self.goal_timeouts:
                            del self.goal_timeouts[(agent_id, goal)]

                stuck_goals = []
                for goal in agent.current_goals:
                    self.goal_timeouts[(agent_id, goal)] += 1
                    if self.goal_timeouts[(agent_id, goal)] >= self.max_goal_time:
                        stuck_goals.append(goal)
                        print(f"Agent {agent_id} stuck on goal {goal}, moving to next goal")

                for goal in stuck_goals:
                    del agent.current_goals[goal]
                    del self.goal_timeouts[(agent_id, goal)]

                if not agent.current_goals:
                    print(f"Agent {agent_id} has no current goals, generating new ones")
                    all_objects = self.env_interface.get_all_objects(obs)
                    goal_prompt = agent.get_goal_state_prompt(all_objects, self.user_prompt)
                    print(f"[LLM Prompt] Goal generation prompt:\n{goal_prompt[:200]}...")
                    screenshot = self.env_interface.get_latest_screenshot(self.render_configs[agent_id])
                    
                    _, analysis = agent.llm(goal_prompt, image=screenshot)
                    print(f"[LLM Response] Goal analysis:\n{analysis}")
                    
                    cleaned = clean_llm_response(analysis)
                    new_goals = eval(cleaned)
                    
                    agent.current_goals.update({
                        k: v for k, v in new_goals.items() if k not in agent.completed_goals
                    })
                    print(f"Updated goals: {agent.current_goals}")

                formatted_obs = agent.format_observation(obs, agent.current_goals, self.render_configs[agent_id])
                
                other_agent_ids = [a_id for a_id in self.agents.keys() if a_id != agent_id]
                other_agent_objects = []
                for other_id in other_agent_ids:
                    if other_id in self.agent_action_history and self.agent_action_history[other_id]:
                        last_action = self.agent_action_history[other_id][-1]
                        if '_object_' in last_action:
                            other_obj = last_action.split('_object_')[1]
                            other_agent_objects.append(other_obj)
                
                coordination_info = f"\nCoordinate with other agents: avoid objects {', '.join(other_agent_objects)}. " if other_agent_objects else ""
                
                avoid_info = ""
                if agent_avoid_objects[agent_id]:
                    avoid_info = f"\nAvoid these already checked objects: {', '.join(agent_avoid_objects[agent_id])}"
                
                prompt = (
                    formatted_obs['prompt'] +
                    f"\nTask: {self.user_prompt}" +
                    f"\nCurrent goals: {agent.current_goals}" +
                    f"\nCompleted goals: {agent.completed_goals}" +
                    coordination_info +
                    avoid_info +
                    f"\nIMPORTANT: If you've already checked a container and did not find the keycard, move on to a different container."
                )
                print(f"[LLM Prompt] Action generation prompt:\n{prompt[:200]}...")
                
                _, response = agent.llm(prompt, sys_msg=agent.system_prompt)
                print(f"[LLM Response] Action response: {response}")
                
                action = response.strip()
                
                if self.is_action_repetitive(agent_id, action):
                    print(f"Agent {agent_id} is stuck in a loop, generating alternative action")
                    
                    parts = action.strip().split()
                    if len(parts) > 2 and 'object_' in parts[-1]:
                        obj_id = parts[-1].replace('object_', '')
                        agent_avoid_objects[agent_id].add(obj_id)
                    
                    action = self.get_alternative_action(agent_id, obs)
                    print(f"Alternative action: {action}")
                
                self.update_action_history(agent_id, action)
                
                if action and action.startswith(f"agent_{agent_id}"):
                    self.update_metrics(action, obs)
                    print(f"Executing: {action}")
                    agent.execute_action(action)
                else:
                    print(f"WARNING: Invalid action format: {action}")
                    fallback_action = f"agent_{agent_id} walk to object_107"
                    print(f"Executing fallback action: {fallback_action}")
                    self.update_action_history(agent_id, fallback_action)
                    agent.execute_action(fallback_action)
            
            steps_taken += 1
            time.sleep(0.1)

        print("\n=== Run complete, max steps reached ===")
        metrics_report = self.get_metrics_report()
        print("Final Metrics:", json.dumps(metrics_report, indent=2))
            
        return self.get_metrics_report()['instruction_effective_rate']

# ==========================================
# 4.  Agent Configuration
# ==========================================
def configure_agents(planner_llm: OpenAIBot, user_prompt: str, render_config: Dict) -> Dict:
    print("=== Starting configure_agents ===")
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

    print("Getting observation...")
    observation({"type": "full"})
    print("Rendering...")
    render(render_config)
    time.sleep(0.1)
    screenshot_dir = render_config.get('screenshot_dir', r"E:\environment-sourcecode\environment\Saved\Screenshots\WindowsEditor")
    print(f"Screenshot dir: {screenshot_dir}")
    print("Capturing screenshot...")
    screenshot = capture_screenshot(render_config, screenshot_dir)
    print(f"Screenshot captured, length: {len(screenshot)}")

    print("Calling planner LLM...")
    try:
        _, config_response = planner_llm(
            f"Task: {user_prompt}\n\nDetermine the optimal agent configuration for this task.",
            sys_msg=system_prompt,
            image=screenshot
        )
        print("Raw config_response:", config_response)
    except Exception as e:
        print(f"Error calling planner LLM: {e}")
        print("Using fallback configuration...")
        # Return fallback configuration
        return {
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
    
    print("Cleaning LLM response...")
    cleaned = clean_llm_response(config_response)
    print("Cleaned response:", cleaned)
    
    print("Evaluating agent config...")
    agent_config = eval(cleaned)
    print("Agent config keys:", agent_config.keys())
    
    if 'num_agents' not in agent_config or 'agents' not in agent_config:
        print("WARNING: Required keys missing, using fallback config")
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
    
    print(f"Final agent config: num_agents={agent_config['num_agents']}, agent keys={list(agent_config['agents'].keys())}")
    return agent_config

# ==========================================
# 5.  Main Experiment Runner
# ==========================================
def run_multi_agent_experiment(
    user_prompt: str,
    api_key: str,
    model_provider: str = "openai",
    model: str = "gpt-4o",
    debug: bool = False,
    environment: str = "EscapeRoom1",
    screenshot_dir: str = r"E:\environment-sourcecode\environment\Saved\Screenshots\WindowsEditor",
    max_steps: int = 20
) -> dict:
    """Single-call entry: sets up env, agents, controller, returns metrics."""
    print("\n===  Escape-Room Experiment  ===")
    safe_mkdir(screenshot_dir)

    observation({"type": "full"}); time.sleep(.2)
    make({"environment": environment}); time.sleep(.2)
    render_cfg = {
        "render_pipeline":'raytracing', "camera_index":[0],
        "image_width":[1920], "image_height":[1080],
        "fps":[60], "fov":[90],
        "screenshot_dir":screenshot_dir
    }
    render(render_cfg)

    planner = get_llm(model_provider, model, api_key)
    agent_cfg = configure_agents(planner, user_prompt, render_cfg)

    agents, r_configs = {}, {}
    for aid, cfg in agent_cfg['agents'].items():
        agents[aid] = Agent(aid, cfg['name'], api_key,
                           model_provider, model, debug,
                           environment, screenshot_dir)
        agents[aid].system_prompt = cfg['system_prompt']
        r_configs[aid] = render_cfg

    controller = MultiAgentController(user_prompt, agents, r_configs,
                                    max_steps=max_steps, debug=debug)
    controller.run()
    metrics = controller.get_metrics_report()
    print("===  Finished  ===")
    print(json.dumps(metrics, indent=2))
    return metrics

# -------------------------------------------------
# Minimal call:
# metrics = run_multi_agent_experiment(
#              user_prompt="Find keycard and escape.",
#              api_key="sk-***" )

if __name__ == "__main__":
    # You need to replace this with a valid API key
    # Get your API key from: https://platform.openai.com/account/api-keys
    API_KEY = "YOUR_OPENAI_API_KEY_HERE"  # Replace with your actual API key
    
    try:
        run_multi_agent_experiment(
            user_prompt="I want 2 agents to collaborate to escape from the room, you need to look around the room to find the key to open the door, the key could be anywhere, if its in the container, you need to open the container first, the output of the two agents should strictly follow the format of the example agent_<agent_id> <action> object_<id>, the action should be one of the allowed actions, the object_<id> should be id(integer) of one of the objects in the room",
            api_key=API_KEY,
            debug=True,
            environment="EscapeRoom1",
            screenshot_dir=r"E:\environment-sourcecode\environment\Saved\Screenshots\WindowsEditor"
        )
    except Exception as e:
        print(f"Error running experiment: {e}")
        print("Please check your API key and make sure it's valid.")
        print("You can get a new API key from: https://platform.openai.com/account/api-keys")

