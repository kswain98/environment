# agents.py

import json
import random
import copy
import numpy as np
from collections import defaultdict
from anytree import AnyNode as Node
from tqdm import tqdm
from client import *
import utils
import wandb
import time
import os
import threading
import torch

# Helper function
def clean_graph(graph, goal_spec, _):
    """Clean and filter the graph based on goal specification"""
    filtered_nodes = []
    
    # Extract all objects from the graph
    for node in graph:
        name = node["name"].lower()
        # Skip floors, walls, and game-specific objects
        if (not name.startswith("floor_") and 
            not name.startswith("wall_") and
            not name.startswith("bp_thirdperson")):
            filtered_nodes.append(node)
            
    return filtered_nodes

# Belief class
class Belief():
    def __init__(self, current_state, agent_id, prior=None, forget_rate=0.0, seed=None, rate=0.5, low_prob=0.001):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agent_id = agent_id
        self.current_state = copy.deepcopy(current_state)
        self.prior = prior
        self.forget_rate = forget_rate
        self.rate = rate
        self.low_prob = low_prob
        self.high_prob = 1e9

        self.container_restrictions = {
            'book': ['cabinet', 'kitchencabinet']
        }
        self.id_restrictions_inside = {}
        self.class_nodes_delete = ['wall', 'floor', 'ceiling', 'curtain', 'window']
        self.categories_delete = ['Doors']
        
        self.initialize_basic_beliefs()
        self.initialize_complex_beliefs()
        
        self.grabbed_object = []
        self.last_opened = None
        self.debug = False

    def initialize_basic_beliefs(self):
        self.id2node = {node['id']: node for node in self.current_state}
        self.node_beliefs = {}
        self.relation_beliefs = {}

    def initialize_complex_beliefs(self):
        self.relation_belief = {}
        self.first_belief = {}
        
        self.container_ids = []
        self.container_index_belief_dict = {}
        for idx, node in enumerate(self.current_state):
            if node['name'].startswith(('BP_Table', 'BP_SideTable', 'SM_TableDining')):
                self.container_ids.append(node['id'])
                self.container_index_belief_dict[node['id']] = idx

    def update_belief(self, observations):
        for node in observations:
            node_id = node['id']
            node_state = node.get('state', [])
            node_relations = node.get('relation', [])

            belief_states = self.node_beliefs.get(node_id, {})
            for state in ['ON', 'OFF', 'OPEN', 'CLOSED']:
                if state in node_state:
                    belief_states[state] = 1.0
                    opposite_state = 'OFF' if state == 'ON' else 'ON' if state == 'OFF' else 'CLOSED' if state == 'OPEN' else 'OPEN'
                    belief_states[opposite_state] = 0.0
            self.node_beliefs[node_id] = belief_states

            for relation_str in node_relations:
                parts = relation_str.split()
                if len(parts) >= 3:
                    relation_type = parts[1]
                    target_name = ' '.join(parts[2:])
                    target_node = next((n for n in self.current_state if n['name'] == target_name), None)
                    if target_node:
                        target_id = target_node['id']
                        self.relation_beliefs[(node_id, relation_type, target_id)] = 1.0

        for key in self.relation_beliefs:
            if key not in [(node['id'], relation.split()[0], next((n['id'] for n in self.current_state if n['name'] == ' '.join(relation.split()[1:])), None)) for node in observations for relation in node.get('relation', [])]:
                self.relation_beliefs[key] *= (1 - self.forget_rate)

    def sample_from_belief(self, as_vh_state=False):
        sampled_state = copy.deepcopy(self.current_state)

        for node in sampled_state:
            node_id = node['id']
            belief_states = self.node_beliefs.get(node_id, {})
            new_state = []

            for state in ['ON', 'OFF', 'OPEN', 'CLOSED']:
                if state in belief_states:
                    prob = belief_states[state]
                    if random.random() < prob:
                        new_state.append(state)

            node['state'] = new_state

        for node in sampled_state:
            node['relation'] = []

        for (node_id, relation_type, target_id), prob in self.relation_beliefs.items():
            if random.random() < prob:
                node = self.id2node.get(node_id)
                target_node = self.id2node.get(target_id)
                if node and target_node:
                    relation_str = f"{relation_type} {target_node['name']}"
                    node_sampled = next((n for n in sampled_state if n['id'] == node_id), None)
                    if node_sampled:
                        node_sampled['relation'].append(relation_str)

        return sampled_state

    def reset_belief(self):
        self.initialize_belief()

    def update(self, origin, final):
        dist_total = origin - final
        ratio = (1 - np.exp(-self.rate * np.abs(origin-final)))
        return origin - ratio * dist_total

    def reset_to_prior_if_invalid(self, belief_node):
        if belief_node[1].max() == self.low_prob:
            belief_node[1] = self.prior

    def update_to_prior(self):
        for node_name in self.relation_belief:
            self.relation_belief[node_name]['INSIDE'][1] = self.update(
                self.relation_belief[node_name]['INSIDE'][1], 
                self.first_belief[node_name]['INSIDE'][1]
            )

    def update_graph_from_gt_graph(self, gt_graph):
        ids_known_info = [[]]
        if isinstance(gt_graph, dict) and 'nodes' in gt_graph:
            gt_graph = gt_graph['nodes']
        elif not isinstance(gt_graph, list):
            print(f"Unexpected gt_graph type: {type(gt_graph)}")
            return
        
        id2node = {int(x['id']): x for x in gt_graph}
        inside = {}
        grabbed_object = []
        id_updated = []

        for node in gt_graph:
            for relation_str in node.get('relation', []):
                parts = relation_str.split()
                if len(parts) >= 3:
                    from_id = node['id']
                    relation_type = parts[1].upper()
                    target_name = ' '.join(parts[2:])
                    target_node = next((n for n in gt_graph if n['name'] == target_name), None)
                    
                    if target_node:
                        to_id = target_node['id']
                        if relation_type in ['HOLDS_LH', 'HOLDS_RH']:
                            grabbed_object.append(to_id)
                        if relation_type == 'INSIDE':
                            if from_id in inside:
                                print('Already inside', id2node[from_id]['class_name'],
                                      id2node[inside[from_id]]['class_name'],
                                      id2node[to_id]['class_name'])
                                raise Exception
                            inside[from_id] = to_id

        visible_ids = [x['id'] for x in gt_graph]
        
        for id_node in self.relation_belief.keys():
            id_updated.append(id_node)
            if id_node in grabbed_object:
                continue

            if id_node in visible_ids:
                if id_node in inside:
                    inside_obj = inside[id_node]
                    if inside_obj in self.container_index_belief_dict:
                        index_inside = self.container_index_belief_dict[inside_obj]
                        self.relation_belief[id_node]['INSIDE'][1][:] = self.low_prob
                        self.relation_belief[id_node]['INSIDE'][1][index_inside] = 1.

        for id_node in self.container_ids:
            if id_node in visible_ids and 'OPEN' in id2node[id_node]['state']:
                for id_node_child in self.relation_belief.keys():
                    if id_node_child not in inside.keys() or inside[id_node_child] != id_node:
                        ids_known_info[0].append(self.container_index_belief_dict[id_node])
                        self.relation_belief[id_node_child]['INSIDE'][1][self.container_index_belief_dict[id_node]] = self.low_prob

        mask_obj = np.ones(len(self.container_ids))
        if len(ids_known_info[0]):
            mask_obj[np.array(ids_known_info[0])] = 0
        mask_obj = (mask_obj == 1)

        for id_node in self.relation_belief.keys():
            if np.max(self.relation_belief[id_node]['INSIDE'][1]) == self.low_prob:
                self.relation_belief[id_node]['INSIDE'][1] = self.first_belief[id_node]['INSIDE'][1]

# Random_agent class
class Random_agent:
    """Random agent for graph-based environment"""
    def __init__(self, agent_id, char_index,
                 max_episode_length, num_simulation, max_rollout_steps, c_init, c_base, recursive=False,
                 num_samples=1, num_processes=1, comm=None, logging=False, logging_graphs=False, seed=None, env_name='WatchAndHelp1'):
        self.agent_type = 'Random'
        self.verbose = False
        self.recursive = recursive
        self.env_name = env_name
        
        if seed is None:
            seed = random.randint(0,100)
        self.seed = seed
        self.logging = logging
        self.logging_graphs = logging_graphs

        self.agent_id = agent_id
        self.char_index = char_index
        self.current_state = self.load_graph_json()
        self.belief = Belief(self.current_state, agent_id, prior=None, forget_rate=0.0, seed=None, rate=0.5, low_prob=0.001)
        self.max_episode_length = max_episode_length
        self.num_simulation = num_simulation
        self.max_rollout_steps = max_rollout_steps
        self.c_init = c_init
        self.c_base = c_base
        self.num_samples = num_samples
        self.num_processes = num_processes
        
        self.previous_belief_graph = None
        self.verbose = False
        self.last_opened = None
        self.comm = comm

        self.action_map = {
            'walktowards': 'walk to',
            'grab': 'grab',
            'put': 'put',
            'putin': 'putin',
            'open': 'open',
            'close': 'close'
        }

    def filtering_graph(self, graph):
        if not graph:
            return []
        
        for node in graph:
            if 'relation' in node:
                unique_relations = set()
                filtered_relations = []
                for relation in node['relation']:
                    if relation not in unique_relations:
                        unique_relations.add(relation)
                        filtered_relations.append(relation)
                node['relation'] = filtered_relations
        
        return graph

    def sample_belief(self, obs_graph):
        new_graph = self.belief.update_graph_from_gt_graph(obs_graph)
        if new_graph is None:
            new_graph = obs_graph or {"nodes": []}
        
        self.previous_belief_graph = self.filtering_graph(new_graph)
        return new_graph

    def get_relations_char(self, graph):
        char_node = next(
            (node for node in graph if 'character' in node['name'].lower()),
            None
        )
        if char_node:
            print('Character:')
            print(char_node.get('relation', []))
            print('---')

    def get_action(self, obs, goal_spec, opponent_subgoal=None):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(f'{dir_path}/dataset/object_info_small.json', 'r') as f:
            content = json.load(f)
        
        self.sample_belief(obs)
        observation({"type": "graph"})

        possible_actions = ['walktowards', 'grab', 'put', 'open']
        action_name = random.choice(possible_actions)
        print(f"Action: {action_name}")
        action_str = None
        if action_name == 'walktowards':
            objects = [
                    (node['name'], node['id']) for node in obs
                    if any(
                        isinstance(obj_type, str) and obj_type in node['name'].lower()
                        for obj_types in content.values()
                        for obj_type in (obj_types if isinstance(obj_types, list) else [obj_types])
                    ) and 'character' not in node['name'].lower()
                ]
        elif action_name == 'grab':
            objects = [
                (node['name'], node['id']) for node in obs
                if any (obj_type in node['name'].lower() for obj_type in content['objects_grab']) and 'character' not in node['name'].lower()
            ]
        elif action_name == 'put':
            objects = [
                (node['name'], node['id']) for node in obs
                if any (obj_type in node['name'].lower() for obj_type in content['objects_surface'] + content['objects_inside']) and 'character' not in node['name'].lower()
            ]
        elif action_name == 'open':
            objects = [
                (node['name'], node['id']) for node in obs
                if any (obj_type in node['name'].lower() for obj_type in content['objects_inside']) and 'character' not in node['name'].lower()
            ]
        print(f"Objects: {objects}")
        if len(objects) == 0:
            action_str = None
        else:
            selected_object = random.choice(objects)
            obj_name, obj_id = selected_object[0], selected_object[1]
            action_str = self.can_perform_action(action_name, obj_name, obj_id, self.agent_id, obs, teleport=False)

        info = {}
        if self.logging:
            info = {
                'plan': [action_str] if action_str else [],
                'belief': copy.deepcopy(self.belief.edge_belief),
                'belief_graph': copy.deepcopy(self.sim_env.vh_state.to_dict())
            }
            if self.logging_graphs:
                info['obs'] = obs['nodes']

        return action_str, info

    def reset(self, observed_graph, gt_graph, task_goal, seed=0, simulator_type='python', is_alice=False):
        self.last_action = None
        self.last_subgoal = None
        self.previous_belief_graph = None
        self.last_opened = None
        
        self.belief = Belief(gt_graph, agent_id=self.agent_id, seed=seed)
        self.belief.sample_from_belief()
        
        self.sample_belief(observed_graph)
        reset({"task_goal": task_goal})
        
    def can_perform_action(self, action, o1, o1_id, agent_id, graph, teleport=True):
        """Check if an action can be performed"""
        if action == 'no_action':
            return None

        id2node = {node['id']: node for node in graph}
        
        try:
            o1_id = int(float(o1_id))
        except (ValueError, TypeError):
            print(f"Invalid object ID format: {o1_id}")
            return None
        
        if o1_id not in id2node:
            print(f"Object ID {o1_id} not found in graph")
            return None
        
        # Get agent's current state
        grabbed_objects = []
        for node in graph:
            for relation in node.get('relation', []):
                if any(f"holds {hand}" in relation.lower() for hand in ["agent_0", "left hand", "right hand"]):
                    grabbed_objects.append(node['id'])

        if action == 'grab':
            if len(grabbed_objects) > 0:
                return None

        if action.startswith('walk'):
            if o1_id in grabbed_objects:
                return None

        if o1_id == agent_id:
            return None

        if action == 'open':
            if 'open' in id2node[o1_id].get('state', []) or 'closed' not in id2node[o1_id].get('state', []):
                return None
        if action == 'close':
            if 'closed' in id2node[o1_id].get('state', []) or 'open' not in id2node[o1_id].get('state', []):
                return None

        if 'put' in action:
            if len(grabbed_objects) == 0:
                return None
            else:
                o2_id = grabbed_objects[0]
                if o2_id == o1_id:
                    return None

        if action.startswith('put'):
            properties = id2node[o1_id].get('properties', [])
            if isinstance(properties, str):
                properties = [properties]
            
            if 'containers' in properties:
                action = 'putin'
            elif 'surfaces' in properties:
                action = 'putback'

        if action.startswith('walk') and teleport:
            action = 'walk'

        if action.startswith('walk'):
            return f"agent_{agent_id} walk to object_{o1_id}"
        elif action in ['put', 'putin', 'putback']:
            o2_id = grabbed_objects[0]
            return f"agent_{agent_id} put object_{o2_id} object_{o1_id}"
        elif action == 'grab':
            return f"agent_{agent_id} grab object_{o1_id}"
        else:
            return f"agent_{agent_id} {action} object_{o1_id}"

    def args_per_action(self, action):
        action_dict = {
            'turnleft': 0,
            'walkforward': 0,
            'turnright': 0,
            'walktowards': 1,
            'open': 1,
            'close': 1,
            'putback': 1,
            'putin': 1,
            'put': 1,
            'grab': 1,
            'no_action': 0,
            'walk': 1
        }
        return action_dict[action]

    def load_graph_json(self):
        observation({"type": "graph"})
        with open('graph.json', 'r') as f:
            data = json.load(f)
            return data[self.env_name]

    def run(self, goal_spec):
        """Run the random agent with proper object detection"""
        # Load and filter graph
        graph = self.load_graph_json()
        filtered_graph = clean_graph(graph, goal_spec, None)
        
        # Create mapping of normalized names to original names
        name_mapping = {}
        for node in filtered_graph:
            # Remove BP_ prefix and convert to lowercase for comparison
            normalized_name = node['name'].lower()
            if normalized_name.startswith('bp_'):
                normalized_name = normalized_name[3:]
            name_mapping[normalized_name] = node['name']
        
        print("Available objects:", [node['name'] for node in filtered_graph])
        
        # Check if goal objects exist using normalized names
        goal_objects = set()
        for (subject, _, target) in goal_spec.keys():
            goal_objects.add(subject.lower())
            if target not in ['table', 'floor', 'counter']:  # Common surfaces
                goal_objects.add(target.lower())
        
        available_objects = set(name_mapping.keys())
        for obj in goal_objects:
            if obj not in available_objects:
                print(f"Warning: Required object '{obj}' not found in environment!")
                print(f"Available objects: {available_objects}")
                return 0.0  # Return failure
        
        if isinstance(goal_spec, tuple):
            goal_spec = {goal_spec: 1}
        
        for i in range(self.max_episode_length):

            action_str, info = self.get_action(filtered_graph, goal_spec, None)
            print(f"Action: {action_str}")
            if action_str is None:
                continue
            action = utils.sequence([action_str])[0]
            print(f"Action: {action}")
            set_action(action)
            time.sleep(5)
            print("Action taken")
            observation({"type": "graph"})
            if self.check_goal_reached(goal_spec):
                print(f"Goal reached in {i} steps")
                break
        print("Goal not reached")

    def check_goal_reached(self, goal_spec):
        """
        Check if a specific goal has been reached.
        
        Args:
            goal_spec (tuple or dict): Goal specification
            
        Returns:
            bool: True if goal is reached, False otherwise
        """
        # Convert tuple to dictionary if needed
        if isinstance(goal_spec, tuple):
            goal_spec = {goal_spec: 1}
        
        for (subject, relation_type, target), count in goal_spec.items():
            for node in self.current_state:
                if subject.lower() in node['name'].lower():
                    if relation_type == 'state':
                        if target in node['state']:
                            return True
                    else:
                        for relation in node.get('relation', []):
                            if f"{subject} {relation_type} {target}" in relation:
                                return True
        return False

# Oracle class
class Oracle:
    def __init__(self, max_episode_length=100, env_name="WatchAndHelp1"):
        self.max_episode_length = max_episode_length
        self.num_steps = 0
        self.env_name = env_name
        self.last_action = None
        self.last_subgoal = None
        self.current_state = self.update_graph()
        self.belief = Belief(self.current_state, agent_id=0, seed=0)
    
    def update_graph(self):
        data = {"type": "graph"}
        observation(data)
        with open("graph.json", "r") as f:
            self.current_state = json.load(f)[self.env_name]
        return self.current_state

    def reset(self):
        self.num_steps = 0
        self.last_action = None
        self.last_subgoal = None
        with open("graph.json", "r") as f:
            graph = json.load(f)
        data = {"env_index": [0],"graph": graph}
        reset(data)
        
    def get_action(self, graph, task_goal):
        # Convert tuple to dictionary if needed
        if isinstance(task_goal, tuple):
            task_goal = {task_goal: 1}
        
        system_agent_action, system_agent_info = self.get_system_agent_action(
            graph,
            task_goal,
            self.last_action,
            self.last_subgoal
        )
        
        if system_agent_action is not None:
            self.last_action = system_agent_action
            if system_agent_info['subgoals']:
                self.last_subgoal = system_agent_info['subgoals'][0]
        
        action_str = f"{system_agent_action}" if system_agent_action else None
        
        reward, done, info = self._compute_reward(graph, task_goal)
        reward = torch.Tensor([reward])
        
        if self.num_steps >= self.max_episode_length:
            done = True
        done = np.array([done])
        
        return action_str, {
            'reward': reward,
            'done': done,
            'info': info,
            'subgoals': system_agent_info.get('subgoals', []),
            'plan': system_agent_info.get('plan', [])
        }

    

    def get_system_agent_action(self, graph, task_goal, last_action, last_subgoal):
        # Convert tuple to dictionary if needed
        if isinstance(task_goal, tuple):
            task_goal = {task_goal: 1}
        
        unsatisfied = {}
        for (subject, relation_type, target), count in task_goal.items():
            satisfied = self.check_progress({(subject, relation_type, target): count})
            if satisfied < count:
                unsatisfied[(subject, relation_type, target)] = count - satisfied

        if not unsatisfied:
            return None, {'subgoals': []}

        (subject, relation_type, target), count = next(iter(unsatisfied.items()))
        
        subject_nodes = [n for n in graph if subject.lower() in n['name'].lower()]
        target_nodes = [n for n in graph if target.lower() in n['name'].lower()]
        
        if not subject_nodes or not target_nodes:
            return None, {'subgoals': []}
        
        subject_node = subject_nodes[0]
        target_node = target_nodes[0]
        plan = []
        subgoal = None
        
        if relation_type == 'on':
            if not any('holds' in r for r in subject_node.get('relation', [])):
                plan = [
                    f"agent_0 walk to object_{int(subject_node['id'])}",
                    f"agent_0 grab object_{int(subject_node['id'])}"
                ]
                subgoal = f"grab_{subject}"
            else:
                plan = [
                    f"agent_0 walk to object_{int(target_node['id'])}",
                    f"agent_0 put object_{int(subject_node['id'])} object_{int(target_node['id'])}"
                ]
                subgoal = f"put_{subject}_{target}"
                
        elif relation_type == 'inside':
            if 'closed' in target_node.get('state', []):
                plan = [
                    f"agent_0 walk to object_{int(target_node['id'])}",
                    f"agent_0 open object_{int(target_node['id'])}"
                ]
                subgoal = f"open_{target}"
            elif not any('holds' in r for r in subject_node.get('relation', [])):
                plan = [
                    f"agent_0 walk to object_{int(subject_node['id'])}",
                    f"agent_0 grab object_{int(subject_node['id'])}"
                ]
                subgoal = f"grab_{subject}"
            else:
                plan = [
                    f"agent_0 walk to object_{int(target_node['id'])}",
                    f"agent_0 putin object_{int(subject_node['id'])} object_{int(target_node['id'])}"
                ]
                subgoal = f"putIn_{subject}_{target}"

        elif relation_type == 'state':
            if target == 'on':
                plan = [
                    f"agent_0 walk to object_{int(subject_node['id'])}",
                    f"agent_0 switchon object_{int(subject_node['id'])}"
                ]
                subgoal = f"switch on_{subject}"
            elif target == 'off':
                plan = [
                    f"agent_0 walk to object_{int(subject_node['id'])}",
                    f"agent_0 switchoff object_{int(subject_node['id'])}"
                ]
                subgoal = f"switch off_{subject}"

        action = plan[0] if plan else None
        return action, {
            'plan': plan,
            'subgoals': [subgoal] if subgoal else []
        }

    def _compute_reward(self, graph, task_goal):
        """
        Compute reward based on goal completion.
        
        Args:
            graph (dict): Current state graph
            task_goal (tuple or dict): Goal specification
            
        Returns:
            tuple: (reward, done, info)
        """
        # Convert tuple to dictionary if needed
        if isinstance(task_goal, tuple):
            task_goal = {task_goal: 1}
        
        total_goals = sum(count for _, count in task_goal.items())
        satisfied = sum(self.check_progress({goal: count}) for goal, count in task_goal.items())
        
        reward = 1.0 if satisfied >= total_goals else 0.0
        done = satisfied >= total_goals
        
        return reward, done, {
            'satisfied': satisfied,
            'total': total_goals
        }

    def check_progress(self, goal_spec):
        # Convert tuple to dictionary if needed
        if isinstance(goal_spec, tuple):
            goal_spec = {goal_spec: 1}
        
        satisfied_count = 0
        
        for (subject, relation_type, target), count in goal_spec.items():
            if relation_type == 'state':
                for node in self.current_state:
                    if subject.lower() in node['name'].lower() and target.lower() in node['name'].lower():
                        satisfied_count += 1
                        break
            else:
                for node in self.current_state:
                    if subject.lower() in node['name'].lower():
                        for relation in node.get('relation', []):
                            parts = relation.split()
                            if (len(parts) >= 3 and
                                relation_type == ' '.join(parts[1:-1]) and
                                target.lower() in parts[-1].lower()):
                                satisfied_count += 1
                                break

        return satisfied_count

    def sample_belief(self, obs_graph):
        new_graph = self.belief.update_graph_from_gt_graph(obs_graph)
        if new_graph is None:
            new_graph = obs_graph or {"nodes": []}
        self.previous_belief_graph = self.filtering_graph(new_graph)
        return new_graph

    def filtering_graph(self, graph):
        if not graph:
            return []
        
        for node in graph:
            if 'relation' in node:
                unique_relations = set()
                filtered_relations = []
                for relation in node['relation']:
                    if relation not in unique_relations:
                        unique_relations.add(relation)
                        filtered_relations.append(relation)
                node['relation'] = filtered_relations
        
        return graph
    def execute_task(self, goal_specifications):
        """
        Execute a task with multiple goal specifications.
        
        Args:
            goal_specifications (list): List of goal specifications to achieve
            
        Returns:
            dict: Metrics about the execution
        """
        initial_graph = self.update_graph()
        if not initial_graph:
            raise RuntimeError("Failed to get initial state")

        metrics = {
            "total_actions": 0,
            "achieved_subgoals": 0,
            "execution_success": False,
            "goals_achieved": 0,
            "total_goals": len(goal_specifications)
        }
        num_steps = 0
        
        # Track progress for each goal specification
        for goal_idx, goal_spec in enumerate(goal_specifications):
            print(f"\nExecuting goal {goal_idx + 1}/{len(goal_specifications)}: {goal_spec}")
            
            action_str, info = self.get_action(initial_graph, goal_spec)
            
            # Update metrics for this goal
            goal_metrics = {
                "actions": len(info['plan']) if info['plan'] else 0,
                "subgoals": len(info['subgoals']) if info['subgoals'] else 0,
                "success": info['plan'] is not None and len(info['plan']) > 0
            }
            
            metrics["total_actions"] += goal_metrics["actions"]
            metrics["achieved_subgoals"] += goal_metrics["subgoals"]
            
            while not info['done']: 
                if info['plan']:
                    action_sequence = utils.sequence(info['plan'])
                    for i, action_dict in enumerate(action_sequence):
                        self.update_graph()
                        print(f"Executing action {i+1}/{len(action_sequence)}: {action_dict}")
                        set_action(action_dict)
                        time.sleep(5)
                        num_steps += 1  
                        reward, done, action_info = self._compute_reward(self.current_state, goal_spec)
                        
                        if num_steps >= self.max_episode_length:
                            info['done'] = True
                        if self.check_goal_reached(goal_spec):  # Changed to self.check_goal_reached
                            metrics["goals_achieved"] += 1
                            print(f"Goal {goal_idx + 1} achieved!")
                            break
            
                # Update metrics for this goal attempt
                goal_metrics.update({
                    "reward": reward,
                    "steps_taken": i + 1,
                    "achieved": done
                })
                
                # Add goal-specific metrics to overall metrics
                metrics[f"goal_{goal_idx+1}"] = goal_metrics

        # Calculate overall success
        metrics["execution_success"] = metrics["goals_achieved"] == metrics["total_goals"]
        metrics["success_rate"] = metrics["goals_achieved"] / metrics["total_goals"]

        return metrics

    def check_goal_reached(self, goal_spec):
        """
        Check if a specific goal has been reached.
        
        Args:
            goal_spec (tuple or dict): Goal specification
            
        Returns:
            bool: True if goal is reached, False otherwise
        """
        # Convert tuple to dictionary if needed
        if isinstance(goal_spec, tuple):
            goal_spec = {goal_spec: 1}
        
        for (subject, relation_type, target), count in goal_spec.items():
            for node in self.current_state:
                if subject.lower() in node['name'].lower():
                    if relation_type == 'state':
                        if target in node['state']:
                            return True
                    else:
                        for relation in node.get('relation', []):
                            if f"{subject} {relation_type} {target}" in relation:
                                return True
        return False

class MCTS:
    def __init__(self, agent_id, max_episode_length, num_simulation, max_rollout_step, c_init, c_base, seed=1 , env_name = 'Kitchen'):
        """
        Initializes the MCTS algorithm with the given parameters.

        Args:
            agent_id (int): Identifier for the agent.
            max_episode_length (int): Maximum length of an episode.
            num_simulation (int): Number of simulations to run per MCTS iteration.
            max_rollout_step (int): Maximum steps during rollout.
            c_init (float): Initial exploration constant.
            c_base (float): Base exploration constant.
            seed (int): Random seed for reproducibility.
            env_name (str): Name of the environment.
        """
        # Initialize wandb
        wandb.init(
            project="mcts-planning",
            config={
                "agent_id": agent_id,
                "max_episode_length": max_episode_length,
                "num_simulation": num_simulation,
                "max_rollout_step": max_rollout_step,
                "c_init": c_init,
                "c_base": c_base,
                "seed": seed,
                "env_name": env_name,
                "discount": 1.0
            }
        )
        
        # Get initial observation before loading graph
        observation_request = {"type": "full"}
        observation(observation_request)
        
        self.discount = 1.0  # Discount factor for future rewards; consider tuning this parameter
        self.agent_id = agent_id
        self.max_episode_length = max_episode_length
        self.num_simulation = num_simulation
        self.max_rollout_step = max_rollout_step
        self.c_init = c_init 
        self.c_base = c_base
        self.seed = seed
        self.heuristic_dict = None
        self.last_opened = None
        self.verbose = 1
        self.env_name = env_name

        # Set random seeds for reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # Load and map the graph data
        self.graph = self.load_graph_json()
        self.node_map = self.create_node_map()
        self.relation_map = self.create_relation_map()
        self.relation_list = ["on", "inside", "next to"]
       
        
        # Initialize state tracking
        self.current_state = None
        self.last_observation = None

        # Add this at class level (right after class MCTS definition)
        self.file_lock = threading.Lock()

    def load_graph_json(self):
        """
        Loads the graph data from the JSON file.

        Returns:
            list: List of node dictionaries from the graph.
        """
        with open('graph.json', 'r') as f:
            return json.load(f)[self.env_name]

    def create_node_map(self):
        """
        Creates a mapping from node IDs to node data for quick access.

        Returns:
            dict: Mapping of node ID to node data.
        """
        return {node['id']: node for node in self.graph}

    def create_relation_map(self):
        """
        Creates a mapping of relations between objects.

        Returns:
            defaultdict: Mapping of relation types to lists of (subject, object) pairs.
        """
        relation_map = defaultdict(list)
        for node in self.graph:
            for relation in node.get('relation', []):
                parts = relation.split()
                if len(parts) >= 3:
                    subject = parts[0]
                    relation_type = ' '.join(parts[1:-1])  # e.g., "on", "inside"
                    target = parts[-1]
                    relation_map[relation_type].append((subject, target))
        return relation_map

    def check_progress(self, goal_spec):
        """
        Checks progress towards goals based on current state.

        Args:
            goal_spec (dict): Dictionary of goal specifications in format {(subject, relation_type, target): count}.

        Returns:
            int: Number of satisfied goal conditions.
        """
        satisfied_count = 0
        
        for (subject, relation_type, target), count in goal_spec.items():
            # Handle state-based goals
            if relation_type == 'state':
                for node in self.current_state:
                    if subject.lower() in node['name'].lower() and target.lower() in node['name'].lower():
                        satisfied_count += 1
                        break
            # Handle relation-based goals
            else:
                for node in self.current_state:
                    if subject.lower() in node['name'].lower():
                        for relation in node.get('relation', []):
                            parts = relation.split()
                            # Check if relation matches the goal
                            if (len(parts) >= 3 and
                                relation_type == ' '.join(parts[1:-1]) and
                                target.lower() in parts[-1].lower()):
                                satisfied_count += 1
                                break

        return satisfied_count

    def get_subgoal_space(self, satisfied, unsatisfied, verbose=0):
        """
        Get subgoal space based on current state and unsatisfied predicates.
        
        Args:
            satisfied (dict): Dictionary of satisfied predicates
            unsatisfied (dict): Dictionary of unsatisfied predicates
            verbose (int): Verbosity level
            
        Returns:
            list: List of possible subgoals in format [subgoal_action, predicate, tmp_predicate]
        """
        if not self.current_state:
            return []

        # Get observable objects and objects being held
        inhand_objects = []
        for node in self.current_state:
            for relation in node.get('relation', []):
                if f"holds agent_{self.agent_id}" in relation:
                    inhand_objects.append(node['id'])

        subgoal_space = []
        
        # Process each unsatisfied predicate
        for predicate, count in unsatisfied.items():
            if not isinstance(predicate, tuple) or len(predicate) != 3:
                continue
                
            obj, relation_type, target = predicate
            
            # Find the first matching object and target instead of all possibilities
            matching_node = next((node for node in self.current_state 
                                if obj in node['name'].lower() and 
                                not any(f"{relation_type} object_" in r for r in node.get('relation', []))), 
                               None)
            target_node = next((node for node in self.current_state 
                              if target in node['name'].lower()), 
                             None)
            
            if matching_node and target_node:
                if relation_type == 'on':
                    subgoal_type = 'put'
                    tmp_predicate = f"on_{matching_node['id']}_{target_node['id']}"
                    if tmp_predicate not in satisfied.get(predicate, []):
                        subgoal = f"{subgoal_type}_{matching_node['id']}_{target_node['id']}"
                        subgoal_entry = [subgoal, predicate, tmp_predicate]
                        
                        if matching_node['id'] in inhand_objects:
                            return [subgoal_entry]
                        subgoal_space.append(subgoal_entry)

                # Similar modifications for other relation types...
                elif relation_type == 'inside':
                    subgoal_type = 'putIn'
                    tmp_predicate = f"inside_{matching_node['id']}_{target_node['id']}"
                    if tmp_predicate not in satisfied.get(predicate, []):
                        subgoal = f"{subgoal_type}_{matching_node['id']}_{target_node['id']}"
                        subgoal_entry = [subgoal, predicate, tmp_predicate]
                        subgoal_space.append(subgoal_entry)
            elif relation_type == 'state':
                if target == 'ON':
                    subgoal_type = 'switch on'
                elif target == 'OFF':
                    subgoal_type = 'switch off'
                elif target == 'OPEN':
                    subgoal_type = 'open'
                elif target == 'CLOSED':
                    subgoal_type = 'close'
                else:
                    continue

                matching_nodes = [node for node in self.current_state if obj in node['name'].lower()]
                for node in matching_nodes:
                    tmp_predicate = f"state_{node['id']}_{target}"
                    if tmp_predicate not in satisfied.get(predicate, []):
                        subgoal = f"{subgoal_type}_{node['id']}"
                        subgoal_space.append([subgoal, predicate, tmp_predicate])

            elif relation_type == 'holds':
                if int(target) == self.agent_id:
                    subgoal_type = 'grab'
                    matching_nodes = [node for node in self.current_state if obj in node['name'].lower()]
                    for node in matching_nodes:
                        tmp_predicate = f"holds_{node['id']}_{self.agent_id}"
                        if tmp_predicate not in satisfied.get(predicate, []):
                            subgoal = f"{subgoal_type}_{node['id']}"
                            subgoal_space.append([subgoal, predicate, tmp_predicate])

            elif relation_type == 'sit':
                if int(obj) == self.agent_id:
                    subgoal_type = 'sit'
                    target_nodes = [node for node in self.current_state if target in node['name'].lower()]
                    for node in target_nodes:
                        tmp_predicate = f"sit_{self.agent_id}_{node['id']}"
                        if tmp_predicate not in satisfied.get(predicate, []):
                            subgoal = f"{subgoal_type}_{node['id']}"
                            subgoal_space.append([subgoal, predicate, tmp_predicate])
        if verbose:
            print(f"Generated {len(subgoal_space)} subgoals")
            if subgoal_space:
                print("Sample subgoals:", subgoal_space[:3])

        return subgoal_space 
    def relation_to_subgoal(self, relation):
        """
        Converts a relation string to a subgoal action.

        Args:
            relation (str): Relation string from state.

        Returns:
            str or None: Subgoal action string if applicable.
        """
        parts = relation.split()
        if len(parts) < 3:
            return None

        subject = parts[0]
        relation_type = ' '.join(parts[1:-1])
        target = parts[-1]

        # Map relation types to actions
        action_map = {
            'on': 'put',
            'inside': 'putIn',
            'holds': 'grab',
            'sit': 'sit',
            'state': 'switch'
        }

        action = action_map.get(relation_type)
        if action:
            return f"{action}_{subject}_{target}"
        return None

    def update_state(self):
        """Updates the current state using the observation function."""
        max_retries = 5
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                # Request new observation first
                observation_request = {"type": "full"}
                observation(observation_request)
                
                # Check if file exists
                if not os.path.exists("graph.json"):
                    continue
                
                # Use file lock when reading
                with self.file_lock:
                    with open("graph.json", "r") as f:
                        content = f.read().strip()
                    
                    # Parse JSON silently - remove error output
                    try:
                        new_state = json.loads(content)
                        if self.env_name in new_state:  # Verify environment exists in state
                            self.last_observation = new_state
                            self.current_state = new_state[self.env_name]
                            self.node_map = self.create_node_map()
                            self.relation_map = self.create_relation_map()
                            return True
                        else:
                            continue
                    
                    except json.JSONDecodeError:
                        continue
            
            except Exception:
                if attempt == max_retries - 1:
                    return False
            
        return False

    def run(self, curr_root, t, heuristic_dict, last_subgoal):
        """
        Executes the main MCTS algorithm, performing simulations and selecting the best action sequence.
        """
        # Update state before running MCTS
        if not self.update_state():
            print("Failed to update state")
            return None, [], []
            
        self.heuristic_dict = heuristic_dict
        if self.verbose:
            print('Checking subgoals...')

        # Unpack current root node's state
        curr_state_tmp, _, satisfied, unsatisfied, _, actions_parent = curr_root.id[1]
        
        # Generate subgoals based on the current state
        subgoals = self.get_subgoal_space(satisfied, unsatisfied)
        print(f"subgoals: {subgoals}")
        
        if self.verbose:
            print(f'satisfied: {satisfied}')
            print(f'unsatisfied: {unsatisfied}')
            print(f'subgoals: {subgoals}')
            print(f'last_subgoal: {last_subgoal}')

        # Retrieve objects currently held by the agent
        inhand_objs = self.get_inhand_objects()
        
        # Calculate the number of needed objects based on unsatisfied predicates
        needed_obj_count = self.calculate_needed_objects(unsatisfied, inhand_objs)

        # Determine objects that need to be placed inside containers
        remained_to_put = self.calculate_remained_to_put(unsatisfied)

        # Determine if a container needs to be closed
        need_to_close = self.check_need_to_close(remained_to_put)

        # Initialize complete plan and all subgoals
        complete_plan = []
        all_subgoals = []

        # Handle planning for all subgoals
        for subgoal in subgoals:
            # Generate plan for current subgoal
            plan = self.generate_plan(subgoal, need_to_close)
            if plan:
                complete_plan.extend(plan)
                all_subgoals.append(subgoal[0])  # Add the subgoal identifier
                continue

        # If we have a complete plan, return it
        if complete_plan:
            return None, complete_plan, all_subgoals

        # Check if agent is holding any objects
        agent_name = f"agent_{self.agent_id}"
        print(f"agent_name: {agent_name}")
        holding_object = False
        for node in self.current_state:
            for relation in node.get('relation', []):
                if f"holds {agent_name}" in relation:
                    holding_object = True
                    plan = self.generate_hold_plan(last_subgoal, need_to_close)
                    if plan:
                        return None, plan, [last_subgoal]

        # If no direct plans were found, perform MCTS simulations
        if not curr_root.is_expanded:
            curr_root = self.expand(curr_root, t)

        # Perform MCTS simulations with progress tracking
        simulation_rewards = []
        for explore_step in tqdm(range(self.num_simulation), disable=not self.verbose):
            node_path = self.perform_simulation(curr_root, t)
            if node_path:
                value = self.rollout(node_path[-1], t)
                discount_factor = self.discount ** node_path[-2].id[1][-2] if len(node_path) > 1 else 1.0
                final_value = value * discount_factor
                simulation_rewards.append(final_value)
                self.backup(final_value, node_path)

        # Log simulation metrics
        wandb.log({
            "timestep": t,
            "mean_simulation_reward": np.mean(simulation_rewards) if simulation_rewards else 0,
            "max_simulation_reward": np.max(simulation_rewards) if simulation_rewards else 0,
            "min_simulation_reward": np.min(simulation_rewards) if simulation_rewards else 0,
            "num_simulations_completed": len(simulation_rewards),
            "tree_depth": len(node_path) if node_path else 0
        })

        # Select the next root node based on visited children
        next_root, plan, subgoals = self.select_best_plan(curr_root, need_to_close)

        if self.verbose and plan:
            print(plan[0])

        return next_root, plan, subgoals

    def get_inhand_objects(self):
        """
        Retrieves the list of objects currently held by the agent.

        Args:
            state (dict): Current state of the environment.

        Returns:
            list: Names of objects being held by the agent.
        """
        held_objects = []
        agent_name = f"agent_{self.agent_id}"
        for node in self.current_state:
            for relation in node.get('relation', []):
                if f"holds {agent_name}" in relation:
                    held_objects.append(node['name'])
        return held_objects

    def calculate_needed_objects(self, unsatisfied, inhand_objs):
        """
        Calculates the number of needed objects based on unsatisfied tuples.

        Args:
            unsatisfied (dict): Dictionary of unsatisfied predicates in format {(obj, relation, target): count}.
            inhand_objs (list): List of objects currently held by the agent.

        Returns:
            defaultdict: Mapping of object names to their needed counts.
        """
        needed_obj_count = defaultdict(int)
        for predicate, count in unsatisfied.items():
            obj, relation_type, target = predicate
            if relation_type in ['on', 'inside']:
                needed_obj_count[obj] += count
                if obj in inhand_objs:
                    needed_obj_count[obj] -= 1
        return needed_obj_count

    def calculate_remained_to_put(self, unsatisfied):
        """
        Determines the number of objects that need to be placed inside containers.

        Args:
            unsatisfied (dict): Dictionary of unsatisfied predicates in format {(obj, relation, target): count}.

        Returns:
            defaultdict: Mapping of container IDs to the number of objects remaining to put.
        """
        remained_to_put = defaultdict(int)
        for predicate, count in unsatisfied.items():
            obj, relation_type, target = predicate
            if relation_type == 'inside':
                # Find target node ID from name
                target_node = next((node for node in self.current_state if node['name'] == target), None)
                if target_node:
                    container_id = target_node['id']
                    remained_to_put[container_id] += count
        return remained_to_put

    def check_need_to_close(self, remained_to_put):
        """
        Determines if a container needs to be closed.

        Args:
            state (dict): Current state of the environment.
            remained_to_put (dict): Objects that still need to be put somewhere.

        Returns:
            bool: True if a container needs to be closed.
        """
        if not self.last_opened:
            for node in self.current_state:
                # Check if node is a container that can be opened/closed
                if (node['name'].lower() in ['fridge', 'kitchencabinets', 'cabinet', 'microwave', 'dishwasher', 'stove'] 
                    and any('OPEN' in state for state in node.get('state', []))):
                    self.last_opened = [node['name'], str(node['id'])]
                    break

        if self.last_opened and self.last_opened[0].lower() != 'toilet':
            container_node = next((n for n in self.current_state if str(n['id']) == self.last_opened[1]), None)
            if container_node and any('OPEN' in state for state in container_node.get('state', [])):
                # Check if we still need to put anything in this container
                container_id = container_node['id']
                if remained_to_put.get(container_id, 0) == 0:
                    return True
                
                # Check if there are any objects that need to be put somewhere
                return all(count == 0 for count in remained_to_put.values())
            else:
                self.last_opened = None
        return False

    def generate_plan(self, subgoal, need_to_close):
        """
        Generates an action plan for a given subgoal.

        Args:
            subgoal (list): Subgoal information.
            need_to_close (bool): Whether a container needs to be closed.

        Returns:
            list: List of action strings constituting the plan.
        """
        heuristic = self.heuristic_dict.get(subgoal[0].split('_')[0])
        if not heuristic:
            return []
        
        # Modified: Pass only agent_id and goal to heuristic functions
        actions, costs = heuristic(
            self.agent_id,  
            subgoal[0]  # Just pass the subgoal string
        )
        
        if actions:
            plan = [self.get_action_str(action) for action in actions]
            if need_to_close and (
                plan[0].startswith('walk') or 
                (plan[0].startswith(f'agent_{self.agent_id} open') and 
                len(plan[0].split(' ')) > 3 and 
                plan[0].split(' ')[3] != f"object_{self.last_opened[1]}")
            ):
                close_action = self.construct_close_action()
                if close_action:
                    plan.insert(0, close_action)
        else:
            plan = []
        
        if plan and plan[0].startswith(f'agent_{self.agent_id} open') and len(plan[0].split(' ')) > 3:
            elements = plan[0].split(' ')
            obj_id = elements[3].split('_')[1]
            obj_name = next((node['name'] for node in self.current_state if str(node['id']) == obj_id), None)
            if obj_name:
                self.last_opened = [obj_name, obj_id]
        print(f'plan: {plan}')
        return plan

    def generate_hold_plan(self, last_subgoal, need_to_close):
        """
        Generates an action plan for holding-related subgoals.

        Args:
            last_subgoal (str): The last subgoal pursued.
            need_to_close (bool): Whether a container needs to be closed.

        Returns:
            list: List of action strings constituting the hold plan.
        """
        heuristic = self.heuristic_dict.get(last_subgoal.split('_')[0])
        if not heuristic:
            return []
        
        actions, costs = heuristic(
            self.agent_id,  
            {},  # 'unsatisfied' is handled separately
            self.current_state,  
            last_subgoal
        )
        if actions:
            plan = [self.get_action_str(action) for action in actions]
            if need_to_close and (
                plan[0].split(' ')[1] == 'walk' or 
                (plan[0].split(' ')[1] == 'open' and len(plan[0].split(' ')) > 2 and plan[0].split(' ')[2] != self.last_opened[1])
            ):
                close_action = self.construct_close_action()
                if close_action:
                    plan.insert(0, close_action)
        else:
            plan = []
        
        if plan and plan[0].startswith(f'agent_{self.agent_id} open') and len(plan[0].split(' ')) > 3:
            elements = plan[0].split(' ')
            # Extract object name and ID from "object_<id>"
            obj_id = elements[3].split('_')[1]  # Get ID from "object_<id>"
            obj_name = next((node['name'] for node in self.current_state if str(node['id']) == obj_id), None)
            if obj_name:
                self.last_opened = [obj_name, obj_id]
        return plan

    def construct_close_action(self):
        """
        Constructs a close action string based on the last opened container.

        Returns:
            str or None: Close action string or None if no container is open.
        """
        if self.last_opened:
            return f"agent_{self.agent_id} close object_{self.last_opened[1]}"
        return None

    def perform_simulation(self, curr_root, t):
        """
        Performs a single simulation (selection and expansion) in the MCTS tree.

        Args:
            curr_root (Node): Current root node of the MCTS tree.
            t (int): Current timestep.

        Returns:
            list: Path of nodes traversed during the simulation.
        """
        node_path = [curr_root]
        current_node = curr_root

        # Selection: Traverse the tree until a non-expanded node is found
        while current_node.is_expanded:
            selected_child = self.select_child(current_node)
            if not selected_child:
                break
            node_path.append(selected_child)
            current_node = selected_child

        # Expansion: Expand the node if it's not already expanded
        if not current_node.is_expanded:
            current_node = self.expand(current_node, t)
            if current_node:
                node_path.append(current_node)

        return node_path

    def select_best_plan(self, curr_root, need_to_close):
        """
        Selects the best action plan after simulations based on visit counts.

        Args:
            curr_root (Node): Current root node of the MCTS tree.
            need_to_close (bool): Whether a container needs to be closed.

        Returns:
            tuple: Next root node, action plan, and list of subgoals.
        """
        next_root = None
        plan = []
        subgoals = []

        while curr_root.is_expanded:
            actions_taken, children_visit, next_root = self.select_next_root(curr_root)
            if not next_root:
                break
            curr_root = next_root
            plan += actions_taken
            subgoals.append(next_root.id[0])

        if plan:
            first_action = plan[0].split(' ')[1]  # Get action type from "agent_id action ..."
            if need_to_close and first_action in ['walk', 'open']:
                close_action = self.construct_close_action()
                if close_action:
                    plan.insert(0, close_action)

            if plan[0].split(' ')[1] == 'open' and len(plan[0].split(' ')) > 3:
                elements = plan[0].split(' ')
                obj_id = elements[3].split('_')[1]  # Get ID from "object_<id>"
                obj_name = next((node['name'] for node in self.current_state if str(node['id']) == obj_id), None)
                if obj_name:
                    self.last_opened = [obj_name, obj_id]
        
        return next_root, plan, subgoals

    def rollout(self, leaf_node, t):
        """
        Simulates a rollout from the leaf node to estimate the value of the state.

        Args:
            leaf_node (Node): Leaf node from which to start the rollout.
            t (int): Current timestep.

        Returns:
            float: Estimated value of the leaf node.
        """
        # Get current state information from leaf node
        _, goal_spec, satisfied, unsatisfied, num_steps, _ = leaf_node.id[1]
        
        sum_reward = 0
        last_reward = 0

        # Deep copy to avoid mutating the actual state
        satisfied = copy.deepcopy(satisfied)
        unsatisfied = copy.deepcopy(unsatisfied)

        # Generate subgoals for rollout
        subgoals = self.get_subgoal_space(satisfied, unsatisfied)
        # Shuffle subgoals to introduce randomness
        random.shuffle(subgoals)

        curr_state = copy.deepcopy(self.current_state)

        for rollout_step in range(min(len(subgoals), self.max_rollout_step)):
            subgoal = subgoals[rollout_step][0]
            # Add debug print to see what subgoal type we're getting
            subgoal_type = subgoal.split('_')[0]
            print(f"Subgoal type: {subgoal_type}")  # Debug print
            
            # Get the appropriate heuristic function
            heuristic = self.heuristic_dict.get(subgoal_type)
            if not heuristic:
                print(f"Warning: No heuristic found for subgoal type: {subgoal_type}")  # Debug print
                continue

            actions, costs = heuristic(
                self.agent_id,
                subgoal
            )

            if actions:
                num_steps += len(actions)
                total_cost = sum(costs)
                
                # Simulate the effects of actions on the state
                for action in actions:
                    # Update the simulated state based on action type
                    action_type = action[0]
                    if action_type == 'grab':
                        obj_id = action[1][1]
                        # Add holds relation
                        obj_node = next((node for node in curr_state if node['id'] == obj_id), None)
                        if obj_node:
                            if 'relation' not in obj_node:
                                obj_node['relation'] = []
                            obj_node['relation'].append(f"holds agent_{self.agent_id}")
                    
                    elif action_type in ['put', 'putIn']:
                        obj_id = action[1][1]
                        target_id = action[2][1]
                        relation_type = 'on' if action_type == 'put' else 'inside'
                        
                        # Remove holds relation and add new position relation
                        obj_node = next((node for node in curr_state if node['id'] == obj_id), None)
                        if obj_node:
                            if 'relation' in obj_node:
                                obj_node['relation'] = [r for r in obj_node['relation'] 
                                                      if not r.startswith('holds')]
                            obj_node['relation'].append(f"{relation_type} object_{target_id}")
                    
                    elif action_type in ['open', 'close']:
                        obj_id = action[1][1]
                        obj_node = next((node for node in curr_state if node['id'] == obj_id), None)
                        if obj_node:
                            new_state = 'OPEN' if action_type == 'open' else 'CLOSED'
                            if 'state' not in obj_node:
                                obj_node['state'] = []
                            obj_node['state'] = [s for s in obj_node['state'] 
                                               if s not in ['OPEN', 'CLOSED']]
                            obj_node['state'].append(new_state)

                # Calculate reward based on goal progress
                curr_reward = self.check_progress(goal_spec)
                delta_reward = curr_reward - last_reward - total_cost
                sum_reward += delta_reward
                last_reward = curr_reward

        # Track metrics during rollout
        actions_taken = 0
        total_cost = 0
        
        for rollout_step in range(min(len(subgoals), self.max_rollout_step)):
            # ... existing rollout logic ...
            actions_heuristic, costs = heuristic(
                self.agent_id,
                subgoal
            )
            actions_taken += len(actions_heuristic)
            total_cost += sum(costs)

        # Log rollout metrics
        wandb.log({
            "rollout_actions_taken": actions_taken,
            "rollout_total_cost": total_cost,
            "rollout_final_reward": sum_reward
        })

        return sum_reward

    def calculate_score(self, curr_node, child):
        """
        Calculates the UCB1 score for a child node to balance exploration and exploitation.

        Args:
            curr_node (Node): Current parent node.
            child (Node): Child node for which to calculate the score.

        Returns:
            float: UCB1 score of the child node.
        """
        parent_visit_count = curr_node.num_visited
        self_visit_count = child.num_visited
        subgoal_prior = child.subgoal_prior

        if self_visit_count == 0:
            u_score = 1e6  # High exploration value
            q_score = 0
        else:
            exploration_rate = np.log((1 + parent_visit_count + self.c_base) / self.c_base) + self.c_init
            u_score = exploration_rate * subgoal_prior * np.sqrt(parent_visit_count) / (1 + self_visit_count)
            q_score = child.sum_value / self_visit_count

        score = q_score + u_score
        return score

    def select_child(self, curr_node):
        """
        Selects the child node with the highest UCB1 score.

        Args:
            curr_node (Node): Current node from which to select a child.

        Returns:
            Node or None: Selected child node or None if no children exist.
        """
        if not curr_node.children:
            return None

        scores = [self.calculate_score(curr_node, child) for child in curr_node.children]
        max_score = max(scores)
        candidates = [child for child, score in zip(curr_node.children, scores) if score == max_score]
        selected_child = random.choice(candidates)
        return selected_child

    def backup(self, value, node_list):
        """
        Backpropagates the value through the node path, updating node statistics.

        Args:
            value (float): Value to backpropagate.
            node_list (list): List of nodes traversed during the simulation.
        """
        for node in node_list:
            node.sum_value += value
            node.num_visited += 1

    def select_next_root(self, curr_root):
        """
        Selects the next root node based on the highest visit count among children.

        Args:
            curr_root (Node): Current root node of the MCTS tree.

        Returns:
            tuple: Actions taken, list of visit counts, and the selected child node.
        """
        children = curr_root.children
        if not children:
            return [], [], None

        visits = np.array([child.num_visited for child in children])
        max_visit = visits.max()
        candidates = [child for child, visit in zip(children, visits) if visit == max_visit]
        selected_child = random.choice(candidates)

        actions = selected_child.id[1][-1] if len(selected_child.id[1]) >= 7 else []
        return actions, visits.tolist(), selected_child

    def transition_subgoal(self, satisfied, unsatisfied, subgoal):
        """
        Updates the satisfied predicates based on the completed subgoal.

        Args:
            satisfied (dict): Dictionary of satisfied predicates in format {(obj, relation, target): True}.
            unsatisfied (dict): Dictionary of unsatisfied predicates in format {(obj, relation, target): count}.
            subgoal (str): The subgoal that has been completed.
        """
        elements = subgoal.split('_')
        if not elements or len(elements) < 3:
            return

        action, obj_id, surface_id = elements[0], elements[1], elements[2]
        obj_node = self.node_map.get(float(obj_id))
        surface_node = self.node_map.get(float(surface_id))
        
        if not obj_node or not surface_node:
            return

        if action == 'put':
            relation_key = (obj_node['name'], 'on', surface_node['name'])
            satisfied[relation_key] = True
            if relation_key in unsatisfied:
                unsatisfied[relation_key] -= 1
                if unsatisfied[relation_key] <= 0:
                    del unsatisfied[relation_key]
                    
        elif action == 'putIn':
            relation_key = (obj_node['name'], 'inside', surface_node['name'])
            satisfied[relation_key] = True
            if relation_key in unsatisfied:
                unsatisfied[relation_key] -= 1
                if unsatisfied[relation_key] <= 0:
                    del unsatisfied[relation_key]
        
                    
        # Add more actions as needed

    def expand(self, leaf_node, t):
        """
        Expands a leaf node by creating child nodes for each possible subgoal.

        Args:
            leaf_node (Node): Leaf node to expand.
            t (int): Current timestep.

        Returns:
            Node: The expanded leaf node or the original node if no expansion occurred.
        """
        curr_state = leaf_node.id[1][1]
        if t >= self.max_episode_length:
            return leaf_node

        expanded_leaf_node = self.initialize_children(leaf_node)
        if expanded_leaf_node:
            leaf_node.is_expanded = True
            leaf_node = expanded_leaf_node
        return leaf_node

    def initialize_children(self, node):
        """
        Creates child nodes for all possible subgoals from the current state.

        Args:
            node (Node): Node for which to initialize children.

        Returns:
            Node or None: The first expanded child node or None if no children were added.
        """
        leaf_node_values = node.id[1]
        state, goal_spec, satisfied, unsatisfied, steps, actions_parent = leaf_node_values

        # Generate subgoals based on current state and goals
        subgoals = self.get_subgoal_space(satisfied, unsatisfied)
        
        if not subgoals:
            return None

        goals_expanded = 0
        for goal_predicate in subgoals:
            goal, predicate, aug_predicate = goal_predicate
            heuristic = self.heuristic_dict.get(goal.split('_')[0])
            if not heuristic:
                continue

            actions_heuristic, costs = heuristic(
                self.agent_id,
                goal
            )
            if not actions_heuristic:
                continue

            # Create a simulated next state by applying actions
            next_state = copy.deepcopy(self.current_state)
            
            # Apply each action's effects to the simulated state
            for action in actions_heuristic:
                action_type = action[0]
                
                if action_type == 'grab':
                    obj_id = action[1][1]
                    obj_node = next((node for node in next_state if node['id'] == obj_id), None)
                    if obj_node:
                        if 'relation' not in obj_node:
                            obj_node['relation'] = []
                        obj_node['relation'].append(f"holds agent_{self.agent_id}")
                
                elif action_type in ['put', 'putIn']:
                    obj_id = action[1][1]
                    target_id = action[2][1]
                    relation_type = 'on' if action_type == 'put' else 'inside'
                    
                    obj_node = next((node for node in next_state if node['id'] == obj_id), None)
                    if obj_node:
                        if 'relation' in obj_node:
                            obj_node['relation'] = [r for r in obj_node['relation'] 
                                                  if not r.startswith('holds')]
                        obj_node['relation'].append(f"{relation_type} object_{target_id}")
                
                elif action_type in ['open', 'close']:
                    obj_id = action[1][1]
                    obj_node = next((node for node in next_state if node['id'] == obj_id), None)
                    if obj_node:
                        new_state = 'OPEN' if action_type == 'open' else 'CLOSED'
                        if 'state' not in obj_node:
                            obj_node['state'] = []
                        obj_node['state'] = [s for s in obj_node['state'] 
                                           if s not in ['OPEN', 'CLOSED']]
                        obj_node['state'].append(new_state)
                
                elif action_type in ['switchon', 'switchoff']:
                    obj_id = action[1][1]
                    obj_node = next((node for node in next_state if node['id'] == obj_id), None)
                    if obj_node:
                        new_state = 'ON' if action_type == 'switchon' else 'OFF'
                        if 'state' not in obj_node:
                            obj_node['state'] = []
                        obj_node['state'] = [s for s in obj_node['state'] 
                                           if s not in ['ON', 'OFF']]
                        obj_node['state'].append(new_state)

            goals_expanded += 1

            # Update satisfied and unsatisfied predicates
            next_satisfied = copy.deepcopy(satisfied)
            next_unsatisfied = copy.deepcopy(unsatisfied)

            if aug_predicate:
                if predicate not in next_satisfied:
                    next_satisfied[predicate] = []
                next_satisfied[predicate].append(aug_predicate)
                if predicate in next_unsatisfied:
                    next_unsatisfied[predicate] -= 1
                    if next_unsatisfied[predicate] <= 0:
                        del next_unsatisfied[predicate]

            # Convert actions to strings for the action plan
            action_strings = [self.get_action_str(action) for action in actions_heuristic]

            # Create child node
            child_node = Node(
                parent=node,
                id=(goal, [
                    next_state,
                    goal_spec,
                    next_satisfied,
                    next_unsatisfied,
                    steps + len(actions_heuristic),
                    action_strings
                ]),
                num_visited=0,
                sum_value=0,
                subgoal_prior=1.0 / len(subgoals),
                is_expanded=False
            )
        
        if goals_expanded == 0:
            return None

        return node

    def get_action_str(self, action_tuple):
        """
        Convert action tuple to string format for new environment.
        
        Args:
            action_tuple (tuple): (action_name, (obj_name, obj_id), additional_args)
            
        Returns:
            str: Formatted action string
        """
        action_name = action_tuple[0]
        obj_args = [x for x in list(action_tuple)[1:] if x is not None]
        
        # Format depends on action type
        if action_name in ['walk', 'run', 'sit']:
            objects_str = ' '.join([f"object_{x[1]}" for x in obj_args])
            return f"agent_{self.agent_id} {action_name} to {objects_str}"
        else:
            objects_str = ' '.join([f"object_{x[1]}" for x in obj_args])
            return f"agent_{self.agent_id} {action_name} {objects_str}"

# Example Node class extension if needed (using AnyNode from anytree)
# If using AnyNode with extra attributes, ensure to initialize them appropriately.
# For example:
# Node(id=(...), num_visited=0, sum_value=0, subgoal_prior=..., is_expanded=False)
    def find_heuristic(self, agent_id, goal):
        """Find heuristic adapted for string-based relations"""
        observations = self.current_state  # Use self.current_state directly

        id2node = {node['id']: node for node in observations}

        # Parse container relationships from relation strings
        containerdict = {}
        for node in observations:
            for relation in node.get('relation', []):
                parts = relation.split()
                if 'inside' in relation:
                    from_id = node['id']
                    to_nodes = [n for n in observations if n['name'] == parts[-1]]
                    if to_nodes:
                        to_id = to_nodes[0]['id']
                        containerdict[from_id] = to_id

        target = int(float(goal.split('_')[-1]))
        observation_ids = [x['id'] for x in observations]

        # Find character's room
        try:
            char_node = next(node for node in observations if node['id'] == agent_id)
            room_relations = [r for r in char_node.get('relation', []) if 'inside' in r]
            if room_relations:
                room_name = room_relations[0].split()[-1]
                room_char = next(n['id'] for n in observations if n['name'] == room_name)
            else:
                room_char = None
        except Exception as e:
            print(f'Error finding room for character: {e}')
            return [], []

        action_list = []
        cost_list = []

        # Follow container chain until target is visible
        while target not in observation_ids:
            try:
                container = containerdict[target]
            except KeyError:
                print(f'Could not find container for target {id2node[target]["name"]}')
                return [], []

            if 'Room' in id2node[container]['name']:  # Assuming rooms have "Room" in name
                action_list = [('walk', (id2node[target]['name'], target), None)] + action_list
                cost_list = [0.5] + cost_list
            elif 'CLOSED' in id2node[container].get('state', []) or ('OPEN' not in id2node[container].get('state', [])):
                action = ('open', (id2node[container]['name'], container), None)
                action_list = [action] + action_list
                cost_list = [0.05] + cost_list

            target = container

        # Check if agent is close to target
        target_node = id2node[target]
        char_node = id2node[agent_id]
        is_close = any(
            'close' in relation.lower() and target_node['name'] in relation
            for relation in char_node.get('relation', [])
        )

        if not is_close:
            action_list = [('walk', (target_node['name'], target), None)] + action_list
            cost_list = [1] + cost_list

        return action_list, cost_list
    
    def grab_heuristic(self, agent_id, goal):
        """Grab heuristic adapted for string-based relations"""
        observations = self.current_state
        target_id = int(float(goal.split('_')[-1]))

        observed_ids = [x['id'] for x in observations]

        # Check if agent is close to target
        char_node = next(node for node in observations if node['id'] == agent_id)
        target_node = next(node for node in observations if node['id'] == target_id)

        is_close = any(
            'close' in relation.lower() and target_node['name'] in relation
            for relation in char_node.get('relation', [])
        )

        # Check if object is already grabbed
        is_grabbed = any(
            'holds' in relation.lower() and target_node['name'] in relation
            for relation in char_node.get('relation', [])
        )

        action_list = []
        cost_list = []

        if not is_grabbed:
            target_action = [('grab', (target_node['name'], target_id), None)]
            cost = [0.05]
        else:
            target_action = []
            cost = []

        if is_close and target_id in observed_ids:
            action_list += target_action
            cost_list += cost
        else:
            find_actions, find_costs = self.find_heuristic(agent_id, goal)
            action_list += find_actions + target_action
            cost_list += find_costs + cost

        return action_list, cost_list
    def turnOn_heuristic(self, agent_id, goal):
        """Heuristic function for turning on an object."""
        observations = self.current_state
        target_id = int(float(goal.split('_')[-1]))

        observed_ids = [node['id'] for node in observations]
        agent_node = next(node for node in observations if node['id'] == agent_id)
        target_node = next(node for node in observations if node['id'] == target_id)

        # Check if agent is close to target
        is_close = any(
            'in arms reach' in relation.lower() and target_node['name'] in relation
            for relation in agent_node.get('relation', [])
        )

        # Check if target is already switched on
        is_on = 'ON' in target_node.get('state', [])

        action_list = []
        cost_list = []

        if not is_on:
            target_action = [('switchon', (target_node['name'], target_id), None)]
            cost = [0.05]
        else:
            target_action = []
            cost = []

        if is_close and target_id in observed_ids:
            action_list += target_action
            cost_list += cost
        else:
            find_actions, find_costs = self.find_heuristic(agent_id, f'find_{target_id}')
            action_list += find_actions + target_action
            cost_list += find_costs + cost

        return action_list, cost_list
    def turnOff_heuristic(self, agent_id, goal):
        """Heuristic function for turning off an object."""
        observations = self.current_state
        target_id = int(float(goal.split('_')[-1]))

        observed_ids = [node['id'] for node in observations]
        agent_node = next(node for node in observations if node['id'] == agent_id)
        target_node = next(node for node in observations if node['id'] == target_id)

        is_close = any(
            'in arms reach' in relation.lower() and target_node['name'] in relation
            for relation in agent_node.get('relation', [])
        )

        is_off = 'OFF' in target_node.get('state', [])

        action_list = []
        cost_list = []  
        if not is_off:
            target_action = [('switchoff', (target_node['name'], target_id), None)]
            cost = [0.05]
        else:
            target_action = []
            cost = []
        if is_close and target_id in observed_ids:
            action_list += target_action
            cost_list += cost
        else:
            find_actions, find_costs = self.find_heuristic(agent_id, f'find_{target_id}')
            action_list += find_actions + target_action
            cost_list += find_costs + cost

        return action_list, cost_list
    
    def sit_heuristic(self, agent_id, goal):
        """Heuristic function for sitting on an object."""
        observations = self.current_state
        target_id = int(float(goal.split('_')[-1]))

        observed_ids = [node['id'] for node in observations]
        agent_node = next(node for node in observations if node['id'] == agent_id)
        target_node = next(node for node in observations if node['id'] == target_id)

        # Check if agent is close to target
        is_close = any(
            'close' in relation.lower() and target_node['name'] in relation
            for relation in agent_node.get('relation', [])
        )

        # Check if agent is already sitting on the target
        is_sitting = any(
            'sit' in relation.lower() and target_node['name'] in relation
            for relation in agent_node.get('relation', [])
        )

        action_list = []
        cost_list = []

        if not is_sitting:
            target_action = [('sit', (target_node['name'], target_id), None)]
            cost = [0.05]
        else:
            target_action = []
            cost = []

        if is_close and target_id in observed_ids:
            action_list += target_action
            cost_list += cost
        else:
            find_actions, find_costs = self.find_heuristic(agent_id, f'find_{target_id}')
            action_list += find_actions + target_action
            cost_list += find_costs + cost

        return action_list, cost_list

    def put_heuristic(self, agent_id, goal):
        """Heuristic function for putting an object on another object."""
        observations = self.current_state

        # Modified line: Convert to float first, then to int
        target_grab_id, target_put_id = [int(float(x)) for x in goal.split('_')[-2:]]

        # Check if object is already placed on target
        is_placed = False
        for node in observations:
            if node['id'] == target_put_id:
                target_put_name = node['name']
        for node in observations:
            if node['id'] == target_grab_id:
                for relation in node.get('relation', []):
                    if f"on {target_put_name}" in relation:
                        is_placed = True
                        break

        if is_placed:
            return [], []

        # Check if object is already held by someone else (not the agent)
        is_held_by_other = False
        for node in observations:
            for relation in node.get('relation', []):
                if f"holds" in relation and node['id'] == target_grab_id and f"agent_{agent_id}" not in relation:
                    is_held_by_other = True
                    break

        if is_held_by_other:
            return None, None

        target_node = next(node for node in observations if node['id'] == target_grab_id)
        target_node2 = next(node for node in observations if node['id'] == target_put_id)

        # Check if agent is holding the object
        target_grabbed = any(
            f"holds agent_{agent_id}" in relation
            for relation in target_node.get('relation', [])
        )

        if not target_grabbed:
            grab_obj1, cost_grab_obj1 = self.grab_heuristic(agent_id, f'grab_{target_grab_id}')
        else:
            grab_obj1 = []
            cost_grab_obj1 = []

        # Need to find where to put the object
        find_obj2, cost_find_obj2 = self.find_heuristic(agent_id, f'find_{target_put_id}')
        action = [('put', (target_node['name'], target_grab_id), (target_node2['name'], target_put_id))]
        cost = [0.05]
        res = grab_obj1 + find_obj2 + action
        cost_list = cost_grab_obj1 + cost_find_obj2 + cost

        return res, cost_list

    def open_heuristic(self, agent_id, goal):
        """Heuristic function for opening an object."""
        observations = self.current_state
        target_id = int(float(goal.split('_')[-1]))
        target_node = next(node for node in observations if node['id'] == target_id)

        # Check if target is already open
        is_open = 'OPEN' in target_node.get('state', [])
        if is_open:
            return [], []

        # Generate actions
        find_actions, find_costs = self.find_heuristic(agent_id, f'find_{target_id}')
        action_open = [('open', (target_node['name'], target_id), None)]
        cost_open = [0.05]
        res = find_actions + action_open
        cost_list = find_costs + cost_open

        return res, cost_list

    def putIn_heuristic(self, agent_id, goal):
        """Heuristic function for putting an object inside another object."""
        observations = self.current_state

        target_grab_id, target_put_id = [int(float(x)) for x in goal.split('_')[-2:]]

        # Check if object is already placed inside target
        is_placed = False
        for node in observations:
            if node['id'] == target_grab_id:
                for relation in node.get('relation', []):
                    if f"inside object_{target_put_id}" in relation:
                        is_placed = True
                        break

        if is_placed:
            return [], []

        # Check if object is held by someone else
        is_held_by_other = any(
            f"holds" in relation and node['id'] == target_grab_id and f"agent_{agent_id}" not in relation
            for node in observations
            for relation in node.get('relation', [])
        )

        if is_held_by_other:
            return None, None

        target_node = next(node for node in observations if node['id'] == target_grab_id)
        target_node2 = next(node for node in observations if node['id'] == target_put_id)

        # Check if agent is holding the object
        target_grabbed = any(
            f"holds agent_{agent_id}" in relation
            for relation in target_node.get('relation', [])
        )

        if not target_grabbed:
            grab_obj1, cost_grab_obj1 = self.grab_heuristic(agent_id, f'grab_{target_grab_id}')
        else:
            grab_obj1 = []
            cost_grab_obj1 = []

        # Need to find the container
        find_obj2, cost_find_obj2 = self.find_heuristic(agent_id, f'find_{target_put_id}')

        target_put_state = target_node2.get('state', [])
        action_open = [('open', (target_node2['name'], target_put_id), None)]
        action_put = [('putIn', (target_node['name'], target_grab_id), (target_node2['name'], target_put_id))]
        cost_open = [0.05]
        cost_put = [0.05]

        if 'CLOSED' in target_node2.get('state', []) or 'OPEN' not in target_node2.get('state', []):
            res = grab_obj1 + find_obj2 + action_open + action_put
            cost_list = cost_grab_obj1 + cost_find_obj2 + cost_open + cost_put
        else:
            res = grab_obj1 + find_obj2 + action_put
            cost_list = cost_grab_obj1 + cost_find_obj2 + cost_put

        return res, cost_list

    def close_heuristic(self, agent_id, goal):
        """Heuristic function for closing an object."""
        observations = self.current_state
        target_id = int(float(goal.split('_')[-1]))
        target_node = next(node for node in observations if node['id'] == target_id)

        # Check if target is already closed
        is_closed = 'CLOSED' in target_node.get('state', [])
        if is_closed:
            return [], []

        # Generate actions
        find_actions, find_costs = self.find_heuristic(agent_id, f'find_{target_id}')
        action_close = [('close', (target_node['name'], target_id), None)]
        cost_close = [0.05]
        res = find_actions + action_close
        cost_list = find_costs + cost_close

        return res, cost_list
    def check_goal_reached(self, goal, count):
        data = {"type": "full"}
        observation(data)
        # Check if the goal is reached based on the current state
        time.sleep(1)
        # goal is in the form of ('object', 'relation', 'target')
        sucesss_count = 0
        for node in self.current_state:
            if goal[0] in node['name'].lower():
                if goal[1] == 'state' and goal[2] in node['state']:
                    sucesss_count += 1
                elif f"{goal[0]} {goal[1]} {goal[2]}" in node['relation']:
                    sucesss_count += 1
        return sucesss_count >= count
    def run_mcts_task(self, goal_specification):
        """
        Runs the MCTS planning and execution for given goals.
        
        Args:
            goal_specification (dict): Dictionary of goals in format {(subject, relation_type, target): count_needed}
            
        Returns:
            float: Success rate of achieved goals
        """
        # Initialize satisfied and unsatisfied predicates
        satisfied_predicates = {}
        unsatisfied_predicates = {k: v for k, v in goal_specification.items()}

        # Create root node
        root = Node(
            id=('initial', [
                self.current_state,
                goal_specification,
                satisfied_predicates,
                unsatisfied_predicates,
                0,
                []
            ]),
            num_visited=0,
            sum_value=0,
            subgoal_prior=1.0,
            is_expanded=False
        )

        # Initialize heuristic dictionary
        heuristic_dict = {
            'find': self.find_heuristic,
            'grab': self.grab_heuristic,
            'switch on': self.turnOn_heuristic,
            'sit': self.sit_heuristic,
            'put': self.put_heuristic,
            'open': self.open_heuristic,
            'putIn': self.putIn_heuristic,
            'close': self.close_heuristic
        }

        # Run MCTS
        next_root, plan, subgoals = self.run(
            curr_root=root,
            t=0,
            heuristic_dict=heuristic_dict,
            last_subgoal=None
        )

        if self.verbose:
            print("Planned actions:", plan)
            print("Subgoals:", subgoals)

        # Execute actions
        if plan:
            action_sequence = utils.sequence(plan)
            for i, action_dict in enumerate(action_sequence):
                if self.verbose:
                    print(action_dict)
                    print(f"Action {i+1}/{len(action_sequence)}: {action_dict.get('task', 'unknown')}")
                
                set_action(action_dict)
                time.sleep(4)  # Wait for action to complete
                self.update_state()

        # Check goal completion
        success = sum(1 for goal, count in goal_specification.items() 
                    if self.check_goal_reached(goal, count))
        success_rate = success / len(goal_specification)

        if self.verbose:
            print(f"{success} goals reached out of {len(goal_specification)}")
            print(f"Success rate: {success_rate}")

        # Log metrics
        wandb.log({
            "success_goals": success,
            "total_goals": len(goal_specification),
            "success_rate": success_rate,
            "total_actions": len(plan) if plan else 0,
            "achieved_subgoals": len(subgoals)
        })

        return success_rate



# if __name__ == "__main__":

#         ## 1. Random Agent

#     goal_spec = {
#         # Format: (subject, relation_type, target): count_needed
#         #('apple', 'on', 'table'): 1,
#         ('milk', 'on', 'table'): 1,
#     }
#     env_name = 'WatchAndHelp1'

#     agent = Random_agent(agent_id=0, char_index=0, max_episode_length=100, num_simulation=1000, max_rollout_steps=5, c_init=1.25, c_base=19652, recursive=False, num_samples=1, num_processes=1, comm=None, logging=False, logging_graphs=False, seed=None, env_name=env_name)
#     graph = agent.load_graph_json()
#     filtered_graph = clean_graph(graph, goal_spec, None)
#     #print("Filtered graph nodes:", [node['name'] for node in filtered_graph])
#     for goal_tuple in goal_spec.keys():
#         print(f"Goal: {goal_tuple}")
#         # Create a single-goal specification dictionary
#         single_goal_spec = {goal_tuple: goal_spec[goal_tuple]}
#         agent.run(single_goal_spec)
