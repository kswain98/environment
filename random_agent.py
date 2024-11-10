import numpy as np
import random
import time
import math
import copy
import importlib
import json
import multiprocessing
import pickle
from pathlib import Path
import os
from collections import *
from client import *
import belief
import utils    
import sys
sys.path.append('..')


def clean_graph(obs_graph, goal_spec, last_opened):
    """
    Cleans and filters the graph to only include nodes relevant to the current goals.
    
    Args:
        state (dict): Current state containing nodes with embedded relations
        goal_spec (dict): Dictionary of goal specifications in format {(subject, relation_type, target): count}
        last_opened (tuple or None): Information about last opened container (name, id)
    
    Returns:
        dict: Filtered graph containing only relevant nodes
    """
    # Create id to node mapping for quick lookups
    id2node = {node['id']: node for node in obs_graph}
    
    # Track important nodes
    important_nodes = set()
    
    # Process goal specifications to find relevant nodes
    for (subject, relation_type, target), count in goal_spec.items():
        # Find nodes matching subject
        subject_nodes = [
            node['id'] for node in obs_graph
            if subject.lower() in node['name'].lower()
        ]
        important_nodes.update(subject_nodes)
        
        # Find nodes matching target
        target_nodes = [
            node['id'] for node in obs_graph 
            if target.lower() in node['name'].lower()
        ]
        important_nodes.update(target_nodes)
    
    # Add essential nodes (character, rooms)
    important_nodes.update(
        node['id'] for node in obs_graph
        if 'character' in node['name'].lower() or 'room' in node['name'].lower()
    )
    
    # Build containment relationships
    containment = defaultdict(list)
    for node in obs_graph:
        for relation in node.get('relation', []):
            if 'inside' in relation.lower():
                parts = relation.split()
                if len(parts) >= 3:
                    container_name = parts[-1]
                    container_nodes = [n for n in obs_graph if n['name'].lower() == container_name.lower()]
                    if container_nodes:
                        containment[node['id']].append(container_nodes[0]['id'])
    
    # Recursively add contained objects
    to_process = list(important_nodes)
    while to_process:
        current_id = to_process.pop()
        if current_id in containment:
            new_nodes = [
                node_id for node_id in containment[current_id] 
                if node_id not in important_nodes
            ]
            important_nodes.update(new_nodes)
            to_process.extend(new_nodes)
    
    # Add last opened container if specified
    if last_opened is not None:
        obj_id = int(last_opened[1][1:-1])
        important_nodes.add(obj_id)
    
    # Add potential storage locations for cleanup tasks
    augmented_class_names = []
    for goal in goal_spec.keys():
        #goal spec is in the form of ('object', 'relation', 'target')
        if goal[1] == 'state' and goal[2] == 'OFF':
            if goal[0] in ['dishwasher', 'kitchentable']:
                augmented_class_names += ['kitchencabinet', 'kitchendrawer', 'kitchencounter']
            if goal[0] in ['sofa', 'chair']:
                augmented_class_names += ['coffeetable']
    containers = [[node['id'], node['name']] for node in obs_graph if node['name'] in augmented_class_names]
    for obj_id in containers:
        if obj_id not in important_nodes:
            important_nodes.add(obj_id)

    # Create filtered graph with only important nodes
    return [id2node[node_id] for node_id in important_nodes]


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
        self.belief = belief.Belief(self.current_state, agent_id, prior=None, forget_rate=0.0, seed=None, rate=0.5, low_prob=0.001)
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

        # Add action mapping dictionary
        self.action_map = {
            'walktowards': 'walk to',
            'grab': 'grab',
            'put': 'put',
            'putin': 'putin',
            'open': 'open',
            'close': 'close'
        }

    def filtering_graph(self, graph):
        """
        Filters duplicate relations from the graph.
        
        Args:
            graph (dict): Graph containing nodes with relations
            
        Returns:
            dict: Graph with filtered relations
        """
        if not graph:
            return []
        
        # Create a set of unique relations for each node
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
        """
        Updates belief based on new observation.
        
        Args:
            obs_graph (dict): Observed graph state
            
        Returns:
            dict: Updated belief graph
        """
        new_graph = self.belief.update_graph_from_gt_graph(obs_graph)
        if new_graph is None:
            # Handle the error case - either return the original graph or an empty graph
            new_graph = obs_graph or {"nodes": []}
        
        self.previous_belief_graph = self.filtering_graph(new_graph)
        return new_graph

    def get_relations_char(self, graph):
        """
        Gets all relations involving the character.
        
        Args:
            graph (dict): Current graph state
            
        Returns:
            None: Prints character relations
        """
        char_node = next(
            (node for node in graph if 'character' in node['name'].lower()),
            None
        )
        if char_node:
            print('Character:')
            print(char_node.get('relation', []))
            print('---')

    def get_action(self, obs, goal_spec, opponent_subgoal=None):
        """
        Generates a random valid action based on current state.
        Returns action string instead of dictionary.
        """
        # Load object information
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(f'{dir_path}/dataset/object_info_small.json', 'r') as f:
            content = json.load(f)
        
        # Update belief
        self.sample_belief(obs)
        
        # Remove sim_env reset and replace with client observation
        observation({"type": "graph"})

        # Select random action type and try until we find a valid one
        possible_actions = ['walktowards', 'grab', 'put', 'open']
        action_name = random.choice(possible_actions)
        
        action_str = None
        # Get valid objects based on action type
        # check if action can be performed
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
                    if node['name'].lower() in content['objects_grab']
                    and 'character' not in node['name'].lower()
                ]
        elif action_name == 'put':
            objects = [
                    (node['name'], node['id']) for node in obs
                    if node['name'].lower() in content['objects_surface'] + content['objects_inside']
                    and 'character' not in node['name'].lower()
            ]
        elif action_name == 'open':
            objects = [
                (node['name'], node['id']) for node in obs
                    if node['name'].lower() in content['objects_inside']
                    and 'character' not in node['name'].lower()
                ]

        # Try each object until we find a valid action
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
        """
        Resets the agent's state for a new episode.
        
        Args:
            observed_graph (dict): Initially observed graph
            gt_graph (dict): Ground truth graph
            task_goal (dict): Goal specifications
            seed (int): Random seed
            simulator_type (str): Type of simulator
            is_alice (bool): Whether agent is Alice
            
        Returns:
            None
        """
        self.last_action = None
        self.last_subgoal = None
        self.previous_belief_graph = None
        self.last_opened = None
        
        # Initialize belief
        self.belief = belief.Belief(gt_graph, agent_id=self.agent_id, seed=seed)
        self.belief.sample_from_belief()
        
        # Update graph using client functions
        self.sample_belief(observed_graph)
        reset({"task_goal": task_goal})
        
    def can_perform_action(self, action, o1, o1_id, agent_id, graph, graph_helper=None, teleport=True):
        """
        Checks if an action can be performed in the current state.
        """
        if action == 'no_action':
            return None

        # Create id to node mapping
        id2node = {node['id']: node for node in graph}
        
        # Format object ID properly - ensure it's a clean integer
        try:
            o1_id = int(float(o1_id))
        except (ValueError, TypeError):
            print(f"Invalid object ID format: {o1_id}")
            return None
        
        # Verify object exists
        if o1_id not in id2node:
            print(f"Object ID {o1_id} not found in graph")
            return None

        # Get clean object name (remove BP_ prefix or SM_ prefix if exists) 
        o1_clean = o1.replace('BP_', '').replace('SM_', '').lower()
        
        # Check for grabbed objects using relations
        agent_node = next((node for node in graph if "character" in node['name'].lower()), None)
        if not agent_node:
            print(f"Agent ID {agent_id} not found in graph")
            return None
        
        grabbed_objects = []
        for node in graph:
            for relation in node.get('relation', []):
                if f"holds agent_{agent_id}" in relation:
                    grabbed_objects.append(node['id'])
        

        # Calculate number of arguments based on action type
        num_args = 1  # Most actions require 1 argument
        if 'put' in action:
            num_args = 2  # Put actions require 2 arguments

        if num_args != self.args_per_action(action):
            return None

        # Check proximity using relations
        close_to_object = False
        if agent_node:
            for relation in agent_node.get('relation', []):
                if f"close object_{o1_id}" in relation:
                    close_to_object = True
                    break

        # Check grab conditions
        if action == 'grab':
            if len(grabbed_objects) > 0:
                return None

        # Check walk conditions
        if action.startswith('walk'):
            if o1_id in grabbed_objects:
                return None

        # Check self-targeting
        if o1_id == agent_id:
            return None

        # Check proximity requirements
        if (action in ['grab', 'open', 'close']) and not close_to_object:
            return None

        # Check open conditions
        if action == 'open':
            if graph_helper is not None:
                if id2node[o1_id]['class_name'] not in graph_helper.object_dict_types['objects_inside']:
                    return None
            if 'OPEN' in id2node[o1_id].get('state', []) or 'CLOSED' not in id2node[o1_id].get('state', []):
                return None

        # Check close conditions
        if action == 'close':
            if graph_helper is not None:
                if id2node[o1_id]['class_name'] not in graph_helper.object_dict_types['objects_inside']:
                    return None
            if 'CLOSED' in id2node[o1_id].get('state', []) or 'OPEN' not in id2node[o1_id].get('state', []):
                return None

        # Handle put actions
        if 'put' in action:
            if len(grabbed_objects) == 0:
                return None
            else:
                o2_id = grabbed_objects[0]
                if o2_id == o1_id:
                    return None

        # Determine put action type
        if action.startswith('put'):
            if graph_helper is not None:
                if id2node[o1_id]['class_name'] in graph_helper.object_dict_types['objects_inside']:
                    action = 'putin'
                if id2node[o1_id]['class_name'] in graph_helper.object_dict_types['objects_surface']:
                    action = 'putback'
            else:
                # Check properties in node's state or properties
                properties = id2node[o1_id].get('properties', [])
                if isinstance(properties, str):
                    properties = [properties]
                
                if 'CONTAINERS' in properties:
                    action = 'putin'
                elif 'SURFACES' in properties:
                    action = 'putback'

        # Handle walk teleportation
        if action.startswith('walk') and teleport:
            action = 'walk'

        # Format action string to new format at the end
        if action.startswith('walk'):
            return f"agent_{agent_id} walk to object_{o1_id}"
        elif action in ['put', 'putin', 'putback']:
            o2_id = grabbed_objects[0]
            return f"agent_{agent_id} place object_{o2_id} on object_{o1_id}"
        elif action == 'grab':
            return f"agent_{agent_id} pickup object_{o1_id}"
        else:
            return f"agent_{agent_id} {action} object_{o1_id}"

    def args_per_action(self, action):
        """
        Returns the number of arguments required for a given action type.
        
        Args:
            action (str): Action type
            
        Returns:
            int: Number of required arguments
        """
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
        """
        Loads the graph data from the JSON file.

        Returns:
            list: List of node dictionaries from the graph.
        """
        observation({"type": "graph"})
        with open('graph.json', 'r') as f:
            data = json.load(f)
            # Return the nodes list for the specified environment
            return data[self.env_name]
    def run(self, goal_spec):
        # Convert single goal tuple to goal spec dictionary if needed
        if isinstance(goal_spec, tuple):
            goal_spec = {goal_spec: 1}
        
        filtered_graph = clean_graph(self.current_state, goal_spec, None)
        for i in range(self.max_episode_length):
            action_str, info = self.get_action(filtered_graph, goal_spec, None)
            if action_str is None:
                continue
            action = utils.sequence([action_str])[0]
            print(f"Action: {action}")
            set_action(action)
            print("Action taken")
            observation({"type": "graph"})
            if self.check_goal_reached(goal_spec):
                print(f"Goal reached in {i} steps")
                break
        print("Goal not reached")

    def check_goal_reached(self, goal_spec):
        for goal in goal_spec.keys():
            # goal is in the form of ('object', 'relation', 'target')
            for node in self.current_state:
                if goal[0] in node['name'].lower():
                    if goal[1] == 'state' and goal[2] in node['state']:
                        return True
                    else:
                        if f"{goal[0]} {goal[1]} {goal[2]}" in node['relation']:
                            return True
        return False

if __name__ == "__main__":
    # Initialize environment and get initial state
    data = {"environment": "WatchAndHelp1"}
    make(data)
    
    goal_spec = {
        # Format: (subject, relation_type, target): count_needed
        ('apple', 'on', 'table'): 1,
        ('microwave', 'state', 'on'): 1
    }
    env_name = 'WatchAndHelp1'

    agent = Random_agent(agent_id=0, char_index=0, max_episode_length=100, num_simulation=1000, max_rollout_steps=5, c_init=1.25, c_base=19652, recursive=False, num_samples=1, num_processes=1, comm=None, logging=False, logging_graphs=False, seed=None, env_name=env_name)
    graph = agent.load_graph_json()
    filtered_graph = clean_graph(graph, goal_spec, None)
    print("Filtered graph nodes:", [node['name'] for node in filtered_graph])
    for goal_tuple in goal_spec.keys():
        print(f"Goal: {goal_tuple}")
        # Create a single-goal specification dictionary
        single_goal_spec = {goal_tuple: goal_spec[goal_tuple]}
        agent.run(single_goal_spec)
