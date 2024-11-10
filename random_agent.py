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

from . import belief

import sys
sys.path.append('..')


def clean_graph(state, goal_spec, last_opened):
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
    id2node = {node['id']: node for node in state['nodes']}
    
    # Track important nodes
    important_nodes = set()
    
    # Process goal specifications to find relevant nodes
    for (subject, relation_type, target), count in goal_spec.items():
        # Find nodes matching subject
        subject_nodes = [
            node['id'] for node in state['nodes']
            if subject.lower() in node['name'].lower()
        ]
        important_nodes.update(subject_nodes)
        
        # Find nodes matching target
        target_nodes = [
            node['id'] for node in state['nodes']
            if target.lower() in node['name'].lower()
        ]
        important_nodes.update(target_nodes)
    
    # Add essential nodes (character, rooms)
    important_nodes.update(
        node['id'] for node in state['nodes']
        if 'character' in node['name'].lower() or 'room' in node['name'].lower()
    )
    
    # Build containment relationships
    containment = defaultdict(list)
    for node in state['nodes']:
        for relation in node.get('relation', []):
            if 'inside' in relation.lower():
                parts = relation.split()
                if len(parts) >= 3:
                    container_name = parts[-1]
                    container_nodes = [n for n in state['nodes'] if n['name'].lower() == container_name.lower()]
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
    storage_locations = {
        ('dishwasher', 'kitchentable'): ['kitchencabinet', 'kitchendrawer', 'kitchencounter'],
        ('sofa', 'chair'): ['coffeetable']
    }
    
    for (subject, relation_type, target), count in goal_spec.items():
        if relation_type == 'state' and target == 'OFF':
            subject_nodes = [
                node for node in state['nodes']
                if subject.lower() in node['name'].lower()
            ]
            for node in subject_nodes:
                for source_classes, target_classes in storage_locations.items():
                    if any(cls.lower() in node['name'].lower() for cls in source_classes):
                        # Add storage location nodes
                        important_nodes.update(
                            n['id'] for n in state['nodes']
                            if any(cls.lower() in n['name'].lower() for cls in target_classes)
                        )
    
    # Create filtered graph with only important nodes
    return {
        "nodes": [id2node[node_id] for node_id in important_nodes]
    }


class Random_agent:
    """Random agent for graph-based environment"""
    def __init__(self, agent_id, char_index,
                 max_episode_length, num_simulation, max_rollout_steps, c_init, c_base, recursive=False,
                 num_samples=1, num_processes=1, comm=None, logging=False, logging_graphs=False, seed=None):
        self.agent_type = 'Random'
        self.verbose = False
        self.recursive = recursive

        if seed is None:
            seed = random.randint(0,100)
        self.seed = seed
        self.logging = logging
        self.logging_graphs = logging_graphs

        self.agent_id = agent_id
        self.char_index = char_index
        self.belief = None
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
        # Create a set of unique relations for each node
        for node in graph['nodes']:
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
            (node for node in graph['nodes'] if 'character' in node['name'].lower()),
            None
        )
        if char_node:
            print('Character:')
            print(char_node.get('relation', []))
            print('---')

    def get_action(self, obs, goal_spec, opponent_subgoal=None):
        """
        Generates a random valid action based on current state.
        
        Args:
            obs (dict): Current observation
            goal_spec (dict): Goal specifications
            opponent_subgoal (str, optional): Opponent's current subgoal
            
        Returns:
            tuple: (action_dict, info) where action_dict is the selected action and info contains additional data
        """
        # Load object information
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(f'{dir_path}/dataset/object_info_small.json', 'r') as f:
            content = json.load(f)
        
        # Update belief and environment state
        self.sample_belief(obs)
        self.sim_env.reset(self.previous_belief_graph, {0: goal_spec, 1: goal_spec})

        # Select random action type
        action_name = random.choice(['walktowards', 'grab', 'put', 'open'])
        
        # Get valid objects based on action type
        if action_name == 'walktowards':
            objects = [
                (node['name'], node['id']) for node in obs['nodes']
                if any(obj_type in node['name'].lower() for obj_type in content.values())
            ]
        elif action_name == 'grab':
            objects = [
                (node['name'], node['id']) for node in obs['nodes']
                if node['name'].lower() in content['objects_grab']
            ]
        elif action_name == 'put':
            objects = [
                (node['name'], node['id']) for node in obs['nodes']
                if node['name'].lower() in content['objects_surface'] + content['objects_inside']
            ]
        else:  # open
            objects = [
                (node['name'], node['id']) for node in obs['nodes']
                if node['name'].lower() in content['objects_inside']
            ]

        # Generate action if valid objects exist
        action_dict = None
        if objects:
            selected_object = random.choice(objects)
            obj_name, obj_id = selected_object
            action = self.can_perform_action(action_name, obj_name, obj_id, self.agent_id, obs, teleport=False)
            if action:
                action_dict = self.convert_action_to_dict(action)

        # Prepare info dictionary
        info = {}
        if self.logging:
            info = {
                'plan': [action_dict] if action_dict else [],
                'belief': copy.deepcopy(self.belief.edge_belief),
                'belief_graph': copy.deepcopy(self.sim_env.vh_state.to_dict())
            }
            if self.logging_graphs:
                info['obs'] = obs['nodes']

        return action_dict, info

    def convert_action_to_dict(self, action_str):
        """
        Converts action string to dictionary format for set_action.
        
        Args:
            action_str (str): Action string in format "[action] <obj1> (id1) <obj2> (id2)"
            
        Returns:
            dict: Action dictionary for set_action
        """
        if not action_str:
            return None

        parts = action_str.split()
        action_type = parts[0][1:-1]  # Remove brackets
        
        action_dict = {
            "action": self.action_map.get(action_type, action_type),
            "agent_id": self.agent_id
        }

        # Extract object IDs from format "<name> (id)"
        if len(parts) > 1:
            obj_id = parts[1].split('(')[1].rstrip(')')
            action_dict["object_id"] = int(obj_id)
        
        if len(parts) > 2:
            target_id = parts[2].split('(')[1].rstrip(')')
            action_dict["target_id"] = int(target_id)

        return action_dict

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
        
        # Update graph and environment
        graph_belief = self.sample_belief(observed_graph)
        self.sim_env.reset(graph_belief, task_goal)
        self.sim_env.to_pomdp()
        
    def can_perform_action(self, action, o1, o1_id, agent_id, graph, graph_helper=None, teleport=True):
        """
        Checks if an action can be performed in the current state.
        
        Args:
            action (str): Action type to check
            o1 (str): Object name
            o1_id (int): Object ID
            agent_id (int): Agent ID
            graph (dict): Current graph state
            graph_helper (object, optional): Helper object with object type information
            teleport (bool): Whether teleportation is allowed
            
        Returns:
            str or None: Formatted action string if action is valid, None otherwise
        """
        if action == 'no_action':
            return None

        id2node = {node['id']: node for node in graph['nodes']}
        num_args = 0 if o1 is None else 1
        grabbed_objects = [
            edge['to_id'] for edge in graph['edges'] 
            if edge['from_id'] == agent_id and edge['relation_type'] in ['HOLDS_RH', 'HOLD_LH']
        ]
        
        if num_args != self.args_per_action(action):
            return None

        close_edge = len([
            edge['to_id'] for edge in graph['edges'] 
            if edge['from_id'] == agent_id and edge['to_id'] == o1_id and edge['relation_type'] == 'CLOSE'
        ]) > 0

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
        if (action in ['grab', 'open', 'close']) and not close_edge:
            return None

        # Check open conditions
        if action == 'open':
            if graph_helper is not None:
                if id2node[o1_id]['class_name'] not in graph_helper.object_dict_types['objects_inside']:
                    return None
            if 'OPEN' in id2node[o1_id]['states'] or 'CLOSED' not in id2node[o1_id]['states']:
                return None

        # Check close conditions
        if action == 'close':
            if graph_helper is not None:
                if id2node[o1_id]['class_name'] not in graph_helper.object_dict_types['objects_inside']:
                    return None
            if 'CLOSED' in id2node[o1_id]['states'] or 'OPEN' not in id2node[o1_id]['states']:
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
                if 'CONTAINERS' in id2node[o1_id]['properties']:
                    action = 'putin'
                elif 'SURFACES' in id2node[o1_id]['properties']:
                    action = 'putback'

        # Handle walk teleportation
        if action.startswith('walk') and teleport:
            action = 'walk'

        # Format action string to match MCTS format
        if action.startswith('walk'):
            return f"agent_{agent_id} {action} to object_{o1_id}"
        elif action in ['put', 'putin', 'putback']:
            o2_id = grabbed_objects[0]
            return f"agent_{agent_id} {action} object_{o2_id} object_{o1_id}"
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