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
from contextlib import contextmanager

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
        self.relation_list = ["on", "inside", "right of", "left of", "behind", "infront"]
        self.fixed_action_ids = {
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
            state (dict): Current state of the environment.
            goal_spec (list): List of goal specifications.

        Returns:
            int: Number of satisfied goal conditions.
        """
        satisfied_count = 0
        for node in self.current_state:
            for relation in node.get('relation', []):
                # Split relation into parts
                parts = relation.split()
                if len(parts) >= 3:
                    relation_type = ' '.join(parts[1:-1])
                    if relation_type in goal_spec:
                        satisfied_count += 1
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
                observation_request = {"type": "graph"}
                observation(observation_request)
                time.sleep(retry_delay)  # Wait for observation to be processed
                
                # Check if file exists
                if not os.path.exists("graph.json"):
                    print(f"Attempt {attempt + 1}: Waiting for graph.json to be created")
                    continue
                
                # Use file lock when reading
                with self.file_lock:
                    with open("graph.json", "r") as f:
                        content = f.read().strip()
                    
                    # Parse JSON
                    try:
                        new_state = json.loads(content)
                        if self.env_name in new_state:  # Verify environment exists in state
                            self.last_observation = new_state
                            self.current_state = new_state[self.env_name]
                            self.node_map = self.create_node_map()
                            self.relation_map = self.create_relation_map()
                            return True
                        else:
                            print(f"Environment {self.env_name} not found in state")
                            continue
                    
                    except json.JSONDecodeError as je:
                        print(f"Attempt {attempt + 1}: JSON parsing error: {str(je)}")
                        continue
                
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error in update_state: {str(e)}")
                if attempt == max_retries - 1:
                    return False
                
                time.sleep(retry_delay)
        
        return False

    def run(self, curr_root, t, heuristic_dict, last_subgoal):
        """
        Executes the main MCTS algorithm, performing simulations and selecting the best action sequence.

        Args:
            curr_root (Node): Current root node of the MCTS tree.
            t (int): Current timestep.
            heuristic_dict (dict): Dictionary of heuristic functions for subgoals.
            last_subgoal (str): The last subgoal that was pursued.

        Returns:
            tuple: Next root node, action plan, and list of subgoals.
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

        # Handle planning for specific subgoals
        plan = []
        for subgoal in subgoals:
            ''' if subgoal[0] == last_subgoal:'''
            plan = self.generate_plan(subgoal, need_to_close)
            if self.verbose:
                print(f'Repeat subgoal plan: {plan}')
            if plan:
                return None, plan, [last_subgoal]

        # Check if agent is holding any objects
        agent_name = f"agent_{self.agent_id}"
        print(f"agent_name: {agent_name}")
        holding_object = False
        for node in self.current_state:
            for relation in node.get('relation', []):
                if f"holds {agent_name}" in relation:
                    holding_object = True
                    plan = self.generate_hold_plan(last_subgoal, need_to_close)
                    if self.verbose:
                        print(plan[0] if plan else "No actions generated.")
                    if plan:
                        return None, plan, [last_subgoal]

        # Expand the current root node if it's not already expanded
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

        curr_state = copy.deepcopy(self.current_state)  # Work with a copy of the current state

        for rollout_step in range(min(len(subgoals), self.max_rollout_step)):
            subgoal = subgoals[rollout_step][0]
            heuristic = self.heuristic_dict.get(subgoal.split('_')[0])
            if not heuristic:
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
            'close' in relation.lower() and target_node['name'] in relation
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
            if node['id'] == target_grab_id:
                for relation in node.get('relation', []):
                    if f"on object_{target_put_id}" in relation:
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

if __name__ == "__main__":

    data = {"environment": "WatchAndHelp1"}
    make(data)
    data = {"type": "graph"}
    observation(data)
    '''TODO: change the env to reading the graph.json and updating it using observation function'''
    mcts = MCTS(
        agent_id=0,
        max_episode_length=100,
        num_simulation=1000,
        max_rollout_step=5,
        c_init=1.25,
        c_base=19652,
        env_name="WatchAndHelp1",
    )

    # Initialize state by getting first observation
    if not mcts.update_state():
        print("Failed to get initial state")
        exit(1)

    # arbitrarily set goal specification, satisfied predicates, and unsatisfied predicates
    goal_specification = {
    # Format: (subject, relation_type, target): count_needed
    ('apple', 'on', 'table'): 1
    }
    satisfied_predicates = {}
    unsatisfied_predicates = {
       ('apple', 'on', 'table'): 1
    }

    # Create root node using current state instead of loading from file
    root = Node(
        id=('initial', [
            mcts.current_state,           # State as dictionary (same as above since we don't have separate formats)
            goal_specification,           # Goal predicates
            satisfied_predicates,         # Currently satisfied predicates
            unsatisfied_predicates,       # Currently unsatisfied predicates
            0,                           # Number of steps taken
            []                           # Actions taken so far
        ]),
        num_visited=0,
        sum_value=0,
        subgoal_prior=1.0,
        is_expanded=False
    )

    # Initialize wandb run for the main execution
    wandb.init(
        project="mcts-planning",
        name="main-execution",
        config={
            "goal_specification": goal_specification,
            "max_episode_length": mcts.max_episode_length,
            "environment": "WatchAndHelp1"
        }
    )

    # Run MCTS with state updates
    next_root, plan, subgoals = mcts.run(
        curr_root=root,
        t=0,
        heuristic_dict={
            'find': mcts.find_heuristic,  # Use class method
            'grab': mcts.grab_heuristic,  # Use class method
            'turnOn': mcts.turnOn_heuristic,
            'sit': mcts.sit_heuristic,
            'put': mcts.put_heuristic,
            'open': mcts.open_heuristic,
            'putIn': mcts.putIn_heuristic
        },
        last_subgoal=None
    )

    print("Planned actions: ", plan)
    print("Subgoals: ", subgoals)
    
    # Track metrics for the actual execution
    episode_metrics = {
        "total_actions": len(plan),
        "achieved_subgoals": len(subgoals),
        "execution_success": len(plan) > 0
    }
    wandb.log(episode_metrics)

    # Execute actions and track progress
    action_sequence = utils.sequence(plan)
    for i, action_dict in enumerate(action_sequence):
        print(f"Executing action {i+1}/{len(action_sequence)}: {action_dict}")
        
        # Set action and wait for state update
        set_action(action_dict)
        time.sleep(5)  # Increased delay between actions
        
        # Update state with retry
        if not mcts.update_state():
            print(f"Warning: Failed to update state for action {i+1}")
        
        # Log action execution
        wandb.log({
            "action_step": i,
            "action_type": action_dict.get('action', "unknown"),
            "action_sequence_progress": (i + 1) / len(action_sequence)
        })

    # Add final delay before finishing
    time.sleep(0.5)
    
    # Finish wandb run
    wandb.finish()
