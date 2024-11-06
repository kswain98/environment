import json
import random
import copy
import numpy as np
from collections import defaultdict
from anytree import AnyNode as Node
from tqdm import tqdm
from client import *
import utils

class MCTS:
    def __init__(self, env, agent_id, char_index, max_episode_length, num_simulation, max_rollout_step, c_init, c_base, seed=1 , env_name = 'Kitchen'):
        """
        Initializes the MCTS algorithm with the given parameters.

        Args:
            env: The environment in which the agent operates.
            agent_id (int): Identifier for the agent.
            char_index (int): Character index or related identifier.
            max_episode_length (int): Maximum length of an episode.
            num_simulation (int): Number of simulations to run per MCTS iteration.
            max_rollout_step (int): Maximum steps during rollout.
            c_init (float): Initial exploration constant.
            c_base (float): Base exploration constant.
            seed (int): Random seed for reproducibility.
            env_name (str): Name of the environment.
        """
        self.env = env
        self.discount = 1.0  # Discount factor for future rewards; consider tuning this parameter
        self.agent_id = agent_id
        self.char_index = char_index
        self.max_episode_length = max_episode_length
        self.num_simulation = num_simulation
        self.max_rollout_step = max_rollout_step
        self.c_init = c_init 
        self.c_base = c_base
        self.seed = seed
        self.heuristic_dict = None
        self.last_opened = None
        self.verbose = False
        self.env_name = env_name

        # Set random seeds for reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # Load and map the graph data
        self.graph = self.load_graph_json()
        self.node_map = self.create_node_map()
        self.relation_map = self.create_relation_map()
        self.relation_list = ["supported by", "inside", "right of", "left of", "behind", "infront"]
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

    def load_graph_json(self):
        """
        Loads the graph data from the JSON file.

        Returns:
            list: List of node dictionaries from the graph.
        """
        with open('MCTS/graph.json', 'r') as f:
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
                    relation_type = ' '.join(parts[1:-1])  # e.g., "supported by", "inside"
                    target = parts[-1]
                    relation_map[relation_type].append((subject, target))
        return relation_map

    def check_progress(self, state, goal_spec):
        """
        Checks progress towards goals based on current state.

        Args:
            state (dict): Current state of the environment.
            goal_spec (list): List of goal specifications.

        Returns:
            int: Number of satisfied goal conditions.
        """
        satisfied_count = 0
        for node in state['nodes']:
            for relation in node.get('relation', []):
                # Split relation into parts
                parts = relation.split()
                if len(parts) >= 3:
                    relation_type = ' '.join(parts[1:-1])
                    if relation_type in goal_spec:
                        satisfied_count += 1
        return satisfied_count

    def get_subgoal_space(self, state, satisfied, unsatisfied, verbose=0):
        """
        Generates possible subgoals based on current state.

        Args:
            state (dict): Current state of the environment.
            satisfied (dict): Already satisfied relations.
            unsatisfied (dict): Unsatisfied relations.
            verbose (int): Verbosity level.

        Returns:
            list: List of potential subgoals.
        """
        subgoal_space = []
        # instead of looking at state['nodes'], look at the graph
        for node in self.graph:
            for relation in node["relation"]:
                parts = relation.split()
                if len(parts) >= 3:
                    relation_type = ' '.join(parts[1:-1])
                    if relation_type not in satisfied and relation_type not in unsatisfied:
                        subgoal = self.relation_to_subgoal(relation)
                        if subgoal:
                            subgoal_space.append([subgoal, relation_type, None])
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
            'supported by': 'put',
            'inside': 'putIn',
            'holds': 'grab'
        }

        action = action_map.get(relation_type)
        if action:
            return f"{action}_{subject}_{target}"
        return None

    def update_state(self):
        """
        Updates the current state using the observation function.
        """
        observation_request = {"type": "graph"}
        observation(observation_request)
        
        # Wait for and process the observation response
        # Note: You'll need to implement a way to receive the response
        # This could be through a callback or waiting for a response
        try:
            with open("graph.json", "r") as f:
                new_state = json.load(f)
                self.last_observation = new_state
                self.current_state = new_state[self.env_name]
                # Update node and relation maps with new state
                self.node_map = self.create_node_map()
                self.relation_map = self.create_relation_map()
        except Exception as e:
            print(f"Error updating state: {e}")
            return False
        return True

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
        curr_vh_state_tmp, curr_state_tmp, _, satisfied, unsatisfied, _, actions_parent = curr_root.id[1]
        
        # Generate subgoals based on the current state
        subgoals = self.get_subgoal_space(curr_state_tmp, satisfied, unsatisfied, verbose=1)
        
        if self.verbose:
            print(f'satisfied: {satisfied}')
            print(f'unsatisfied: {unsatisfied}')
            print(f'subgoals: {subgoals}')
            print(f'last_subgoal: {last_subgoal}')

        # Retrieve objects currently held by the agent
        inhand_objs = self.get_inhand_objects(curr_state_tmp)
        
        # Calculate the number of needed objects based on unsatisfied predicates
        needed_obj_count = self.calculate_needed_objects(unsatisfied, inhand_objs)

        # Determine objects that need to be placed inside containers
        remained_to_put = self.calculate_remained_to_put(unsatisfied)

        # Determine if a container needs to be closed
        need_to_close = self.check_need_to_close(curr_state_tmp, remained_to_put)

        # Handle planning for specific subgoals
        for subgoal in subgoals:
            if subgoal[0] == last_subgoal:
                plan = self.generate_plan(subgoal, need_to_close, curr_state_tmp)
                if self.verbose:
                    print(f'Repeat subgoal plan: {plan}')
                if plan:
                    return None, plan, [last_subgoal]

        # If the subgoal is related to holding objects, generate a plan accordingly
        for edge in curr_state_tmp['edges']:
            if edge['relation_type'].startswith('HOLDS') and self.agent_id in [edge['from_id'], edge['to_id']]:
                plan = self.generate_hold_plan(last_subgoal, need_to_close, curr_state_tmp)
                if self.verbose:
                    print(plan[0] if plan else "No actions generated.")
                if plan:
                    return None, plan, [last_subgoal]

        # Expand the current root node if it's not already expanded
        if not curr_root.is_expanded:
            curr_root = self.expand(curr_root, t)

        # Perform MCTS simulations with progress tracking
        for explore_step in tqdm(range(self.num_simulation), disable=not self.verbose):
            node_path = self.perform_simulation(curr_root, t)
            if node_path:
                value = self.rollout(node_path[-1], t)
                discount_factor = self.discount ** node_path[-2].id[1][-2] if len(node_path) > 1 else 1.0
                self.backup(value * discount_factor, node_path)

        # Select the next root node based on visited children
        next_root, plan, subgoals = self.select_best_plan(curr_root, need_to_close, curr_state_tmp)

        if self.verbose and plan:
            print(plan[0])

        return next_root, plan, subgoals

    def get_inhand_objects(self, state):
        """
        Retrieves the list of objects currently held by the agent.

        Args:
            state (dict): Current state of the environment.

        Returns:
            list: Names of objects being held by the agent.
        """
        held_objects = []
        agent_name = f"agent_{self.agent_id}"
        for node in state['nodes']:
            for relation in node.get('relation', []):
                if f"holds {agent_name}" in relation:
                    held_objects.append(node['name'])
        return held_objects

    def calculate_needed_objects(self, unsatisfied, inhand_objs):
        """
        Calculates the number of needed objects based on unsatisfied predicates.

        Args:
            unsatisfied (dict): Dictionary of unsatisfied relations with counts.
            inhand_objs (list): List of objects currently held by the agent.

        Returns:
            defaultdict: Mapping of object names to their needed counts.
        """
        needed_obj_count = defaultdict(int)
        for predicate, count in unsatisfied.items():
            elements = predicate.split('_')
            if elements[0] in ['on', 'inside'] and len(elements) >= 2:
                obj = elements[1]
                needed_obj_count[obj] += count
                if obj in inhand_objs:
                    needed_obj_count[obj] -= 1
        return needed_obj_count

    def calculate_remained_to_put(self, unsatisfied):
        """
        Determines the number of objects that need to be placed inside containers.

        Args:
            unsatisfied (dict): Dictionary of unsatisfied relations with counts.

        Returns:
            defaultdict: Mapping of container IDs to the number of objects remaining to put.
        """
        remained_to_put = defaultdict(int)
        for predicate, count in unsatisfied.items():
            elements = predicate.split('_')
            if elements[0] == 'inside' and len(elements) > 2:
                try:
                    container_id = int(elements[2])
                    remained_to_put[container_id] += count
                except ValueError:
                    if self.verbose:
                        print(f"Invalid container ID in predicate: {predicate}")
        return remained_to_put

    def check_need_to_close(self, state, remained_to_put):
        """
        Determines if a container needs to be closed.

        Args:
            state (dict): Current state of the environment.
            remained_to_put (dict): Objects that still need to be put somewhere.

        Returns:
            bool: True if a container needs to be closed.
        """
        if not self.last_opened:
            for node in state['nodes']:
                # Check if node is a container that can be opened/closed
                if (node['name'].lower() in ['fridge', 'kitchencabinets', 'cabinet', 'microwave', 'dishwasher', 'stove'] 
                    and any('OPEN' in state for state in node.get('state', []))):
                    self.last_opened = [node['name'], str(node['id'])]
                    break

        if self.last_opened and self.last_opened[0].lower() != 'toilet':
            container_node = next((n for n in state['nodes'] if str(n['id']) == self.last_opened[1]), None)
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

    def generate_plan(self, subgoal, need_to_close, state):
        """
        Generates an action plan for a given subgoal.

        Args:
            subgoal (list): Subgoal information.
            need_to_close (bool): Whether a container needs to be closed.
            state (dict): Current state of the environment.

        Returns:
            list: List of action strings constituting the plan.
        """
        heuristic = self.heuristic_dict.get(subgoal[0].split('_')[0])
        if not heuristic:
            return []
        
        actions, costs = heuristic(
            self.agent_id, 
            self.char_index, 
            {},  # 'unsatisfied' is handled separately
            state, 
            self.env, 
            subgoal[0]
        )
        if actions:
            plan = [self.get_action_str(action) for action in actions]
            if need_to_close and (plan[0].startswith('[walk]') or 
                                  (plan[0].startswith('[open]') and len(plan[0].split(' ')) > 2 and plan[0].split(' ')[2] != self.last_opened[1])):
                close_action = self.construct_close_action()
                if close_action:
                    plan.insert(0, close_action)
        else:
            plan = []
        
        if plan and plan[0].startswith('[open]') and len(plan[0].split(' ')) > 2:
            elements = plan[0].split(' ')
            self.last_opened = [elements[1], elements[2]]
        return plan

    def generate_hold_plan(self, last_subgoal, need_to_close, state):
        """
        Generates an action plan for holding-related subgoals.

        Args:
            last_subgoal (str): The last subgoal pursued.
            need_to_close (bool): Whether a container needs to be closed.
            state (dict): Current state of the environment.

        Returns:
            list: List of action strings constituting the hold plan.
        """
        heuristic = self.heuristic_dict.get(last_subgoal.split('_')[0])
        if not heuristic:
            return []
        
        actions, costs = heuristic(
            self.agent_id, 
            self.char_index, 
            {},  # 'unsatisfied' is handled separately
            state, 
            self.env, 
            last_subgoal
        )
        if actions:
            plan = [self.get_action_str(action) for action in actions]
            if need_to_close and (plan[0].split(' ')[1] == 'walk' or 
                                  (plan[0].split(' ')[1] == 'open' and len(plan[0].split(' ')) > 2 and plan[0].split(' ')[2] != self.last_opened[1])):
                close_action = self.construct_close_action()
                if close_action:
                    plan.insert(0, close_action)
        else:
            plan = []
        
        if plan and plan[0].startswith('[open]') and len(plan[0].split(' ')) > 2:
            elements = plan[0].split(' ')
            self.last_opened = [elements[1], elements[2]]
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

    def select_best_plan(self, curr_root, need_to_close, state):
        """
        Selects the best action plan after simulations based on visit counts.

        Args:
            curr_root (Node): Current root node of the MCTS tree.
            need_to_close (bool): Whether a container needs to be closed.
            state (dict): Current state of the environment.

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
            first_action = plan[0].split(' ')[1]  # Changed from [0] to [1]
            if need_to_close and first_action in ['walk', 'open']:
                close_action = self.construct_close_action()
                if close_action:
                    plan.insert(0, close_action)

            if plan[0].split(' ')[1] == 'open' and len(plan[0].split(' ')) > 3:
                elements = plan[0].split(' ')
                self.last_opened = [elements[2], elements[3]]  # Changed from [1], [2] to [2], [3]
        
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
        # Update state before rollout
        if not self.update_state():
            return 0  # Return neutral value if state update fails
            
        curr_vh_state, curr_state, goal_spec, satisfied, unsatisfied, num_steps, actions_parent = leaf_node.id[1]
        sum_reward = 0
        last_reward = 0

        # Deep copy to avoid mutating the actual state
        satisfied = copy.deepcopy(satisfied)
        unsatisfied = copy.deepcopy(unsatisfied)

        # Generate subgoals for rollout
        subgoals = self.get_subgoal_space(curr_state, satisfied, unsatisfied, verbose=0)

        # Shuffle subgoals to introduce randomness
        random.shuffle(subgoals)

        for rollout_step in range(min(len(subgoals), self.max_rollout_step)):
            subgoal = subgoals[rollout_step][0]
            heuristic = self.heuristic_dict.get(subgoal.split('_')[0])
            if not heuristic:
                continue

            actions, costs = heuristic(
                self.agent_id, 
                self.char_index, 
                unsatisfied, 
                curr_state, 
                self.env, 
                subgoal
            )

            if actions:
                num_steps += len(actions)
                total_cost = sum(costs)
                for action in actions:
                    action_str = self.get_action_str(action)
                    try:
                        next_vh_state = self.env.transition(curr_vh_state, {0: action_str})
                        curr_vh_state = next_vh_state
                        curr_state = next_vh_state.to_dict()
                    except Exception as e:
                        if self.verbose:
                            print(f"Transition error during rollout: {e}")
                        break

                curr_reward = self.check_progress(curr_state, goal_spec)
                delta_reward = curr_reward - last_reward - total_cost
                sum_reward += delta_reward
                last_reward = curr_reward

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
            satisfied (dict): Dictionary of satisfied relations.
            unsatisfied (dict): Dictionary of unsatisfied relations.
            subgoal (str): The subgoal that has been completed.
        """
        """TODO: Subject to change on new important actions"""
        elements = subgoal.split('_')
        if not elements or len(elements) < 3:
            return

        action, obj_id, surface_id = elements[0], elements[1], elements[2]
        if action == 'put':
            obj_node = self.node_map.get(float(obj_id))
            surface_node = self.node_map.get(float(surface_id))
            if obj_node and surface_node:
                relation_key = (obj_node['name'], 'on', surface_node['name'])
                satisfied[relation_key] = True
                if relation_key in unsatisfied:
                    del unsatisfied[relation_key]
        elif action == 'putIn':
            obj_node = self.node_map.get(float(obj_id))
            container_node = self.node_map.get(float(surface_id))
            if obj_node and container_node:
                relation_key = (obj_node['name'], 'inside', container_node['name'])
                satisfied[relation_key] = True
                if relation_key in unsatisfied:
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
        vh_state, state, goal_spec, satisfied, unsatisfied, steps, actions_parent = leaf_node_values

        subgoals = self.get_subgoal_space(state, satisfied, unsatisfied, verbose=0)
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
                self.char_index, 
                unsatisfied, 
                state, 
                self.env, 
                goal
            )
            if not actions_heuristic:
                continue

            # Apply actions to transition to the next state
            next_vh_state = vh_state
            actions_str = []
            for action in actions_heuristic:
                action_str = self.get_action_str(action)
                actions_str.append(action_str)
                try:
                    next_vh_state = self.env.transition(next_vh_state, {0: action_str})
                except Exception as e:
                    if self.verbose:
                        print(f"Transition error during expansion: {e}")
                    continue

            goals_expanded += 1

            # Update satisfied and unsatisfied predicates
            next_satisfied = copy.deepcopy(satisfied)
            next_unsatisfied = copy.deepcopy(unsatisfied)

            if aug_predicate:
                next_satisfied[predicate].append(aug_predicate)
                next_unsatisfied[predicate] -= 1
                if next_unsatisfied[predicate] <= 0:
                    del next_unsatisfied[predicate]

            # Create a new child node
            child_node = Node(
                parent=node,
                id=(goal, [
                    next_vh_state, 
                    next_vh_state.to_dict(), 
                    goal_spec, 
                    next_satisfied, 
                    next_unsatisfied,
                    len(actions_heuristic), 
                    actions_str
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
        """Convert action tuple to string format for new environment"""
        action_name = action_tuple[0]
        obj_args = [x for x in list(action_tuple)[1:] if x is not None]
        
        # Format depends on action type
        if action_name in ['walk', 'run']:
            return f"{action_name} {' '.join(str(x) for x in obj_args)}"
        else:
            objects_str = ' '.join([f"object_{x[1]}" for x in obj_args])
            return f"agent_{self.agent_id} {action_name} {objects_str}"

# Example Node class extension if needed (using AnyNode from anytree)
# If using AnyNode with extra attributes, ensure to initialize them appropriately.
# For example:
# Node(id=(...), num_visited=0, sum_value=0, subgoal_prior=..., is_expanded=False)
def find_heuristics(agent_id, char_index, unsatisfied, state, env, goal):
    pass
def grab_heuristics(agent_id, char_index, unsatisfied, state, env, goal):
    pass
def put_heuristics(agent_id, char_index, unsatisfied, state, env, goal):
    pass
def cook_heuristics(agent_id, char_index, unsatisfied, state, env, goal):
    pass
def open_heuristics(agent_id, char_index, unsatisfied, state, env, goal):
    pass
def close_heuristics(agent_id, char_index, unsatisfied, state, env, goal):
    pass

if __name__ == "__main__":

    data = {"environment": "KswainEscapeRoom4"}
    make(data)
    data = {"type": "graph"}
    observation(data)
    '''TODO: change the env to reading the graph.json and updating it using observation function'''
    mcts = MCTS(
        env=env,
        agent_id=1,
        char_index=0,
        max_episode_length=100,
        num_simulation=1000,
        max_rollout_step=5,
        c_init=1.25,
        c_base=19652,
        verbose=1,
        env_name="KswainEscapeRoom4"
    )

    # Initialize state by getting first observation
    mcts.update_state()
    # arbitrarily set goal specification, satisfied predicates, and unsatisfied predicates
    goal_specification = {
    # Format: (subject, relation_type, target): count_needed
    ('cup', 'inside', 'cabinet'): 1,
    ('plate', 'supported by', 'counter'): 1,
    ('fridge', 'state', 'CLOSED'): 1,
        ('microwave', 'state', 'OFF'): 1
    }
    satisfied_predicates = {}
    unsatisfied_predicates = {
        ('cup', 'inside', 'cabinet'): 1,
        ('plate', 'supported by', 'counter'): 1,
        ('fridge', 'state', 'CLOSED'): 1,
        ('microwave', 'state', 'OFF'): 1
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

    # Run MCTS with state updates
    next_root, plan, subgoals = mcts.run(
        curr_root=root,
        t=0,
        heuristic_dict={
            'find': find_heuristic,
            'grab': grab_heuristic,
            'put': put_heuristic,
            # ... other heuristics
        },
        last_subgoal=None
    )

    print("Planned actions: ", plan)
    print("Subgoals: ", subgoals)
    
    # Execute actions and update state after each one
    for action in utils.sequence(plan):
        set_action(action)
        mcts.update_state()  # Update state after each action



