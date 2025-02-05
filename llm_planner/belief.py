import numpy as np
import json
import copy

import random
import copy

class Belief():
    def __init__(self, current_state, agent_id, prior=None, forget_rate=0.0, seed=None, rate=0.5, low_prob=0.001):
        """
        Initializes the Belief class with the current state and agent ID.

        Args:
            current_state (list): List of node dictionaries representing the current environment state.
            agent_id (int): Identifier for the agent.
            prior (dict, optional): Prior probabilities for states and relations. Defaults to None.
            forget_rate (float, optional): Rate at which belief decays over time. Defaults to 0.0.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            rate (float, optional): Rate parameter for belief updates. Defaults to 0.5.
            low_prob (float, optional): Minimum probability threshold. Defaults to 0.001.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Basic belief parameters
        self.agent_id = agent_id
        self.current_state = copy.deepcopy(current_state)
        self.prior = prior
        self.forget_rate = forget_rate
        self.rate = rate
        self.low_prob = low_prob
        self.high_prob = 1e9
        # Container and class restrictions
        self.container_restrictions = {
            'book': ['cabinet', 'kitchencabinet']
        }
        self.id_restrictions_inside = {}
        self.class_nodes_delete = ['wall', 'floor', 'ceiling', 'curtain', 'window']
        self.categories_delete = ['Doors']
        
        # Initialize belief structures
        self.initialize_basic_beliefs()  # Original belief initialization
        self.initialize_complex_beliefs()  # Additional belief structures from watch_and_help
        
        # Additional tracking
        self.grabbed_object = []
        self.last_opened = None
        self.debug = False

    def initialize_basic_beliefs(self):
        """Original belief initialization for binary states and simple relations"""
        # Original initialization code here
        self.id2node = {node['id']: node for node in self.current_state}
        self.node_beliefs = {}
        self.relation_beliefs = {}
        # ... rest of original initialize_belief code ...

    def initialize_complex_beliefs(self):
        """Initialize additional belief structures"""
        self.relation_belief = {}
        self.first_belief = {}
        
        # Initialize container IDs and index mappings
        self.container_ids = []
        self.container_index_belief_dict = {}
        for idx, node in enumerate(self.current_state):
            if node['name'].startswith(('BP_Table', 'BP_SideTable', 'SM_TableDining')):
                self.container_ids.append(node['id'])
                self.container_index_belief_dict[node['id']] = idx

    def update_belief(self, observations):
        """
        Updates the belief state based on new observations.

        Args:
            observations (list): List of observed nodes with updated states and relations.
        """
        # Update node beliefs
        for node in observations:
            node_id = node['id']
            node_state = node.get('state', [])
            node_relations = node.get('relation', [])

            # Update state beliefs
            belief_states = self.node_beliefs.get(node_id, {})
            for state in ['ON', 'OFF', 'OPEN', 'CLOSED']:
                if state in node_state:
                    belief_states[state] = 1.0
                    opposite_state = 'OFF' if state == 'ON' else 'ON' if state == 'OFF' else 'CLOSED' if state == 'OPEN' else 'OPEN'
                    belief_states[opposite_state] = 0.0
            self.node_beliefs[node_id] = belief_states

            # Update relation beliefs
            for relation_str in node_relations:
                parts = relation_str.split()
                if len(parts) >= 3:  # Format: "{object} {relation_type} {target}"
                    relation_type = parts[1]  # e.g. "inside" or "on"
                    target_name = ' '.join(parts[2:])
                    target_node = next((n for n in self.current_state if n['name'] == target_name), None)
                    if target_node:
                        target_id = target_node['id']
                        # Update belief to 1.0 for observed relations
                        self.relation_beliefs[(node_id, relation_type, target_id)] = 1.0

        # Decay beliefs for unobserved relations
        for key in self.relation_beliefs:
            if key not in [(node['id'], relation.split()[0], next((n['id'] for n in self.current_state if n['name'] == ' '.join(relation.split()[1:])), None)) for node in observations for relation in node.get('relation', [])]:
                self.relation_beliefs[key] *= (1 - self.forget_rate)

    def sample_from_belief(self, as_vh_state=False):
        """
        Samples a possible world from the current belief state.

        Returns:
            list: A list of node dictionaries representing the sampled state.
        """
        sampled_state = copy.deepcopy(self.current_state)

        # Sample node states
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

        # Clear existing relations
        for node in sampled_state:
            node['relation'] = []

        # Sample relations based on beliefs
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
        """
        Resets the belief to the initial prior.
        """
        self.initialize_belief()

    def update(self, origin, final):
        """Updates belief values using exponential decay."""
        dist_total = origin - final
        ratio = (1 - np.exp(-self.rate * np.abs(origin-final)))
        return origin - ratio * dist_total

    def reset_to_prior_if_invalid(self, belief_node):
        """Resets belief to prior if probability is too low."""
        # belief_node: [names, probs]
        if belief_node[1].max() == self.low_prob:
            belief_node[1] = self.prior

    def update_to_prior(self):
        """Updates beliefs toward prior values."""
        for node_name in self.relation_belief:
            self.relation_belief[node_name]['INSIDE'][1] = self.update(
                self.relation_belief[node_name]['INSIDE'][1], 
                self.first_belief[node_name]['INSIDE'][1]
            )

    def update_graph_from_gt_graph(self, gt_graph):
        """Updates graph based on ground truth observations."""
        # Initialize tracking lists
        ids_known_info = [[]]  # Track container indices only
        
        # Handle case where gt_graph is a dictionary with 'nodes' key
        if isinstance(gt_graph, dict) and 'nodes' in gt_graph:
            gt_graph = gt_graph['nodes']
        elif not isinstance(gt_graph, list):
            print(f"Unexpected gt_graph type: {type(gt_graph)}")
            return
        
        id2node = {int(x['id']): x for x in gt_graph}
        inside = {}
        grabbed_object = []
        id_updated = []

        # Process relations
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
        
        # Update beliefs based on visible objects and their relations
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

        # Check open containers
        for id_node in self.container_ids:
            if id_node in visible_ids and 'OPEN' in id2node[id_node]['state']:
                for id_node_child in self.relation_belief.keys():
                    if id_node_child not in inside.keys() or inside[id_node_child] != id_node:
                        ids_known_info[0].append(self.container_index_belief_dict[id_node])
                        self.relation_belief[id_node_child]['INSIDE'][1][self.container_index_belief_dict[id_node]] = self.low_prob

        # Update impossible beliefs
        mask_obj = np.ones(len(self.container_ids))
        if len(ids_known_info[0]):
            mask_obj[np.array(ids_known_info[0])] = 0
        mask_obj = (mask_obj == 1)

        for id_node in self.relation_belief.keys():
            if np.max(self.relation_belief[id_node]['INSIDE'][1]) == self.low_prob:
                self.relation_belief[id_node]['INSIDE'][1] = self.first_belief[id_node]['INSIDE'][1]



if __name__ == '__main__':
    graph_init = '../../example_graph/example_graph.json' 
    with open(graph_init, 'r') as f:
        graph = json.load(f)['init_graph']
    Belief(graph)
