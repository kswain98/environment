import torch
import numpy as np
import json
from client import *
import wandb
from belief import Belief
import utils
import time
class Oracle:
    def __init__(self, max_episode_length=100, env_name="WatchAndHelp1"):
        """
        Initialize the Oracle agent.
        
        Args:
            max_episode_length (int): Maximum number of steps per episode
        """
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
        """Reset the oracle's internal state."""
        self.num_steps = 0
        self.last_action = None
        self.last_subgoal = None
        with open("graph.json", "r") as f:
            graph = json.load(f)
        data = {"env_index": [0],"graph": graph}
        reset(data)
        


    def get_action(self, graph, task_goal):
        """
        Get the next action from the oracle based on current state and goal.
        
        Args:
            graph (dict): Current environment graph state
            task_goal (dict): Goal specification
            env: Environment instance for executing actions
            
        Returns:
            tuple: (action_str, info_dict)
        """
        # Get system agent action
        system_agent_action, system_agent_info = self.get_system_agent_action(
            graph,
            task_goal,
            self.last_action,
            self.last_subgoal
        )
        
        # Update tracking
        if system_agent_action is not None:
            self.last_action = system_agent_action
            if system_agent_info['subgoals']:
                self.last_subgoal = system_agent_info['subgoals'][0]
        
        # Prepare action string
        action_str = f"{system_agent_action}" if system_agent_action else None
        
        # Only try to process action if it exists
        if action_str:
            try:
                action = utils.sequence(action_str)
                # Execute action and get results
                set_action(action)
            except IndexError:
                print(f"Warning: Could not process action string: {action_str}")
                action = None
        else:
            action = None
        
        self.num_steps += 1
        
        # Get reward and check terminal conditions
        reward, done, info = self._compute_reward(graph, task_goal)
        reward = torch.Tensor([reward])
        
        # Check max episode length
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
        """
        Determine the next action for the system agent with perfect information.
        """
        # Get nodes from WatchAndHelp1 key
        
        # Get all unsatisfied goals
        unsatisfied = {}
        for (subject, relation_type, target), count in task_goal.items():
            satisfied = self.check_progress({(subject, relation_type, target): count})
            if satisfied < count:
                unsatisfied[(subject, relation_type, target)] = count - satisfied

        if not unsatisfied:
            return None, {'subgoals': []}

        # Get the first unsatisfied goal
        (subject, relation_type, target), count = next(iter(unsatisfied.items()))
        
        # Find relevant objects from the nodes list
        subject_nodes = [n for n in graph if subject.lower() in n['name'].lower()]
        target_nodes = [n for n in graph if target.lower() in n['name'].lower()]
        
        if not subject_nodes or not target_nodes:
            return None, {'subgoals': []}
        
        subject_node = subject_nodes[0]
        target_node = target_nodes[0]
        # Generate plan based on relation type
        plan = []
        subgoal = None
        
        if relation_type == 'on':
            # If not holding object, grab it
            if not any('holds' in r for r in subject_node.get('relation', [])):
                plan = [
                    f"agent_0 walk to object_{int(subject_node['id'])}",
                    f"agent_0 grab object_{int(subject_node['id'])}"
                ]
                subgoal = f"grab_{subject}"
            # If holding object, place it
            else:
                plan = [
                    f"agent_0 walk to object_{int(target_node['id'])}",
                    f"agent_0 put object_{int(subject_node['id'])} object_{int(target_node['id'])}"
                ]
                subgoal = f"put_{subject}_{target}"
                
        elif relation_type == 'inside':
            # First open container if closed
            if 'CLOSED' in target_node.get('state', []):
                plan = [
                    f"agent_0 walk to object_{int(target_node['id'])}",
                    f"agent_0 open object_{int(target_node['id'])}"
                ]
                subgoal = f"open_{target}"
            # Then grab object if not holding
            elif not any('holds' in r for r in subject_node.get('relation', [])):
                plan = [
                    f"agent_0 walk to object_{int(subject_node['id'])}",
                    f"agent_0 grab object_{int(subject_node['id'])}"
                ]
                subgoal = f"grab_{subject}"
            # Finally place object inside
            else:
                plan = [
                    f"agent_0 walk to object_{int(target_node['id'])}",
                    f"agent_0 putin object_{int(subject_node['id'])} object_{int(target_node['id'])}"
                ]
                subgoal = f"putIn_{subject}_{target}"

        elif relation_type == 'state':
            if target == 'ON':
                plan = [
                    f"agent_0 walk to object_{int(subject_node['id'])}",
                    f"agent_0 switchon object_{int(subject_node['id'])}"
                ]
                subgoal = f"switch on_{subject}"
            elif target == 'OFF':
                plan = [
                    f"agent_0 walk to object_{int(subject_node['id'])}",
                    f"agent_0 switchoff object_{int(subject_node['id'])}"
                ]
                subgoal = f"switch off_{subject}"

        # Return first action in plan
        action = plan[0] if plan else None
        return action, {
            'plan': plan,
            'subgoals': [subgoal] if subgoal else []
        }

    def _compute_reward(self, graph, task_goal):
        """
        Compute reward based on goal completion.
        
        Args:
            graph (dict): Current environment graph state
            task_goal (dict): Goal specification
            
        Returns:
            tuple: (reward, done, info)
        """
        # Check how many goals are satisfied
        total_goals = sum(count for _, count in task_goal.items())
        satisfied = sum(self.check_progress({goal: count}) for goal, count in task_goal.items())
        
        # Compute reward
        reward = 1.0 if satisfied >= total_goals else 0.0
        
        # Check if done
        done = satisfied >= total_goals
        
        return reward, done, {
            'satisfied': satisfied,
            'total': total_goals
        }

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

if __name__ == "__main__":
    # Initialize environment
    data = {"environment": "WatchAndHelp1"}
    make(data)
    data = {"type": "graph"}
    observation(data)

    # Initialize Oracle agent
    oracle = Oracle(
        max_episode_length=100,
        env_name="WatchAndHelp1"
    )

    # Initialize wandb run for the main execution
    wandb.init(
        project="oracle-planning",
        name="main-execution",
        config={
            "max_episode_length": oracle.max_episode_length,
            "environment": "WatchAndHelp1"
        }
    )

    # Set goal specification
    goal_specification = {
        # Format: (subject, relation_type, target): count_needed
        ('apple', 'on', 'table'): 1,
        #('microwave', 'state', 'ON'): 1,
        #('plate', 'inside', 'microwave'): 1
        #('microwave', 'state', 'CLOSED'): 1,
    }

    # Get initial state
    initial_graph = oracle.update_graph()
    if not initial_graph:
        print("Failed to get initial state")
        exit(1)

    # Get plan from Oracle
    action, info = oracle.get_action(initial_graph, goal_specification)
    
    print("Planned actions: ", info['plan'])
    print("Subgoals: ", info['subgoals'])
    
    # Track metrics for the execution
    episode_metrics = {
        "total_actions": len(info['plan']) if info['plan'] else 0,
        "achieved_subgoals": len(info['subgoals']) if info['subgoals'] else 0,
        "execution_success": info['plan'] is not None and len(info['plan']) > 0
    }
    wandb.log(episode_metrics)

    # Execute actions and track progress
    if info['plan']:
        action_sequence = utils.sequence(info['plan'])
        for i, action_dict in enumerate(action_sequence):
            print(f"Executing action {i+1}/{len(action_sequence)}: {action_dict}")
            
            # Set action and wait for state update
            set_action(action_dict)
            time.sleep(5)
            
            # Update state
            oracle.update_graph()
            
            # Log action execution
            wandb.log({
                "action_step": i,
                "action_type": action_dict.get('action', "unknown"),
                "action_sequence_progress": (i + 1) / len(action_sequence)
            })

            # Check if goal is reached
            reward, done, info = oracle._compute_reward(oracle.current_state, goal_specification)
            if done:
                print(f"Goal reached after {i+1} steps!")
                break

        # Log final metrics
        wandb.log({
            "final_reward": reward,
            "steps_taken": i + 1,
            "goal_achieved": done
        })

    # Finish wandb run
    wandb.finish()
