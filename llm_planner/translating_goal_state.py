import json
from collections import defaultdict

def translate_predicate_to_goal(predicate):
    """
    Translates a predicate string into a goal tuple format (subject, relation_type, target).
    
    Args:
        predicate (str): Predicate string like 'on_plate_kitchentable' or 'inside_apple_fridge'
        
    Returns:
        tuple: (subject, relation_type, target) or None if invalid format
    """
    try:
        # Split predicate into parts
        parts = predicate.split('_')
        
        if len(parts) < 3:
            return None
            
        relation_type = parts[0]
        subject = parts[1]
        target = parts[2]
        
        # Map relation types
        relation_mapping = {
            'on': 'on',
            'inside': 'inside',
            'holds': 'holds',
            'sit': 'sit',
            'state': 'state'
        }
        
        # Special handling for state predicates
        if relation_type in ['open', 'close', 'switchon', 'switchoff']:
            relation_type = 'state'
            # Map state values
            state_mapping = {
                'open': 'OPEN',
                'close': 'CLOSED',
                'switchon': 'ON', 
                'switchoff': 'OFF'
            }
            target = state_mapping[relation_type]
            
        # Get mapped relation type or use original
        relation_type = relation_mapping.get(relation_type, relation_type)
            
        return (subject, relation_type, target)
        
    except Exception as e:
        print(f"Error translating predicate {predicate}: {str(e)}")
        return None

def process_goals(file_path, save_path, data_type):
    goal_states = {}
    with open(file_path, 'r') as f:
        data = json.load(f)[data_type]
    print("opened")
    for node in data:
        goal = node["goal"]
        print(goal)
        name = node["name"]
        
        # Create goal specification and unsatisfied predicates
        goal_spec = {}
        unsatisfied = {}
        
        for predicate in goal:
            goal_tuple = translate_predicate_to_goal(predicate)
            
            if goal_tuple:
                if goal_tuple not in goal_spec:
                    goal_spec[str(goal_tuple)] = 1
                    unsatisfied[str(goal_tuple)] = 1
        
        if goal_spec:  # Only save if there are valid goals
            goal_states[name] = {
                "goal_specification": goal_spec,
                "satisfied_predicates": {},
                "unsatisfied_predicates": unsatisfied
            }
    
    # Save to JSON file
    with open(save_path, 'w') as f:
        json.dump(goal_states, f, indent=2)

    return goal_states

if __name__ == "__main__":
    goal_states1 = process_goals('gather_data_actiongraph_new_test.json', 'goal_states_new_test.json', 'new_test_data')
    goal_states2 = process_goals('gather_data_actiongraph_test.json', 'goal_states_test.json', 'test_data')
    goal_states3 = process_goals('gather_data_actiongraph_train.json', 'goal_states_train.json', 'train_data')
    print("Goal states have been processed and saved to 'goal_states.json'")
    # Print example of first goal state
    if goal_states1:
        first_key = next(iter(goal_states1))
        print(f"\nExample goal state for {first_key}:")
        print(json.dumps(goal_states1[first_key], indent=2))
    if goal_states2:
        first_key = next(iter(goal_states2))
        print(f"\nExample goal state for {first_key}:")
        print(json.dumps(goal_states2[first_key], indent=2))
    if goal_states3:
        first_key = next(iter(goal_states3))
        print(f"\nExample goal state for {first_key}:")
        print(json.dumps(goal_states3[first_key], indent=2))


