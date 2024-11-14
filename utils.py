# Define the fixed IDs for actions and relations
fixed_action_ids = {
    "walk": 0,
    "run": 1,
    "walk forward": 2,
    "walk backward": 3,
    "walk right": 4,
    "walk left": 5,
    "grab": 6,
    "grab from table": 7,
    "put": 8,
    "put on table": 9
}

fixed_relation_ids = {"on": 1, "under": 2, "inside": 3, "next to": 4}

# Function to update object mapping
def update_mapping(word, mapping):
    if word not in mapping and word.startswith("object"):
        # mapping[word] = len(mapping) + 1
        mapping[word] = int(word.replace("object", '').replace('_', ''))
    return mapping

# Process phrase with fixed action and relation IDs
def process_phrase_with_relations(
    phrase, action_ids, relation_ids, object_mapping=None
):
    if object_mapping is None:
        object_mapping = {}

    words = phrase.split()
    processed = {"agent_index": [], "task": [0, 0, 0, 0]}  # Action, Object1, Object2, Relation
    ignore_object2 = False  # Flag to ignore object2 if "to" or "from" is in the phrase

    for word in words:
        if word.startswith("agent_"):
            processed["agent_index"].append(int(word.split("_")[1]))
        elif word in action_ids:
            processed["task"][0] = action_ids[word]  # Set the action
        elif word in relation_ids:
            processed["task"][3] = relation_ids[word]  # Set the relation (last position)
        elif word in ["to", "from"]:
            ignore_object2 = True  # Mark that we should ignore the second object
        else:
            # Update mapping only for words that start with 'object'
            object_mapping = update_mapping(word, object_mapping)
            if word.startswith("object"):
                if processed["task"][1] == 0:
                    processed["task"][1] = object_mapping[word]  # Set object1
                elif not ignore_object2 and processed["task"][2] == 0:
                    processed["task"][2] = object_mapping[word]  # Set object2

    return processed, object_mapping

# Define the sequence function
def sequence(phrases):
    object_mapping = {}  # Initialize an empty object mapping

    data_list = []
    for phrase in phrases:
        processed, object_mapping = process_phrase_with_relations(
            phrase, fixed_action_ids, fixed_relation_ids, object_mapping
        )

        # Extract agent_index and task for the emit message
        agent_index = processed["agent_index"][0]  # Assuming one agent per phrase
        task = processed["task"]

        # Emit the message
        data = {"agent_index": [agent_index], "task": task}
        data_list.append(data)
        # set_action(data)
    return data_list


if __name__ == "__main__":
    action_list = ["agent_0 run to object_140"]

    ret = sequence(action_list)
    print(f"\nparsed action:\n{ret}")


    # Sequence
    action_list = [
        "agent_1 pickup object_1",
        "agent_2 walk to object_2",
        "agent_3 place object_4 on object_5",
        "agent_4 jump over object_6",
        "agent_5 examine object3 under object_7",
        "agent_6 run from object_8 to object_9",
        "agent_7 pick object_10 and place on object_11",
    ]

    ret = sequence(action_list)
    print(f"\nparsed action:\n{ret}")

