# Define the fixed IDs for actions and relations
fixed_action_ids = {
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
    "switch off":11,
    "sit": 12,
    "sleep": 13,
    "eat": 14,
    "drink": 15,
    "clean": 16,
    "point": 17,
    "cook": 18,
    "cut": 19,
    "pour": 20,
    "shower": 21,
    "dry": 22,
    "lock": 23,
    "unlock": 24,
    "fill": 25,
    "talk": 26,
    "laugh": 27,
    "angry": 28,
    "cry": 29,
    "call": 30,
    "interact": 31,
    "step forward": 32,
    "step backwards": 33,
    "turn left": 34,
    "turn right": 35,
}

fixed_relation_ids = {"on": 1, "under": 2, "inside": 3, "next to": 4}


# Function to update object mapping
def update_mapping(word, mapping):
    if word not in mapping and word.startswith("object"):
        mapping[word] = len(mapping) + 1
    return mapping


# Process phrase with fixed action and relation IDs
def process_phrase_with_relations(
    phrase, action_ids, relation_ids, object_mapping=None
):
    if object_mapping is None:
        object_mapping = {}

    words = phrase.split()
    processed = {"agent_index": [], "task": [0, 0, 0, 0]}

    for word in words:
        if word.startswith("agent_"):
            processed["agent_index"].append(int(word.split("_")[1]))
        elif word in action_ids:
            processed["task"][0] = action_ids[word]
        elif word in relation_ids:
            processed["task"][2] = relation_ids[word]
        else:
            # Update mapping only for words that start with 'object'
            object_mapping = update_mapping(word, object_mapping)
            if word.startswith("object"):
                if processed["task"][1] == 0:
                    processed["task"][1] = object_mapping[word]
                elif processed["task"][3] == 0:
                    processed["task"][3] = object_mapping[word]

    return processed, object_mapping


# Define the sequence function
def sequence(phrases):
    object_mapping = {}  # Initialize an empty object mapping

    for phrase in phrases:
        processed, object_mapping = process_phrase_with_relations(
            phrase, fixed_action_ids, fixed_relation_ids, object_mapping
        )

        # Extract agent_index and task for the emit message
        agent_index = processed["agent_index"][0]  # Assuming one agent per phrase
        task = processed["task"]

        # Emit the message
        data = {"agent_index": agent_index, "task": task}
        sio.emit("set_action", data)
