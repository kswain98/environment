import os
import json
import random
import time
from typing import List, Dict
import openai

class PromptGenerator:
    # Asset information as a constant
    ASSET_INFO = {
        "objects_inside": [
            "toilet", "bathroom_cabinet", "kitchencabinets",
            "bathroom_counter", "kitchencounterdrawer", "cabinet", "fridge", "oven", "dishwasher", "microwave"
        ],
        "objects_surface": [
            "bathroomcabinet", "bathroomcounter", "bed", "bench", "boardgame",
            "bookshelf", "cabinet", "chair", "coffeetable", "cuttingboard",
            "desk", "fryingpan", "kitchencabinets", "kitchencounter",
            "kitchentable", "mousemat", "nightstand", "oventray", "plate",
            "radio", "sofa", "stove", "towelrack"
        ],
        "objects_grab": [
            "pudding", "juice", "pancake", "apple", "book", "coffeepot", 
            "cupcake", "cutleryfork", "dishbowl", "milk", "milkshake", "plate", 
            "poundcake", "remotecontrol", "waterglass", "wine", "wineglass"
        ]
    }
    ALLOWED_ACTIONS = ['walk to', 'grab', 'put', 'putin', 'open', 'close']
    ALLOWED_RELATIONS = ['on', 'inside']
    ALLOWED_STATES = ['open', 'closed', 'switchon', 'switchoff']


    def __init__(self, api_key: str, graph_path: str = 'graph.json'):
        self.api_key = api_key
        # Initialize the OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        # Load graph data once and cache it
        self.graph_data = self.load_graph(graph_path)
    
    def load_graph(self, graph_path: str) -> Dict:
        """Load and parse the scene graph from file."""
        with open(graph_path, 'r') as f:
            data = json.load(f)
        return data
    
    def get_objects_by_type(self, scene_data: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize objects in the scene by their type."""
        object_types = {}
        excluded_objects = {
            "Wall", "SpotLight", "SM_DownLight", "RugsRectanglex", "RectLight",
            "CameraActor", "RecastNavMeshDefault", "NavMeshBoundsVolume",
            "PostProcessVolume", "PlayerStart", "Floor_x",
            "WorldSettings", "Brush", "DefaultPhysicsVolume",
            "GameplayDebuggerPlayerManager", "ChaosDebugDrawActor",
            "BP_ThirdPersonGameMode", "GameSession", "ParticleEventManager",
            "GameNetworkManager", "GameStateBase", "AbstractNavDataDefault",
            "BP_ThirdPersonPlayerController", "PlayerState", "PlayerCameraManager",
            "HUD", "GameplayDebuggerCategoryReplicator", "API",
            "GroupActor"
        }
        
        for obj in scene_data:
            obj_name = obj.get('name', '')
            if not any(excluded in obj_name for excluded in excluded_objects):
                obj_type = obj_name.split('_')[0] if '_' in obj_name else obj_name
                object_types.setdefault(obj_type, []).append(obj)
        return object_types

    def build_scene_description(self, scene_data: List[Dict]) -> str:
        """Build a detailed scene description from the scene data."""
        object_types = self.get_objects_by_type(scene_data)
        description_lines = ["Scene objects:"]
        for obj_type, objects in object_types.items():
            description_lines.append(f"{obj_type}: {len(objects)} instance(s)")
            for obj in objects:
                line = f"- {obj.get('name', 'Unnamed')} (id: {obj.get('id', 'N/A')})"
                if state := obj.get('state'):
                    line += f" state: {state}"
                description_lines.append(line)
        return "\n".join(description_lines)
    
    def generate_prompt(self, environment: str = None) -> str:
        """Generate a creative household task based on the scene objects."""
        # Randomly select an environment if none specified
        if environment is None:
            environment = random.choice(list(self.graph_data.keys()))
        scene_data = self.graph_data[environment]
        scene_description = self.build_scene_description(scene_data)
        
        # Prepare the system message with asset info
        system_message = f"""You are a creative task generator for household scenarios.
Generate a unique, specific task that makes sense for a household.
The task should:
1. Be specific about what needs to be done.
2. Mention relevant objects from the scene.
3. Include a clear goal.
4. Be written in natural language.
5. Be creative and different from typical tasks.
6. Consider different times of day, occasions, or scenarios.
7. Vary the complexity and nature of the task.
8. Only include at most 3 objects in the task.

available actions by the agents are: {', '.join(self.ALLOWED_ACTIONS)}
available relations between objects are: {', '.join(self.ALLOWED_RELATIONS)}
available states of objects are: {', '.join(self.ALLOWED_STATES)}

Generate only one task at a time. These are the available assets:
Objects Inside: {', '.join(self.ASSET_INFO['objects_inside'])}
Objects Surface: {', '.join(self.ASSET_INFO['objects_surface'])}
Objects Grab: {', '.join(self.ASSET_INFO['objects_grab'])}

Scene details:
{scene_description}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": "Generate a unique and creative task based on the above scene."}
                ],
                temperature=0.9
            )
            task = response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")
        
        # Append the generated task to file
        self.save_task(environment, task)
        return task

    def save_task(self, environment: str, task: str):
        """Append the task along with environment info to a log file."""
        with open('tasks.txt', 'a') as f:
            f.write("\n--- New Task ---\n")
            f.write(f"Environment: {environment}\n")
            f.write(f"Task: {task}\n")
            f.write("----------------\n")

if __name__ == "__main__":
    api_key = "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"
    generator = PromptGenerator(api_key)
    
    total_tasks = 1000
    delay = 3  # Initial delay in seconds between requests
    for i in range(total_tasks):
        backoff = delay  # Reset backoff for each task
        while True:
            try:
                prompt = generator.generate_prompt()  # Randomly selects an environment
                print(f"Generated task {i+1}/{total_tasks}:\n{prompt}\n")
                time.sleep(delay)
                break  # Exit the retry loop on success
            except Exception as e:
                print(f"Error generating task {i+1}: {e}")
                print(f"Retrying in {backoff} seconds...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)  # Exponential backoff up to 60 seconds
