import os
import json
from typing import Dict, List, Optional, Tuple
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interface.client import observation, render, set_action, make, reset  # Not used here
from llm import OpenAIBot
from llm_planner.utils import sequence
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import subprocess
import socket
import queue

# Define allowed actions
ALLOWED_ACTIONS = ["walk to", "grab", "put", "putin", "open", "close", "turnon", "turnoff"]

class EnvironmentManager:
    def __init__(self, api_key: str, model: str = "gpt-4o", env_name: str = "WatchAndHelp1"):
        self.llm = OpenAIBot(
            model=model,
            use_openai=True,
            api_key=api_key,
            llm_config={'max_token': 2048},  
        )
        self.env_name = env_name
        self.json_file_path = "graph.json"
        data = {"environment": self.env_name}
        observation({"type": "full"})
        time.sleep(0.1)
        make(data)

    def generation_prompt(self, prompt: str, env_data: List[Dict]) -> Dict:
        """
        Use LLM to parse the user prompt and identify the action type and objects.
        Returns a node configuration in JSON format.
        """
        # Calculate next available ID
        existing_ids = [node['id'] for node in env_data] if env_data else []
        next_id = max(existing_ids, default=0) + 1.0
        
        object_graph = json.load(open("graph.json"))

        system_prompt = """You are a helpful assistant for understanding and manipulating 3D environments.
You must respond with ONLY valid JSON, no other text.
Given a user's natural language instruction, generate a node configuration in this exact format:
{
    "id": """ + str(next_id) + """,  # Use this exact ID
    "name": <string>,
    "relation": [
        "<object_instance> inside/on <target_object>"
    ],
    "state": [],
    "transform": [
        "X=<float> Y=<float> Z=<float>",
        "P=<float> Y=<float> R=<float>",
        "X=<float> Y=<float> Z=<float>"
    ]
}

For existing objects, use exact names from the provided list.
For new objects, use appropriate object type names (e.g., "BP_SideTable", "BP_Chair")."""

        query = f"""User instruction: {prompt}
Available objects and their IDs: {object_graph}

Remember to respond with ONLY the JSON configuration, no other text."""

        _, response = self.llm(query, sys_msg=system_prompt)
        print(response)
        
        # Clean the response to ensure it only contains JSON
        response = response.strip()
        if response.startswith('```json'):
            response = response.split('```json')[1]
        if response.startswith('```'):
            response = response.split('```')[1]
        if response.endswith('```'):
            response = response.rsplit('```', 1)[0]
        response = response.strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # If parsing fails, return a default configuration
            return {
                "id": 1.0,
                "name": "BP_DefaultObject",
                "relation": ["object inside world"],
                "state": [],
                "transform": [
                    "X=0.0 Y=0.0 Z=0.0",
                    "P=0.0 Y=0.0 R=0.0",
                    "X=1.0 Y=1.0 Z=1.0"
                ]
            }

    def update_environment(self, prompt: str) -> bool:
        """
        1. Load local environment data from JSON.
        2. Generate (or parse) a node configuration from the user prompt.
        3. If the node's ID already exists, replace it in the list. Otherwise, append it.
        4. Save back to the JSON file.
        """
        # 1. Load existing environment data from JSON
        observation({"type": "full"})
        time.sleep(0.1)
        if os.path.exists(self.json_file_path):
            with open(self.json_file_path, 'r') as f:
                json_data = json.load(f)
                # Handle both nested and flat array structures
                if isinstance(json_data, dict):
                    env_data = json_data.get(self.env_name, [])
                else:
                    env_data = json_data  # Use the array directly if not nested
        else:
            env_data = []
            json_data = {self.env_name: env_data}

        # 2. Generate node configuration
        node_config = self.generation_prompt(prompt, env_data)
        node_id = node_config['id']

        # 3. Check if the node with the same ID exists
        found_index = None
        for i, node in enumerate(env_data):
            if node['id'] == node_id:
                found_index = i
                break

        if found_index is not None:
            # Replace existing node
            env_data[found_index] = node_config
        else:
            # Append new node
            env_data.append(node_config)

        # 4. Save updated environment data back to JSON
        if isinstance(json_data, dict):
            json_data[self.env_name] = env_data
            save_data = json_data
        else:
            save_data = env_data

        with open(self.json_file_path, 'w') as f:
            json.dump(save_data, f, indent=4)
        
        return True

    def reset_environment(self):
        # Create default empty graph if file doesn't exist or is empty
        default_graph = []
        
        try:
            with open("graph.json", "r") as f:
                content = f.read().strip()
                graph = json.loads(content) if content else default_graph
        except (FileNotFoundError, json.JSONDecodeError):
            graph = default_graph
            print("Graph file not found or is empty, creating default empty graph")
            # Create the file with default empty graph
            with open("graph.json", "w") as f:
                json.dump(graph, f, indent=4)

        data = {
            "env_index": [0], 
            "graph": graph
        }

        reset(data)

    def validate_instruction(self, action: str, obs: List[Dict]) -> Tuple[bool, bool]:
        """
        Validates if an instruction is correct and effective.
        Returns (is_correct, is_effective)
        """
        try:
            # Parse action components
            parts = action.strip().split()
            if len(parts) < 3:
                return False, False
            
            # Validate agent format (should be "agent_0" or "agent_1" etc)
            if not parts[0].startswith('agent_') or not parts[0][6:].isdigit():
                return False, False
            
            agent_id = int(parts[0][6:])  # Extract number after "agent_"
            action_type = ' '.join(parts[1:-1])
            
            # Validate object ID format (should be "object_X" where X is a number)
            if not parts[-1].startswith('object_') or not parts[-1][7:].isdigit():
                return False, False
            
            object_id = float(parts[-1][7:])  # IDs are stored as floats in the graph
            
            # Validate action type
            if action_type not in ALLOWED_ACTIONS:
                return False, False
            
            # Validate object exists
            object_exists = any(obj.get('id') == object_id for obj in obs)
            if not object_exists:
                return False, False
            
            # Find the agent node by checking the relation list for BP_ThirdPersonCharacterX
            agent_node = next(
                (node for node in obs 
                 if node['name'] == "BP_ThirdPersonCharacter" and 
                 any(f"BP_ThirdPersonCharacter{agent_id}" in rel for rel in node.get('relation', []))),
                None
            )
             
            if not agent_node:
                return False, False
            
            # Basic action validation d
            if action_type == 'grab':
                # Check if agent is already holding something
                if any('hand' in rel.lower() and 'empty' not in rel.lower() for rel in agent_node.get('relation', [])):
                    return False, False
            
            # For now, consider correct instructions as potentially effective
            return True, True
            
        except Exception as e:
            print(f"Instruction validation error: {e}")
            return False, False

class DualPanelInterface:
    """
    Creates a dual-panel interface with Unreal Engine visualization on the left
    and a GPT-like control interface on the right.
    """
    def __init__(self, api_key: str, model: str = "gpt-4o", env_name: str = "WatchAndHelp1"):
        self.api_key = api_key
        self.model = model
        self.env_name = env_name
        self.env_manager = EnvironmentManager(api_key, model, env_name)
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Initialize the main window
        self.root = tk.Tk()
        self.root.title("VirtualEnv Control Interface")
        self.root.geometry("1600x900")
        
        # Create the main frame that will contain both panels
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create the left panel for Unreal Engine visualization
        self.unreal_frame = ttk.LabelFrame(self.main_frame, text="Unreal Engine Visualization")
        self.unreal_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create the right panel for GPT-like interface
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Control Interface")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add components to the control panel
        self.setup_control_panel()
        
        # Start the Unreal Engine visualization
        self.setup_unreal_panel()
        
        # Start the processing thread
        self.processing_thread = threading.Thread(target=self.process_commands)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def setup_control_panel(self):
        """Set up the GPT-like control interface"""
        # Chat history display
        self.chat_display = scrolledtext.ScrolledText(self.control_frame, wrap=tk.WORD, height=30)
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.chat_display.config(state=tk.DISABLED)
        
        # Input area
        input_frame = ttk.Frame(self.control_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.command_entry = ttk.Entry(input_frame, width=50)
        self.command_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.command_entry.bind("<Return>", self.submit_command)
        
        submit_button = ttk.Button(input_frame, text="Submit", command=self.submit_command)
        submit_button.pack(side=tk.RIGHT)
        
        # Environment controls
        control_buttons_frame = ttk.Frame(self.control_frame)
        control_buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        reset_button = ttk.Button(control_buttons_frame, text="Reset Environment", 
                                 command=self.reset_environment)
        reset_button.pack(side=tk.LEFT, padx=5)
        
        camera_label = ttk.Label(control_buttons_frame, text="Camera View:")
        camera_label.pack(side=tk.LEFT, padx=(20, 5))
        
        self.camera_var = tk.StringVar(value="First Person")
        camera_options = ["First Person", "Third Person", "Top Down", "Shoulder"]
        camera_dropdown = ttk.Combobox(control_buttons_frame, textvariable=self.camera_var, 
                                      values=camera_options, state="readonly", width=15)
        camera_dropdown.pack(side=tk.LEFT)
        camera_dropdown.bind("<<ComboboxSelected>>", self.change_camera_view)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.control_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Add initial welcome message
        self.add_to_chat("System", "Welcome to VirtualEnv Control Interface. Type commands to interact with the environment.")
        self.add_to_chat("System", "Example commands:\n- Add a coffee mug on the kitchen counter\n- agent_0 walk to object_42\n- Place a chair next to the dining table")

    def setup_unreal_panel(self):
        """Set up the Unreal Engine visualization panel"""
        # This would typically involve embedding or connecting to the Unreal Engine window
        # For this example, we'll use a placeholder frame with a message
        
        # In a real implementation, you might:
        # 1. Launch the Unreal Engine process
        # 2. Capture its window handle and embed it
        # 3. Or establish a connection to stream the visualization
        
        unreal_placeholder = ttk.Label(self.unreal_frame, 
                                      text="Unreal Engine Visualization\n\nIn a full implementation, this would show the real-time 3D environment.")
        unreal_placeholder.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # For demonstration, we could add a button to simulate launching Unreal Engine
        launch_button = ttk.Button(self.unreal_frame, text="Launch Unreal Engine", 
                                  command=self.launch_unreal_engine)
        launch_button.pack(pady=10)

    def launch_unreal_engine(self):
        """Simulate launching Unreal Engine"""
        self.add_to_chat("System", "Attempting to launch Unreal Engine...")
        self.status_var.set("Launching Unreal Engine...")
        
        # In a real implementation, you would launch the actual Unreal Engine process
        # For example:
        # subprocess.Popen(["path/to/UnrealEngine.exe", "path/to/project"])
        
        # For this example, we'll just update the status after a delay
        def update_status():
            self.add_to_chat("System", "Unreal Engine launched successfully.")
            self.status_var.set("Connected to Unreal Engine")
        
        self.root.after(2000, update_status)

    def add_to_chat(self, sender, message):
        """Add a message to the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{sender}: ", "sender")
        self.chat_display.insert(tk.END, f"{message}\n\n", "message")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
        # Apply tags for styling
        self.chat_display.tag_config("sender", foreground="blue", font=("Arial", 10, "bold"))
        self.chat_display.tag_config("message", font=("Arial", 10))

    def submit_command(self, event=None):
        """Handle command submission"""
        command = self.command_entry.get().strip()
        if not command:
            return
        
        self.add_to_chat("You", command)
        self.command_entry.delete(0, tk.END)
        self.status_var.set("Processing command...")
        
        # Add command to processing queue
        self.command_queue.put(command)

    def process_commands(self):
        """Process commands in a separate thread"""
        while True:
            try:
                command = self.command_queue.get()
                
                # Determine if this is an agent action or environment modification
                if command.startswith("agent_"):
                    # This is an agent action command
                    obs = observation({"type": "full"})
                    is_valid, is_effective = self.env_manager.validate_instruction(command, obs)
                    
                    if is_valid:
                        # Execute the action
                        set_action({"action": command})
                        response = "Command executed successfully."
                    else:
                        response = "Invalid command. Please check agent and object IDs and try again."
                else:
                    # This is an environment modification
                    success = self.env_manager.update_environment(command)
                    if success:
                        response = "Environment updated successfully."
                    else:
                        response = "Failed to update environment. Please try a different command."
                
                # Add response to queue to be displayed in the main thread
                self.response_queue.put(response)
                
                # Update UI from main thread
                self.root.after(0, self.update_ui_with_response)
                
            except Exception as e:
                error_msg = f"Error processing command: {str(e)}"
                self.response_queue.put(error_msg)
                self.root.after(0, self.update_ui_with_response)
            
            finally:
                self.command_queue.task_done()

    def update_ui_with_response(self):
        """Update UI with response from processing thread"""
        if not self.response_queue.empty():
            response = self.response_queue.get()
            self.add_to_chat("System", response)
            self.status_var.set("Ready")

    def reset_environment(self):
        """Reset the environment"""
        self.add_to_chat("System", "Resetting environment...")
        self.status_var.set("Resetting environment...")
        
        try:
            self.env_manager.reset_environment()
            self.add_to_chat("System", "Environment reset successfully.")
        except Exception as e:
            self.add_to_chat("System", f"Error resetting environment: {str(e)}")
        
        self.status_var.set("Ready")

    def change_camera_view(self, event=None):
        """Change the camera view in Unreal Engine"""
        view = self.camera_var.get()
        self.add_to_chat("System", f"Changing camera view to: {view}")
        
        # In a real implementation, you would send a command to Unreal Engine
        # to change the camera view
        
        # For this example, we'll just acknowledge the change
        self.status_var.set(f"Camera view: {view}")

    def run(self):
        """Run the interface"""
        self.root.mainloop()

def launch_dual_panel_interface(api_key: str, model: str = "gpt-4o", env_name: str = "WatchAndHelp1"):
    """Launch the dual-panel interface"""
    interface = DualPanelInterface(api_key, model, env_name)
    interface.run()

if __name__ == "__main__":
    # Example usage:
    api_key = "api_key_here"  # Replace with your actual API key
    launch_dual_panel_interface(api_key, env_name="EscapeRoom1")
