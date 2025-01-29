import socketio
import json

sio = socketio.Client()

@sio.event
def connect():
    print("Connected to server")

@sio.event
def disconnect():
    print("Disconnected from server")

@sio.event
def message(data):
    try:
        with open("message.json", "r") as f:
            message = json.load(f)
    except FileNotFoundError:
        message = []

    message.append(data)
    
    with open("message.json", "w") as f:
        json.dump(message, f, indent=4)

@sio.event
def graph(data):
    with open("graph.json", "w") as f:
        json.dump(data, f, indent=4)


def make(data):
    sio.emit('make', data)

def reset(data):
    sio.emit('reset', data)

def observation(data):
    sio.emit('observation', data)

def set_action(data):
    sio.emit('set_action', data)

def system_agent(data):
    sio.emit('system_agent', data)

def render(data):
    sio.emit('render', data)


sio.connect('http://localhost:5000')