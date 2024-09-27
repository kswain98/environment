import socketio
import eventlet

sio = socketio.Server()
app = socketio.WSGIApp(sio)

clients = []

@sio.event
def connect(sid, environ):
    print('Client connected:', sid)
    clients.append(sid)

@sio.event
def disconnect(sid):
    print('Client disconnected:', sid)
    if sid in clients:
        clients.remove(sid)

@sio.event
def message(sid, data):
    sio.emit('message', data)

@sio.event
def graph(sid, data):
    sio.emit('graph', data)

@sio.event
def make(sid, data):
    sio.emit('make', data)

@sio.event
def reset(sid, data):
    sio.emit('reset', data)

@sio.event
def observation(sid, data):
    sio.emit('observation', data)

@sio.event
def set_action(sid, data):
    sio.emit('set_action', data)

@sio.event
def system_agent(sid, data):
    sio.emit('system_agent', data)

@sio.event
def render(sid, data):
    sio.emit('render', data)

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 8000)), app)