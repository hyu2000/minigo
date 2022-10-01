import time
import pickle

import numpy as np
import pandas as pd
import zmq
import multiprocessing as mp
import logging
import go


def start_server(port: int):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    # was "*:port"
    socket.bind(f"tcp://127.0.0.1:{port}")

    while True:
        #  Wait for next request from client
        message = socket.recv()
        print("Received request: ", len(message))
        if len(message) == 0:
            print('server exiting')
            break

        data = pickle.loads(message)
        assert isinstance(data, tuple)
        print(len(data), type(data[0]))

        time.sleep(0.1)
        if len(data) == 2:
            resp = b"World from server"
        else:
            resp = np.ones((2, 3))
        socket.send(pickle.dumps(resp))


def start_client0(server_port: int):
    context = zmq.Context()
    print("Connecting to server...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:%s" % server_port)

    objs = [(3, 'hello'), np.arange(0, 1, .2), pd.DataFrame({'a': range(4)})]
    for i, obj in enumerate(objs):
        data = pickle.dumps(obj)
        print(f"Sending request {i}", type(obj), len(data), "...")
        socket.send(data)
        #  Get the reply.
        message = socket.recv()
        print("Received reply ", "[", message, "]")

    socket.send(b'')


class Stub:
    def __init__(self, server_port: int):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{server_port}")

    def remote_call(self, *args):
        payload = pickle.dumps(args)
        self.socket.send(payload)
        message = self.socket.recv()
        result = pickle.loads(message)
        logging.info('sent %d bytes, response %d bytes', len(payload), len(message))
        return result

    def send_eof(self):
        self.socket.send(b'')


def start_client(server_port):
    stub = Stub(server_port)
    pos = go.Position()
    result = stub.remote_call(pos)
    print(result)
    result = stub.remote_call(np.ones(3))
    print(result)
    stub.send_eof()


def test_run0():
    port = 5555
    server_proc = mp.Process(target=start_server, args=(port,))
    server_proc.start()

    start_client(port)

    server_proc.join()