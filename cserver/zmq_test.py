import time
import pickle

import numpy as np
import pandas as pd
import zmq
import multiprocessing as mp


def start_server(port: int):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)

    while True:
        #  Wait for next request from client
        message = socket.recv()
        print("Received request: ", len(message))
        if len(message) == 0:
            print('server exiting')
            break

        data = pickle.loads(message)
        print(type(data), len(data))
        time.sleep(0.1)
        socket.send(b"World from server")


def start_client(server_port: int):
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


def test_run():
    port = 5555
    server_proc = mp.Process(target=start_server, args=(port,))
    server_proc.start()

    start_client(port)

    server_proc.join()