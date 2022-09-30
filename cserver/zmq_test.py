import time

import zmq
import multiprocessing as mp


def start_server(port: int):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)

    while True:
        #  Wait for next request from client
        message = socket.recv()
        print("Received request: ", message)
        if len(message) == 0:
            print('server exiting')
            break

        time.sleep(0.1)
        socket.send(b"World from server")


def start_client(server_port: int):
    context = zmq.Context()
    print("Connecting to server...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:%s" % server_port)

    for request in range(2):
        print("Sending request ", request, "...")
        socket.send(b"Hello")
        #  Get the reply.
        message = socket.recv()
        print("Received reply ", request, "[", message, "]")
    socket.send(b'')


def test_run():
    port = 5555
    server_proc = mp.Process(target=start_server, args=(port,))
    server_proc.start()

    start_client(port)

    server_proc.join()