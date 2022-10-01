import pickle
import logging
import multiprocessing as mp

import numpy as np
import zmq

import coords
import go
import myconf
from k2net import DualNetwork


SERVER_PORT = 5555


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


class DNNServer:
    def __init__(self, model_fname, port: int):
        self.dnn = DualNetwork(model_fname)
        self.port = port

        self.start()

    def start(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        # was "*:port"
        socket.bind(f"tcp://127.0.0.1:{self.port}")

        while True:
            message = socket.recv()
            if len(message) == 0:
                print('server exiting')
                break

            data = pickle.loads(message)
            assert isinstance(data, tuple)

            pos = data[0]
            assert isinstance(pos, go.Position)
            resp = self.dnn.run(pos)
            logging.info(f'raw_value=%.1f', resp[1])
            socket.send(pickle.dumps(resp))


def start_server_remote(model_fname, port):
    server_proc = mp.Process(target=DNNServer, args=(model_fname, port))
    server_proc.start()


def test_server():
    start_server_remote(f'{myconf.MODELS_DIR}/model5_epoch2.h5', SERVER_PORT)
    stub = Stub(SERVER_PORT)
    pos0 = go.Position()
    priors, raw_value = stub.remote_call(pos0)
    best_c = np.argmax(priors)
    print(coords.flat_to_gtp(best_c), raw_value)

    pos1 = pos0.play_move(coords.from_gtp('D4'))
    priors, raw_value = stub.remote_call(pos1)
    best_c = np.argmax(priors)
    print(coords.flat_to_gtp(best_c), raw_value)
    # print(type(priors), len(priors), raw_value)

    stub.send_eof()
