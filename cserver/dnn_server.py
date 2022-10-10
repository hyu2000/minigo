"""
- go.Position is heavy. we pass it around so server side code is easy (hash, etc.)
- zmq: can we peek into the queue to batch things across requests, now that we have cache?
"""
import os
import pickle
import logging
import multiprocessing as mp
import sys
from typing import List, Tuple

import numpy as np
import zmq

import coords
import go
import myconf
from k2net import DualNetwork


SERVER_PORT = 5555


class RemoteMethods:
    RUN_MANY = 1
    GET_MODEL_ID = 2


class DNNStub:
    def __init__(self, server_port: int = SERVER_PORT, model_file: str = None):
        """
        model_file, if specified, is only for checking that the server is running the same model
        """
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        # self.socket.connect(f"tcp://localhost:{server_port}")
        self.socket.connect(f"ipc:///tmp/zmqdnn{server_port}")

        #
        self.model_id = self._remote_call(RemoteMethods.GET_MODEL_ID)
        if model_file:
            assert os.path.basename(self.model_id) == os.path.basename(model_file)

    def run(self, position: go.Position):
        # adapt from run_many()
        priors, values = self.run_many([position])
        return priors[0], values[0]

    def run_many(self, positions: List[go.Position]) -> Tuple[np.ndarray, np.ndarray]:
        """ stub for dnn.run_many() """
        return self._remote_call(RemoteMethods.RUN_MANY, positions)

    def _remote_call(self, *args):
        payload = pickle.dumps(args)
        self.socket.send(payload)
        message = self.socket.recv()
        result = pickle.loads(message)
        # logging.info('sent %d bytes, response %d bytes', len(payload), len(message))
        return result

    def send_eof(self):
        self.socket.send(b'')


class DNNServer:
    """ centralized DNN compute server with caching """
    def __init__(self, model_fname, port: int = SERVER_PORT):
        self.dnn = DualNetwork(model_fname)
        self.port = port

        self.cache = dict()
        self._num_req_positions = 0
        self._num_pos_evals = 0

        # start it now!
        self.start()

    def start(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        # socket.bind(f"tcp://127.0.0.1:{self.port}")  # was "*:port"
        socket.bind(f"ipc:///tmp/zmqdnn{self.port}")
        logging.info(f'DNNServer started at {self.port}')

        while True:
            message = socket.recv()
            if len(message) == 0:
                self._summarize()
                break

            data = pickle.loads(message)
            assert isinstance(data, tuple)
            # dispatch
            method_idx = data[0]
            if method_idx == RemoteMethods.RUN_MANY:
                resp = self._run_many_with_cache(data[1])
            elif method_idx == RemoteMethods.GET_MODEL_ID:
                resp = self._get_model_id()
            else:
                raise Exception(f'Invalid remote method: {method_idx}')

            socket.send(pickle.dumps(resp))

    def _get_model_id(self):
        return self.dnn.model_id

    def _run_many_with_cache(self, pos_list: List[go.Position]):
        assert isinstance(pos_list, list)

        zhashes = [pos.zobrist_hash for pos in pos_list]
        results = [self.cache.get(x) for x in zhashes]

        idx_to_calc = [i for i, result in enumerate(results) if result is None]
        if len(idx_to_calc) > 0:
            pos_to_calc = [pos_list[i] for i in idx_to_calc]
            priors, values = self.dnn.run_many(pos_to_calc)
            # cache
            for i, pos in enumerate(pos_to_calc):
                # make a ndarray.copy for caching
                self.cache[pos.zobrist_hash] = (priors[i].copy(), values[i])

            # backfill results, assemble response
            for i, idx in enumerate(idx_to_calc):
                results[idx] = (priors[i], values[i])

        all_priors = np.stack([result[0] for result in results])
        all_values = np.stack([result[1] for result in results])
        # logging.info(f'%d -> %s %s', len(pos_list), all_priors.shape, all_values.shape)

        if self._num_req_positions // 10000 != (self._num_req_positions + len(pos_list)) // 10000:
            logging.info(f'Total %d entries, #pos={self._num_req_positions}, #calc={self._num_pos_evals}', len(self.cache))
        self._num_req_positions += len(pos_list)
        self._num_pos_evals += len(idx_to_calc)
        return all_priors, all_values

    def _summarize(self):
        logging.info('server exiting')
        logging.info(f'Total %d entries, #pos={self._num_req_positions}, #calc={self._num_pos_evals}', len(self.cache))


def start_server_remote(model_fname, port):
    server_proc = mp.Process(target=DNNServer, args=(model_fname, port))
    server_proc.start()


def test_server():
    """ we save 15% of eval calls.
    The diff between #calc and #entries indicates that for the same batch, there might be dup positions...
2022-10-02 14:09:05,316 INFO Total 48634 entries, #pos=56656, #calc=48789   4 games
2022-10-02 14:49:02,927 INFO Total 428101 entries, #pos=500000, #calc=431083
2022-10-02 14:58:13,379 INFO Total 760160 entries, #pos=899668, #calc=765432
    """
    start_server_remote(f'{myconf.MODELS_DIR}/model5_epoch2.h5', SERVER_PORT)
    stub = DNNStub(SERVER_PORT)

    assert 'model5_epoch2' in stub.model_id

    pos0 = go.Position()
    pos1 = pos0.play_move(coords.from_gtp('D4'))
    pos2 = pos1.play_move(coords.from_gtp('E5'))
    positions = [pos0, pos1, pos2]

    # test run()
    prior0, raw_value0 = stub.run(pos0)
    print(prior0.shape, raw_value0)

    for i in range(3):
        priors, raw_values = stub.run_many(positions[:i+1])
        best_moves = [coords.flat_to_gtp(x) for x in np.argmax(priors, axis=1)]
        logging.info(f'try {i}: {best_moves}, {raw_values}')

    stub.send_eof()


def test_shutdown_server():
    stub = DNNStub(SERVER_PORT)
    stub.send_eof()


def start_server(argv):
    """ Usage: start_server <model_file> [port]
    """
    port = SERVER_PORT if len(argv) == 2 else argv[2]
    load_file = argv[1]
    logging.info(f'Starting server {port}: {load_file}')
    DNNServer(load_file, port)


if __name__ == '__main__':
    start_server(sys.argv)
