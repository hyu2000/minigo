import logging
import time

import numpy as np

import coords
import go
from zobrist_util import legal_moves_sans_symmetry


def show_legal_moves(legal_moves_1d: np.ndarray):
    moves_on_board = legal_moves_1d[:-1].reshape((go.N, go.N))
    print(moves_on_board)


def test_perf_filter_legal_moves():
    """
    pos0 400 calls: 1.1 sec
    pos2 400 calls: 1.7 sec
    """
    assert go.N == 9

    logging.info('start')
    pos0 = go.Position()
    pos1 = pos0.play_move(coords.from_gtp('E5'))
    pos2 = pos1.play_move(coords.from_gtp('E3'))

    N = 400
    for i, pos in enumerate((pos0, pos1, pos2)):
        stime = time.time()
        for _ in range(N):
            legal_moves_sans_s6y = legal_moves_sans_symmetry(pos)
            # logging.info(legal_moves_sans_s6y.sum())
        telapsed = time.time() - stime

        print(f'{i}: {N} calls takes {telapsed:.1f} seconds')
        logging.info(legal_moves_sans_s6y.sum())
        # show_legal_moves(legal_moves_sans_s6y)

    # legal_moves_sans_s6y = legal_moves_sans_symmetry(pos1)
    # logging.info(len(legal_moves_sans_s6y))
    # show_legal_moves(legal_moves_sans_s6y)

    legal_moves = pos2.all_legal_moves()
    # mirror symmetry
    legal_moves_sans_s6y = legal_moves_sans_symmetry(pos2)
    logging.info(legal_moves_sans_s6y.sum())
    show_legal_moves(legal_moves_sans_s6y)
