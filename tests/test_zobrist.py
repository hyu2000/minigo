import glob
import logging
import os
import random
import time
from itertools import islice

import numpy as np

import coords
import go
import myconf
from sgf_wrapper import SGFReader
from zobrist_util import legal_moves_sans_symmetry, legal_moves_cache_size

assert go.N == 9


def show_legal_moves(legal_moves_1d: np.ndarray):
    moves_on_board = legal_moves_1d[:-1].reshape((go.N, go.N))
    print(moves_on_board)


def test_basic():
    """
    """
    center_moves = [f'{col}{row}' for col in list('CDEFG') for row in range(3, 8)]

    pos0 = go.Position()
    legal_moves_sans_s6y = legal_moves_sans_symmetry(pos0)
    show_legal_moves(legal_moves_sans_s6y)
    assert legal_moves_sans_s6y.sum() == 15 + 1

    move0 = 'E5'
    pos1 = pos0.play_move(coords.from_gtp(move0))
    legal_moves_sans_s6y = legal_moves_sans_symmetry(pos1)
    show_legal_moves(legal_moves_sans_s6y)
    assert legal_moves_sans_s6y.sum() == 14 + 1

    for move1 in ['E4', 'E3', 'D4', 'D3', 'C3']:
        pos2 = pos1.play_move(coords.from_gtp(move1))
        legal_moves_sans_s6y = legal_moves_sans_symmetry(pos2)

        # pick a random move, count legal moves after it
        for move2 in random.sample(center_moves, k=3):
            if move2 != move0 and move2 != move1:
                break

        pos3 = pos2.play_move(coords.from_gtp(move2))
        logging.info(f'{move1}: %d -> %d, {move2}: %d -> %d',
                     pos2.all_legal_moves().sum(), legal_moves_sans_s6y.sum(),
                     pos3.all_legal_moves().sum(), legal_moves_sans_symmetry(pos3).sum())
        # show_legal_moves(legal_moves_sans_s6y)

    logging.info('legal_move cache size = %d', legal_moves_cache_size())


def test_real_sgf():
    """ see how much we save in real games
    All within first 3 moves

'22:54:17        INFO 0-80449152228.sgf'
'22:54:17        INFO E5 1 81 -> 15'
'22:54:17        INFO E3 2 80 -> 44'
'22:54:17        INFO 0-43666695417.sgf'
'22:54:17        INFO E5 1 81 -> 15'
'22:54:17        INFO 0-53042963659.sgf'
'22:54:17        INFO E5 1 81 -> 15'
'22:54:17        INFO E3 2 80 -> 44'
'22:54:17        INFO E4 3 79 -> 43'
    """
    sgf_glob_pattern = f'{myconf.EXP_HOME}/selfplay/sgf/full/0-*.sgf'
    sgf_fnames = glob.glob(sgf_glob_pattern)[-10:]
    print(f'Use first %d sgfs in {sgf_glob_pattern}' % len(sgf_fnames))

    for sgf_fname in sgf_fnames:
        reader = SGFReader.from_file_compatible(f'{sgf_fname}')
        logging.info('%s', os.path.basename(sgf_fname))
        for pwc in islice(reader.iter_pwcs(), 1, 10):
            pos = pwc.position  # type: go.Position
            num_all_moves = pos.all_legal_moves().sum()
            num_uniq_moves = legal_moves_sans_symmetry(pos).sum()
            if num_all_moves > num_uniq_moves:
                logging.info(f'%s {pos.n} %d -> %d', coords.to_gtp(pos.recent[-1].move),
                             num_all_moves, num_uniq_moves)


def test_perf_filter_legal_moves():
    """
    pos0 400 calls: 1.1 sec
    pos2 400 calls: 1.7 sec
    """
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

        logging.info('legal_move cache size = %d', legal_moves_cache_size())

    # legal_moves_sans_s6y = legal_moves_sans_symmetry(pos1)
    # logging.info(len(legal_moves_sans_s6y))
    # show_legal_moves(legal_moves_sans_s6y)

    legal_moves = pos2.all_legal_moves()
    # mirror symmetry
    legal_moves_sans_s6y = legal_moves_sans_symmetry(pos2)
    logging.info(legal_moves_sans_s6y.sum())
    show_legal_moves(legal_moves_sans_s6y)
