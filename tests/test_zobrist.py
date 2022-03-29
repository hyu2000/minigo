from numbers import Number
from typing import List

import numpy as np

import go
import coords
from zobrist import DLGO_ZOBRIST_HASH, EMPTY_BOARD_HASH_19, ZobristHash
from zobrist_util import legal_moves_sans_symmetry, board_hash_canonical


assert go.N == 5


def test_empty_const():
    """ EMPTY_BOARD_HASH is based on 19x19 """
    res = 0
    for (p, color), h in DLGO_ZOBRIST_HASH.items():
        if color is None:
           res ^= h
    assert res == EMPTY_BOARD_HASH_19


def test_basic():
    ztable = ZobristHash(5)
    assert ztable.ztable.shape == (5, 5, 3)
    pos0 = go.Position()
    assert ztable.board_hash(pos0.board) == ztable.EMPTY_BOARD_HASH
    assert pos0.zobrist_hash == ztable.EMPTY_BOARD_HASH

    move0 = coords.from_gtp('C3')
    pos1 = pos0.play_move(move0)
    hash1 = ztable.board_hash(pos1.board)
    assert hash1 == ztable.hash_after_move(pos0, move0, [])
    assert pos1.zobrist_hash == hash1
    print(ztable.EMPTY_BOARD_HASH, hash1)

    move1 = coords.from_gtp('B3')
    pos2 = pos1.play_move(move1)
    hash2 = ztable.board_hash(pos2.board)
    assert ztable.board_hash_slow(pos2.board) == hash2

    # tranposition: now play pass, white B3, C3, we should reach the same board & hash
    pos1 = pos0.play_move(None)
    pos2 = pos1.play_move(move1)
    pos3 = pos2.play_move(move0)
    new_hash2 = ztable.board_hash(pos3.board)
    assert new_hash2 == hash2
    assert hash2 == ztable.hash_after_move(pos2, move0, [])

    # test remove stones: a bit contrived, not real capture


def test_hash_canonical():
    move1 = coords.from_gtp('B3')
    pos1 = go.Position().play_move(move1)
    for move in ['C2', 'C4', 'D3']:
        pos2 = go.Position().play_move(coords.from_gtp(move))
        assert board_hash_canonical(pos1.board) == board_hash_canonical(pos2.board)


def show_legal_moves(legal_moves_1d: np.ndarray):
    moves_on_board = legal_moves_1d[:-1].reshape((go.N, go.N))
    print(moves_on_board)


def test_filter_legal_moves():
    pos0 = go.Position()
    legal_moves_sans_s6y = legal_moves_sans_symmetry(pos0)
    assert sum(legal_moves_sans_s6y) - 1 == 6
    print()
    show_legal_moves(legal_moves_sans_s6y)

    pos1 = pos0.play_move(coords.from_gtp('C2'))
    legal_moves_sans_s6y = legal_moves_sans_symmetry(pos1)
    # mirror symmetry
    assert sum(legal_moves_sans_s6y) - 1 == 14
    show_legal_moves(legal_moves_sans_s6y)

    pos2 = pos1.play_move(coords.from_gtp('C3'))
    legal_moves = pos2.all_legal_moves()
    assert sum(legal_moves) - 1 == 23
    # mirror symmetry
    legal_moves_sans_s6y = legal_moves_sans_symmetry(pos2)
    assert sum(legal_moves_sans_s6y) - 1 == 13
    show_legal_moves(legal_moves_sans_s6y)

    pos3 = pos2.play_move(coords.from_gtp('D3'))
    legal_moves = pos3.all_legal_moves()
    assert sum(legal_moves) - 1 == 22
    # diagonal symmetry
    legal_moves_sans_s6y = legal_moves_sans_symmetry(pos3)
    assert sum(legal_moves_sans_s6y) - 1 == 13
    show_legal_moves(legal_moves_sans_s6y)


def test_unique_states_in_selfplay():
    """ count #unique states (bucketed by move#) in a set of selfplay games
    5x5-2021 selfplay9_1, C2 only(?!)
        40 sgfs
        #zhash:         [1, 1, 10, 15, 21, 25, 27, 27, 28, 30, 31, 31, 33, 35, 38, 37]
        #canonical:     [1, 1, 9, 13, 19, 20, 22, 22, 23, 26, 27, 27, 29, 31, 34, 35]
    model13_2, C2/B2 open, reduce_symmetry: (due to Benson, shortest game has 16 moves)
        40 sgfs
        #zhash:         [1, 2, 7, 11, 17, 24, 26, 25, 24, 23, 23, 24, 24, 26, 26, 26]
        #canonical:     [1, 2, 7, 10, 16, 22, 24, 24, 23, 22, 21, 23, 23, 24, 22, 22]
        80 sgfs:
        #zhash:         [1, 2, 9, 15, 23, 38, 45, 44, 44, 44, 43, 44, 44, 47, 47, 48, 50, 48, 48, 47]

    """
    import os
    from sgf_wrapper import replay_sgf_file
    import myconf

    sgf_dir = f'{myconf.SELFPLAY_DIR}/sgf/full'
    # sgf_dir = f'{myconf.EXP_HOME}/../5x5-2021/selfplay9_1'
    game_hashes = []  # type: List[List[Number]]
    game_moves = []   # type: List[List[str]]
    NUM_SGFS = 80
    sgf_fnames = [x for x in os.listdir(sgf_dir)[:NUM_SGFS] if x.endswith('.sgf')]
    print('Use first %d sgfs' % len(sgf_fnames))
    for sgf_fname in sgf_fnames:
        hashes_in_game = [pwc.position.zobrist_hash for pwc in replay_sgf_file(f'{sgf_dir}/{sgf_fname}')]
        # hashes_in_game = [board_hash_canonical(pwc.position.board) for pwc in replay_sgf_file(f'{sgf_dir}/{sgf_fname}')]
        game_hashes.append(hashes_in_game)
        game_moves.append([coords.to_gtp(pwc.next_move) for pwc in replay_sgf_file(f'{sgf_dir}/{sgf_fname}')])

    num_states_per_step = []
    NUM_MOVES = 20
    for imove in range(NUM_MOVES):
        # if game is shorter than imove, use last hash
        hash_set = {gh[imove] if imove < len(gh) else gh[-1] for gh in game_hashes}
        num_states_per_step.append(len(hash_set))
    print(num_states_per_step)

    game_fmted = [str(len(x)) + ' ' + ' '.join(x) for x in game_moves]
    print('\n'.join(game_fmted))

