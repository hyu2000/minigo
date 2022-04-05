from collections import Counter
from numbers import Number
from typing import List

import numpy as np

import go
import coords
from sgf_wrapper import SGFReader
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


def format_long_array(lst, every=5, istart=0) -> str:
    """ add index to make long array easier to read """
    slst = [f'{idx}:{x}' if (idx % every == 0 or idx == istart) else str(x)
            for idx, x in zip(range(istart, istart+len(lst)), lst)]
    return '[%s]' % ', '.join(slst)


def test_unique_states_in_selfplay():
    """ count #unique states (bucketed by move#) in a set of selfplay games
    5x5-2021 selfplay9_1, C2 only(?!)
        40 sgfs
        #zhash:         [1, 1, 10, 15, 21, 25, 27, 27, 28, 30, 31, 31, 33, 35, 38, 37]
        #canonical:     [1, 1, 9, 13, 19, 20, 22, 22, 23, 26, 27, 27, 29, 31, 34, 35]
    model13_2, C2/B2 open, reduce_symmetry: (due to Benson, shortest game has 16 moves_in_game)
        40 sgfs
        #zhash:         [1, 2, 7, 11, 17, 24, 26, 25, 24, 23, 23, 24, 24, 26, 26, 26, 27, 25, 25, 25]
        #canonical:     [1, 2, 7, 10, 16, 22, 24, 24, 23, 22, 21, 23, 23, 24, 22, 22]
        80 sgfs:
        #zhash:         [1, 2, 9, 15, 23, 38, 45, 44, 44, 44, 43, 44, 44, 47, 47, 48, 50, 48, 48, 47]
        100 sgfs        [1, 2, 15, 23, 33, 41, 50, 48, 46, 47, 46, 46, 46, 48, 48, 49, 50, 50, 50, 54]
        400 sgfs        [1, 2, 26, 45, 71, 103, 128, 122, 120, 123, 120, 128, 128, 134, 134, 139, 144, 138, 138, 130]
            move #5: total 400 ['0.20', '0.16', '0.07', '0.07', '0.05', '0.04', '0.03', '0.02', '0.02', '0.02']
            move #10: total 400 ['0.14', '0.12', '0.12', '0.05', '0.04', '0.03', '0.03', '0.03', '0.02', '0.02']
            move #20: total 309 ['0.16', '0.12', '0.05', '0.04', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03']
            move #30: total 17 ['0.12', '0.12', '0.06', '0.06', '0.06', '0.06', '0.06', '0.06', '0.06', '0.06']
        1000 sgfs       [1, 2, 28, 67, 139, 213, 268, 263, 260, 262, 256, 266, 268, 280, 281, 290, 300, 295, 293, 272]
            move #5: total 1000 ['0.19', '0.18', '0.06', '0.06', '0.04', '0.04', '0.03', '0.03', '0.03', '0.02']
            move #10: total 999 ['0.15', '0.10', '0.10', '0.05', '0.04', '0.03', '0.03', '0.03', '0.03', '0.02']
            move #15: total 998 ['0.15', '0.10', '0.10', '0.05', '0.03', '0.03', '0.03', '0.02', '0.02', '0.02']
            move #20: total 761 ['0.13', '0.11', '0.05', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.02']

536 unique games (out of 1000)
count #moves
 147  18	 B+0.5	C2 C3 D3 B3 D4 B2 C1 C4 C5 B5 D5 B1 D2 A4 A2 E4 E3 E1
  31  25	 B+1.5	C2 C3 D3 D2 D1 E2 C4 B3 B2 D4 B4 A3 A2 E3 A4 D3 D5 B5 C5 B1 C1 E4 A1 pass pass
  24  27	 B+4.5	C2 C3 D3 D2 D1 E2 C4 B3 B2 D4 B4 A3 A2 E3 A4 D3 D5 B5 C5 B1 C1 pass A5 E4 A1 pass pass
  21  25	 B+4.5	C2 C3 D3 D2 D1 B2 E2 C4 D4 D5 B1 A2 B4 B3 A4 E4 E3 B5 E5 C5 E4 A1 C1 pass pass
  16  26	 W+5.5	B2 C3 D3 C2 C4 B3 D2 B4 D4 D1 B5 A2 C5 A4 B1 C1 E2 E4 A5 E5 E1 A1 B2 B1 pass pass
  13  28	 W+0.5	B2 C3 D4 C2 C4 B3 B4 A4 D2 D3 E3 D1 E2 A2 B1 E4 E5 B5 C5 A3 A5 C1 B5 A1 B1 B2 pass pass
  13  25	 B+0.5	C2 C3 D3 D2 D1 D4 E2 B3 B2 A2 E4 D5 B4 C4 B5 B1 C1 A4 E3 A3 E5 A5 A1 C5 B1
  11  25	 B+2.5	C2 C3 D3 D2 D1 E2 C4 B3 B2 D4 B4 A3 A2 E3 A4 D3 D5 B5 C5 B1 C1 pass A5 pass pass
  10  26	 W+9.5	B2 C3 C2 D2 C4 B3 D3 D4 B4 A3 E3 E2 A4 E4 D3 E3 A2 D3 D1 C5 D5 E5 C1 B5 E1 A5
  10  27	 B+4.5	C2 C3 D3 D2 D1 E2 C4 B3 B2 D4 B4 A3 A2 E3 A4 D3 D5 B5 C5 B1 C1 pass A5 E5 E4 pass pass
  10  26	 W+9.5	B2 C3 C2 D2 B4 B3 D3 D4 C4 A3 E3 E2 A4 E4 D3 E3 A2 D3 D1 C5 D5 E5 C1 B5 E1 A5
  10  24	 B+2.5	C2 C3 D3 D2 D1 D4 E2 B3 B2 A2 E4 D5 B4 C4 B5 B1 C1 A4 E3 A3 E5 A1 pass pass

preferred game (no noise):
      18	 B+0.5	C2 C3 D3 B3 D4 B2 C1 C4 C5 B5 D5 B1 D2 A4 A2 E4 E3 E1
      24	 W+1.5	B2 C3 C2 D2 C4 B3 D3 D4 E3 B4 E4 A2 D1 D5 E2 C5 A3 A4 E5 A3 B1 A1 B5 A5
    """
    """
  5x5-2021/  selfplay7-sgfs/:  no reduce-symmetry
test_zobrist.py::test_unique_states_in_selfplay Use first 1000 sgfs
[1, 2, 46, 122, 259, 414, 474, 610, 633, 707, 734, 804, 819, 862, 873, 890, 912, 926, 947, 962]
move #5: total 1000 ['0.12', '0.09', '0.07', '0.06', '0.03', '0.02', '0.01', '0.01', '0.01', '0.01']
move #10: total 1000 ['0.06', '0.03', '0.02', '0.02', '0.02', '0.02', '0.02', '0.01', '0.01', '0.01']
move #15: total 1000 ['0.01', '0.01', '0.01', '0.01', '0.01', '0.01', '0.01', '0.01', '0.01', '0.01']
move #20: total 1000 ['0.01', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00']
999 unique games (out of 1000)
   2  27	 B+0.5	C2 C3 B3 B2 B1 B4 A2 D3 A4 B5 D2 E2 D4 C4 D5 D1 C1 E3 A3 C5 A5 E4 E1 E5 D1 pass pass
   1  ...
   
   selfplay7: reduce-symmetry
test_zobrist.py::test_unique_states_in_selfplay Use first 1000 sgfs
[1, 2, 25, 60, 113, 176, 212, 210, 206, 201, 205, 205, 214, 218, 227, 234, 247, 240, 248, 235]
move #5: total 1000 ['0.30', '0.26', '0.03', '0.03', '0.02', '0.02', '0.02', '0.02', '0.02', '0.01']
move #10: total 1000 ['0.29', '0.12', '0.09', '0.04', '0.03', '0.03', '0.03', '0.02', '0.02', '0.02']
move #15: total 1000 ['0.29', '0.12', '0.09', '0.04', '0.03', '0.02', '0.02', '0.01', '0.01', '0.01']
move #20: total 951 ['0.30', '0.11', '0.07', '0.04', '0.02', '0.02', '0.02', '0.02', '0.01', '0.01']
449 unique games (out of 1000)
 270  24	 B+0.5	B2 C3 C2 D2 B3 C4 D1 E2 D4 D3 B4 B5 E4 C1 A4 E1 B1 D5 A5 C5 A2 D1 pass pass
  53  27	 B+24.5	C2 C3 D3 D2 D1 B2 E2 C4 D4 D5 B1 A2 B4 B3 A4 E4 E3 C5 C1 B5 E5 A3 E4 A5 A4 B4 A1
  41  25	 B+0.5	C2 C3 D3 D2 D1 E2 C4 B3 B2 D4 B4 A3 A4 E3 A2 D3 D5 B1 C1 B5 C5 pass E5 pass pass
  32  27	 B+24.5	C2 C3 D3 D2 D1 B2 E2 C4 D4 D5 B1 A2 B4 B3 A4 E4 E3 C5 C1 B5 E5 A5 E4 A3 A4 B4 A1
  19  27	 B+24.5	C2 C3 D3 D2 D1 B2 E2 C4 D4 D5 B1 A2 B4 B3 A4 E4 E3 C5 C1 A5 E5 B5 E4 A3 A4 B4 A1
  13  25	 B+0.5	C2 C3 D3 D2 D1 D4 E2 B3 E4 D5 B4 C4 B2 A2 B5 B1 C1 A3 E3 A5 E5 C5 A1 A4 B1
  13  25	 B+0.5	C2 C3 D3 D2 D1 E2 C4 B3 B2 D4 B4 A3 A4 E3 A2 D3 D5 B5 C5 B1 C1 pass E5 pass pass
  12  28	 B+24.5	B2 C3 C2 D2 B3 C4 D1 E2 D4 D3 B4 B5 E4 C1 A4 E1 B1 D5 A5 C5 A2 E3 D1 E5 C1 pass D4 E4
  12  25	 B+0.5	C2 C3 D3 D2 D1 D4 E2 B3 E4 D5 B4 C4 B2 A2 B5 B1 C1 A3 E3 A5 A1 A4 B1 C5 E5
  11  24	 W+1.5	C2 C3 C4 D3 B3 B2 B1 B4 A2 D4 D2 E2 A4 B5 D1 C5 E3 E4 A3 A5 D5 E5 E1 E3
      """
    import os
    import myconf

    sgf_dir = f'{myconf.SELFPLAY_DIR}/sgf/full'
    # sgf_dir = f'{myconf.EXP_HOME}/../5x5-2021/selfplay9_1'
    game_hashes = []  # type: List[List[Number]]
    game_moves = []   # type: List[List[str]]
    game_results = []  # type: List[str]
    NUM_SGFS = 1000
    sgf_fnames = [x for x in os.listdir(sgf_dir)[:NUM_SGFS] if x.endswith('.sgf')]
    print('Use first %d sgfs' % len(sgf_fnames))
    void_games = []
    for sgf_fname in sgf_fnames:
        hashes_in_game, moves_in_game = [], []
        reader = SGFReader.from_file_compatible(f'{sgf_dir}/{sgf_fname}')
        if reader.result_str() == 'VOID':
            void_games.append(sgf_fname)
            continue
        for pwc in reader.iter_pwcs():
            hashes_in_game.append(pwc.position.zobrist_hash)
            moves_in_game.append(coords.to_gtp(pwc.next_move))
        # hashes_in_game = [board_hash_canonical(pwc.position.board) for pwc in replay_sgf_file(f'{sgf_dir}/{sgf_fname}')]
        game_hashes.append(hashes_in_game)
        game_moves.append(moves_in_game)
        game_results.append(reader.result_str())
    if len(void_games) > 0:
        print('Found void games: %s' % void_games)

    num_states_per_step = []
    NUM_MOVES = 20
    for imove in range(1, NUM_MOVES):
        hash_set = {gh[imove] for gh in game_hashes if imove < len(gh)}
        num_states_per_step.append(len(hash_set))
    print('#unique-states for each move#:\n\t%s' % format_long_array(num_states_per_step, istart=1))

    # detailed distribution at certain move#
    for imove in [5, 10, 15, 20]:
        cnter = Counter(gh[imove] for gh in game_hashes if imove < len(gh))
        total = sum(cnter.values())
        print(f'move #{imove}: total {total}', ['%.2f' % (cnt / total) for (x, cnt) in cnter.most_common(10)])

    # freq of games played
    games_fmted = [f'{result}\t' + ' '.join(moves) for moves, result in zip(game_moves, game_results)]
    cnter = Counter(games_fmted)
    print(f'%d unique games (out of %d)' % (len(cnter), len(sgf_fnames)))
    for move_str, freq in cnter.most_common():
        print("%4d %3d\t %s" % (freq, len(move_str.split()) - 1, move_str))

