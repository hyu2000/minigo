""" for debugging a0jax training process """
import pytest
import go
from sgf_wrapper import make_sgf_from_gtp_moves


def test_make_sgf_from_gtp_moves():
    assert go.N == 5
    gtp_moves = 'C3 D3 B3 C4 B2 D1 D4 D5 E4 E3 B4 E5 B5 B1 D4 E4 C5 A2 D4 D2 A3 C4 A1 A5 C1 E1 B1 pass D4 A4'
    gtp_moves = 'C4 C3 D1 A3 D3 D2 B3 E3 D4 B2 B4 E2 A2 E4 A4 D5 C5 E1 E5 B1 D5 C1 A1 A3 A2'
    gtp_moves = 'C3 D3 B3 C4 D4 A3 D2 E3 B4 E4 B2 D5 C1 B5 E2 A5 A4 A2 C2 B1 A1 A2 C5 B5 D1 B1 pass pass'

    gtp_moves = 'C2 ' + gtp_moves
    sgf_str = make_sgf_from_gtp_moves(gtp_moves.split(), 1)
    print(sgf_str)
    with open('/Users/hyu/Downloads/make_sgf_from_gtp.C2.sgf', 'w') as f:
        f.write(sgf_str)


def process_one_game_record(line: str):
    moves = line.strip().split()
    result_str = moves.pop(0)
    result = 1 if result_str == 'B+R' else -1 if result_str == 'W+R' else 0
    return result, moves


def test_batch_score():
    """ batch analyze a0jax C2 open games """
    lines = open('/Users/hyu/PycharmProjects/a0-jax/exp-go5C2/train-1.log').readlines()
    # lines = open('/Users/hyu/PycharmProjects/a0-jax/exp-5x5/train5-2.log').readlines()
    # games = lines[8:40]
    # games = lines[373:405]
    games = [x for x in lines if x.startswith('B+R') or x.startswith('W+R')]
    print('Total %d games' % len(games))
    # print(''.join(games))

    for line in games:
        winner, moves = process_one_game_record(line)
        moves.insert(0, 'C2')
        if moves[-1] == 'pass' and moves[-2] == 'pass':
            continue
        else:
            num_moves = len(moves)
            if num_moves >= 50:
                continue
            # these games end in illegal move
            last_turn = 1 if num_moves % 2 == 1 else -1
            assert winner == last_turn   # we got the opposite
            print(f'{winner} %d %s' % (num_moves, ' '.join(moves)))
