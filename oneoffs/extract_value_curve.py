""" extract value estimates of both sides from sgf
  so that we can plot value curve throughout a game
"""
from sgf_wrapper import replay_sgf, translate_sgf_move, make_sgf, make_sgf_from_gtp_moves, SGFReader
import coords


SGF_DIR = '/Users/hyu/PycharmProjects/dlgo/9x9-exp2/eval_review/kata1_5k/model6_2.mlpackage#400'


def extract_values_from_sgf(sgf_file1):
    reader = SGFReader.from_file_compatible(sgf_file1)
    values = []
    for idx, (move, comments) in enumerate(reader.iter_comments()):
        comment = comments[0]
        if not comment:
            print(f'empty comment at move {idx}')
            continue
        if idx == 0 and comment.startswith('Resign Thresh'):
            comment = comment.split('\n')[1]

        value_str = comment.split()[0]
        if value_str.startswith('Q='):
            # normalize k2net value from [-1, 1] to [0, 1]
            value = float(value_str[2:])
            value = (value + 1) / 2
        else:
            value = float(value_str)
        values.append(value)
        # print(idx, move, value_str)
    return values


def test_extract_values():
    sgf_file1 = f'{SGF_DIR}/kata1_5k#400-vs-model6_2#400-0-57660170497.sgf'
    sgf_file2 = f'{SGF_DIR}/kata1_5k#400-vs-model6_2#400-1-57695918266.sgf'
    values = extract_values_from_sgf(sgf_file1)
    print(values)


def test_extract_from_selfplay():
    sgf_dir = '/Users/hyu/PycharmProjects/dlgo/9x9-exp2/selfplay6/sgf/full'
    sgf_file1 = f'{sgf_dir}/0-33674288586.sgf'
    values = extract_values_from_sgf(sgf_file1)
    print(values)
