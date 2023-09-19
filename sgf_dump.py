""" simple utils
"""
import glob
import os.path

import coords
import myconf
from sgf_wrapper import SGFReader


def dump_sgf_moves(sgf_pattern: str):
    """ dump a bunch of *.sgf in gtp moves, for quick inspection """
    for sgf_fname in glob.glob(sgf_pattern):
        basename = os.path.basename(sgf_fname).removesuffix('.sgf')
        reader = SGFReader.from_file_compatible(sgf_fname)
        moves = [coords.to_gtp(x.next_move) for x in reader.iter_pwcs()]
        moves_str = ' '.join(moves)
        result_str = reader.result_str()
        print(f'{basename}: {moves_str}\t{result_str}')


def test_dump_sgf_moves():
    fpattern = f'{myconf.EXP_HOME}/selfplay0/sgf/full/Problem 1-*'
    print()
    dump_sgf_moves(fpattern)
