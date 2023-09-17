"""
puzzle dataset differs from other datasets: we need focus_area

data augmentation? Symmetries are handled in enhance_ds. We could shift (if not corner), swap color
"""
import glob
import itertools
import os
import random
from collections import Counter
from typing import Iterable

import attr
import numpy as np

import coords
import go
import myconf
from puzzle.lnd_puzzle import LnDPuzzle
from sgf_wrapper import SGFReader


@attr.s
class GameInfo:
    game_id: str = attr.ib()
    init_position: go.Position = attr.ib(default=None)
    focus_area: np.array = attr.ib(default=None)
    init_root = attr.ib(default=None)
    max_moves: int = attr.ib(default=myconf.BOARD_SIZE_SQUARED*2)  # increase for full game
    sgf_reader: SGFReader = attr.ib(default=None)


class Puzzle9DataSet1:
    """ first 9x9 puzzle set
    """
    PUZZLES_DIR = '/Users/hyu/PycharmProjects/dlgo/puzzles9x9'
    EASY_COLLECTIONS = [f'{PUZZLES_DIR}/Beginning Shapes/*',
                        f'{PUZZLES_DIR}/easy 2-choice question*/*'
                        ]

    def __init__(self):
        self.collection_patterns = self.EASY_COLLECTIONS

        self._sgf_list = []
        for glob_pattern in self.collection_patterns:
            sgfs = glob.glob(glob_pattern)
            sgfs = [x for x in sgfs
                    # if not x.endswith('Problem 14.sgf')   # two puzzles on the board  todo include
                    # not x.endswith('2023.06.12.sgf') and  # attack side 2 lines from edge
                    # not x.endswith('2023.07.13.sgf')  # capture in the middle, not corner
                    ]
            # todo version sort?
            self._sgf_list.extend(sorted(sgfs))

    def __len__(self):
        # a bit over-estimate: a few puzzles will be filtered out if boundary solver fails
        return len(self._sgf_list)

    def game_generator(self, shuffle: bool = False) -> Iterable[GameInfo]:
        """ underlying generator: deterministic, fixed size """
        sgf_list = self._sgf_list
        if shuffle:
            sgf_list = sgf_list[:]
            random.shuffle(sgf_list)
        for sgf_fname in sgf_list:
            basename = os.path.basename(sgf_fname)
            reader = SGFReader.from_file_compatible(sgf_fname)
            if reader.board_size() != go.N:
                print(f'skipping {basename}, not 9x9')
                continue

            pos = reader.first_pos()
            try:
                contested_area, attack_side = LnDPuzzle.solve_contested_area(pos.board)
            except AssertionError:
                print(f'Solving puzzle boundary failed {basename}, skipping')
                continue
            max_moves = int(np.sum(contested_area) * 2)
            # num_mainline_moves = reader.last_pos().n - pos.n
            # print(f'{basename} mainline: {num_mainline_moves} contested: {max_moves / 2}')
            max_moves = min(myconf.BOARD_SIZE_SQUARED, max_moves)
            yield GameInfo(basename, pos, contested_area,
                           init_root=None, max_moves=max_moves, sgf_reader=reader)

    def game_iter(self, start=0, stop=None, shuffle=False) -> Iterable[GameInfo]:
        """ produce the desired number of games by it.cycle() """
        gen = self.game_generator(shuffle)
        return itertools.islice(itertools.cycle(gen), start, stop)


def test_dataset():
    ds = Puzzle9DataSet1()
    ds_size = len(ds)
    assert ds_size == 126
    for ginfo in itertools.islice(ds.game_generator(), ds_size):
        print(ginfo.game_id, ginfo.max_moves)


def guess_winner(comment: str) -> int:
    winner = 0
    if 'black to kill' in comment:
        winner = 1
    elif 'black to live' in comment:
        winner = 1
    elif 'white to kill' in comment:
        winner = -1
    elif 'white to live' in comment:
        winner = -1
    return winner


def test_solve_info():
    """ extract human annotated results, as well as first moves

    All 6 unknown winner are from "Beginner shape".
    [(1, 106), (0, 6), (-1, 6)]
    """
    ds = Puzzle9DataSet1()
    ds_size = len(ds)
    counter = Counter()
    for ginfo in itertools.islice(ds.game_generator(), ds_size):
        reader = ginfo.sgf_reader
        comments = reader.root_comments()
        comment = comments[0].lower()
        winner = guess_winner(comment)
        print(ginfo.game_id, winner, comments)
        counter[winner] += 1

        moves = [pwc.next_move for pwc in itertools.islice(reader.iter_pwcs(), 4)]
        gtp_moves = ' '.join([coords.to_gtp(x) for x in moves])
        print(gtp_moves)
        print()

    print(counter.most_common())
