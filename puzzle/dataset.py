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
from natsort import natsorted

import coords
import go
import myconf
from puzzle.lnd_puzzle import LnDPuzzle
from sgf_wrapper import SGFReader


def guess_winner(comment: str) -> int:
    """ infer puzzle outcome from root node comment """
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


@attr.s
class GameInfo:
    game_id: str = attr.ib()
    init_position: go.Position = attr.ib(default=None)
    focus_area: np.array = attr.ib(default=None)
    init_root = attr.ib(default=None)
    max_moves: int = attr.ib(default=myconf.BOARD_SIZE_SQUARED*2)  # increase for full game
    sgf_reader: SGFReader = attr.ib(default=None)

    def guess_winner_from_comment(self) -> int:
        comments = self.sgf_reader.root_comments()
        comment = comments[0].lower()
        return guess_winner(comment)



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
            # natsort will put "problem 2" ahead of "problem 11"
            self._sgf_list.extend(natsorted(sgfs))

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
            basename = os.path.basename(sgf_fname).removesuffix('.sgf')
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
        root_comment = reader.root_comments()[0].lower()
        winner = guess_winner(root_comment)
        counter[winner] += 1

        moves = [pwc.next_move for pwc in itertools.islice(reader.iter_pwcs(), 2)]
        gtp_moves = ' '.join([coords.to_gtp(x) for x in moves])

        # it turns out mainline might lead to the wrong solution
        all_comments = [comments for (gtp_move, comments) in reader.iter_comments()]
        assert len(all_comments[-1]) == 1
        final_comment = all_comments[-1][0].lower()
        correct_in_mainline = 'correct' in final_comment
        wrong_in_mainline = 'wrong' in final_comment
        assert correct_in_mainline ^ wrong_in_mainline

        game_id = ginfo.game_id.removesuffix('.sgf')
        print('%-16s %-6s %s %s' % (game_id, 'black' if winner > 0 else 'white' if winner < 0 else '-',
                                    gtp_moves, 'correct' if correct_in_mainline else 'wrong'))

    print(counter.most_common())


def score_selfplay_records(ds: Puzzle9DataSet1, sgf_dir):
    """ evaluate selfplay records to help track progress
    1. outcome agreement with puzzle comment
    2. first move agreement
    3. % moves are inside focus area

Example:
Problem 1: result-match= 7/8, first-move-match= 8/8, occured=8/8
Problem 2: result-match= 8/8, first-move-match= 8/8, occured=8/8 *solved
    """
    SEARCH_KEY_MOVE_IN_FIRST_N = 60

    print(f'Scoring {sgf_dir}, key-move-search in {SEARCH_KEY_MOVE_IN_FIRST_N} ...')
    total_puzzle_solved = 0  # completely solved
    total_sgfs = 0
    for ginfo in ds.game_generator():  # itertools.islice(ds.game_generator(), 4):
        winner_annotated = ginfo.guess_winner_from_comment()
        if winner_annotated == 0:
            continue
        first_move_solution = next(ginfo.sgf_reader.iter_pwcs()).next_move

        game_id = ginfo.game_id
        sgfs = glob.glob(f'{sgf_dir}/{game_id}-*.sgf')
        num_result_agree, num_first_move_agree = 0, 0
        count_key_move_occured = 0
        for sgf in sgfs:
            reader = SGFReader.from_file_compatible(sgf)
            first_moves = [x.next_move for x in itertools.islice(reader.iter_pwcs(), SEARCH_KEY_MOVE_IN_FIRST_N)]
            num_result_agree += reader.result() == winner_annotated
            num_first_move_agree += first_moves[0] == first_move_solution
            count_key_move_occured += first_move_solution in set(first_moves)

        num_sgfs = len(sgfs)

        total_sgfs += num_sgfs
        solved = num_result_agree == num_sgfs == num_first_move_agree and num_sgfs > 0
        total_puzzle_solved += solved
        print(f'{game_id}: result-match= {num_result_agree}/{num_sgfs}, first-move-match= {num_first_move_agree}/{num_sgfs}, '
              f'occured={count_key_move_occured}/{num_sgfs} %s' % ('*solved' if solved else ''))
    print(f'Summary for {sgf_dir}:\n  {total_sgfs} sgfs, completely solved puzzle: {total_puzzle_solved}')


def test_score_selfplay():
    ds = Puzzle9DataSet1()
    score_selfplay_records(ds, f'{myconf.EXP_HOME}/selfplay3/sgf/full')
