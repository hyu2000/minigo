"""
puzzle dataset differs from other datasets: we need focus_area

data augmentation? Symmetries are handled in enhance_ds. We could shift (if not corner), swap color
"""
import glob
import itertools
import os
import random
from collections import Counter
from typing import Iterable, List, Optional

import attr
import numpy as np
from natsort import natsorted

import coords
import go
import myconf
from puzzle.lnd_puzzle import LnDPuzzle
from sgf_wrapper import SGFReader, VariationTraverser


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


PUZZLES_DIR = '/Users/hyu/PycharmProjects/dlgo/puzzles9x9'


class Puzzle9DataSet1:
    """ first 9x9 puzzle set
    """
    EASY_COLLECTIONS = [f'{PUZZLES_DIR}/Beginning Shapes/*',
                        f'{PUZZLES_DIR}/easy 2-choice question*/*'
                        ]
    # guess_winner() doesn't work; mainline is mostly wrong -- the correct path is mostly the last one
    EASY_COLLECTIONS2 = [f'{PUZZLES_DIR}/How to Play Go +/Life and Death*.sgf']

    def __init__(self, collection_patterns=None):
        self.collection_patterns = collection_patterns or self.EASY_COLLECTIONS

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
        """ convenience method: produce the desired number of games by it.cycle() """
        gen = self.game_generator(shuffle)
        return itertools.islice(itertools.cycle(gen), start, stop)


def find_solution_moves(reader: SGFReader) -> Optional[List[str]]:
    """
    Returns gtp moves for a correct solution, None otherwise
    """
    finder = VariationTraverser.CorrectPathFinder()
    traverser = VariationTraverser(finder.path_handler)
    traverser.traverse(reader.raw_sgf)
    solution = finder.correct_path
    if solution:
        return [coords.to_gtp(x) for x in solution]
    return None


def find_mainline_moves(reader: SGFReader) -> Optional[List[str]]:
    """ It turns out that mainline might lead to the wrong solution.
    Returns gtp moves for correct mainline, None otherwise
    """
    tuples = list(reader.iter_comments())
    gtp_moves    = [x[0] for x in tuples]
    all_comments = [x[1] for x in tuples]
    assert len(all_comments[-1]) == 1
    final_comment = all_comments[-1][0].lower()
    correct_in_mainline = 'correct' in final_comment
    wrong_in_mainline = 'wrong' in final_comment
    assert correct_in_mainline ^ wrong_in_mainline
    if correct_in_mainline:
        return gtp_moves
    else:
        return None


def test_dataset():
    ds = Puzzle9DataSet1()
    ds_size = len(ds)
    assert ds_size == 126
    for ginfo in itertools.islice(ds.game_generator(), ds_size):
        print(ginfo.game_id, ginfo.max_moves)

    ds = Puzzle9DataSet1(Puzzle9DataSet1.EASY_COLLECTIONS2)
    assert len(ds) == 20


def test_solve_info():
    """ extract human annotated results, as well as first moves

    All 6 unknown winner are from "Beginner shape".
    [(1, 106), (0, 6), (-1, 6)]
    """
    ds = Puzzle9DataSet1(Puzzle9DataSet1.EASY_COLLECTIONS2)
    ds_size = len(ds)
    counter = Counter()
    for ginfo in itertools.islice(ds.game_generator(), ds_size):
        game_id = ginfo.game_id.removesuffix('.sgf')
        reader = ginfo.sgf_reader
        root_comment = reader.root_comments()[0].lower()
        winner = guess_winner(root_comment)
        counter[winner] += 1

        solution_moves = find_solution_moves(reader)
        if solution_moves is None:
            print(f'{game_id} mainline wrong, skipping')
            continue
        print('%-16s %-6s %s' % (game_id, 'black' if winner > 0 else 'white' if winner < 0 else '-',
                                 ' '.join(solution_moves[:2])))

    print(counter.most_common())


def score_selfplay_records(ds: Puzzle9DataSet1, sgf_dir, skip_no_winner=True):
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
    total_puzzle_used = 0  # puzzle with complete info
    total_puzzle_solved = 0  # completely solved
    num_puzzle_no_winner = 0  # no winner annotation, or can't parse it
    num_puzzle_no_solution = 0  # no solution marked as correct
    total_sgfs = 0
    for ginfo in ds.game_generator():  # itertools.islice(ds.game_generator(), 4):
        winner_annotated = ginfo.guess_winner_from_comment()
        if winner_annotated == 0:
            if skip_no_winner:
                num_puzzle_no_winner += 1
                continue
        solution_moves = find_solution_moves(ginfo.sgf_reader)
        if solution_moves is None:  # skip if no correct solution
            num_puzzle_no_solution += 1
            continue
        total_puzzle_used += 1
        first_move_solution = coords.from_gtp(solution_moves[0])

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
    print(f'Summary for {sgf_dir}:\n  {total_sgfs} sgfs, completely solved puzzle: '
          f'{total_puzzle_solved} / {total_puzzle_used}, '
          f'skipping {num_puzzle_no_winner} (no winner), {num_puzzle_no_solution} (no solution)')


def test_score_selfplay():
    ds = Puzzle9DataSet1()
    score_selfplay_records(ds, f'{myconf.EXP_HOME}/selfplay4/sgf/full')

    # ds = Puzzle9DataSet1(collection_patterns=Puzzle9DataSet1.EASY_COLLECTIONS2)
    # score_selfplay_records(ds, f'{myconf.EXP_HOME}/selfplay4-on-easy2/sgf/full', skip_no_winner=False)



def test_correct_path_stats():
    """ count num correct solutions in a puzzle """
    glob_pattern = f'{PUZZLES_DIR}/Beginning Shapes/*.sgf'
    glob_pattern = f'{PUZZLES_DIR}/easy 2-choice question*/*.sgf'
    glob_pattern = f'{PUZZLES_DIR}/How to Play Go +/Life and Death*.sgf'

    for sgf_fname in natsorted(glob.glob(glob_pattern)):
        basename = os.path.basename(sgf_fname)

        cnter = VariationTraverser.PathCounter()
        traverser = VariationTraverser(cnter.path_handler)
        traverser.traverse_sgf(sgf_fname)

        reader = SGFReader.from_file_compatible(sgf_fname)
        solution = find_solution_moves(reader)
        print(f'{basename} \tcorrect/total = {cnter.num_correct_paths} / {cnter.num_paths}\t{solution}')
