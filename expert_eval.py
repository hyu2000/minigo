""" expert (kata) eval of models, including eval games, selfplay games
"""
import os
from typing import List
import attr
import numpy as np
import pandas as pd

import coords
import go
import myconf
from go import PlayerMove
from katago.analysis_engine import KataEngine, ARequest, RootInfo
from sgf_wrapper import SGFReader


@attr.define
class ChangePoint:
    i: int
    delta: float

    def __repr__(self):
        return f'({self.i}: {self.delta:.2f})'


class ExpertReviewer:
    CROSSOVER_MARGIN_THRESH = 0.03
    JIGO = 0.5

    def __init__(self):
        self.kata = KataEngine().start()

    @staticmethod
    def detect_crossover(black_winrates: List[float]):
        """ do we want a hard cutoff at 0.5? some wiggle room?
        a crossover with a diff of at least 0.02?

        Note this includes positive surprises as well, also we may miss a blunder that didn't cross 0.5, e.g.
        27:0.29, 28:0.26, 29:0.23 -> 30:0.48 -> 31:0.53  the white blunder (0.25) is missed, then a small black
        surprise is declared Change(i=30, 0.05).
        """
        wr_arr = np.array(black_winrates)
        wr_signs = np.sign(wr_arr - ExpertReviewer.JIGO)
        wr_diff = np.diff(wr_arr)

        cps = np.where((np.diff(wr_signs) != 0) & (np.abs(wr_diff) > ExpertReviewer.CROSSOVER_MARGIN_THRESH))
        # np.where returns a tuple of arrays: (array([0, 1, 4, 5]),)
        assert len(cps) == 1
        return [ChangePoint(i, wr_diff[i]) for i in cps[0]]

    def review_a_game(self, sgf_fname):
        """ whenever winrate crosses 0.5, one side has made a mistake:
        - goes up: white made a mistake
        - down: black made a mistake
        """
        reader = SGFReader.from_file_compatible(sgf_fname)

        moves = []
        for pwc in reader.iter_pwcs():
            move = [go.color_str(pwc.position.to_play)[0], coords.to_gtp(pwc.next_move)]
            moves.append(move)

        turns_to_analyze = list(range(len(moves)))
        arequest = ARequest(moves, turns_to_analyze, 500, komi=reader.komi())
        responses = self.kata.analyze(arequest)

        black_winrates = [RootInfo.from_dict(x.rootInfo).winrate for x in responses]
        print(', '.join([f'{i}:{x:.2f}' for i, x in enumerate(black_winrates)]))
        cps = self.detect_crossover(black_winrates)
        return cps


def game_blunder_review(sgfs_dir):
    """ kata review #blunders stats by both sides, ordered by pos.n
    """
    MAX_MOVE_NUMBER = 80 - 1
    reviewer = ExpertReviewer()

    num_games = 0
    mistakes_by_move = np.zeros((2, MAX_MOVE_NUMBER + 1), dtype=int)
    for sgf_fname in os.listdir(f'{sgfs_dir}'):
        if not sgf_fname.endswith('.sgf'):
            continue

        num_games += 1
        cps = reviewer.review_a_game(sgf_fname)
        # todo this is testing positive / negative surprise rather than black/white
        black_mistakes = [x.i for x in cps if x.delta < 0]
        mistakes_by_move[0, np.minimum(black_mistakes, MAX_MOVE_NUMBER)] += 1
        white_mistakes = [x.i for x in cps if x.delta > 0]
        mistakes_by_move[1, np.minimum(white_mistakes, MAX_MOVE_NUMBER)] += 1

    df = pd.DataFrame(mistakes_by_move, index=['black', 'white'])
    print(f'Total {num_games} games')
    print(df)


def test_crossover_detect0():
    winrates = [0.53, 0.44, 0.5, 0.51, 0.59, 0.49, 0.67, 0.6, 0.67]
    crosses = ExpertReviewer.detect_crossover(winrates)
    print(crosses)
    assert len(crosses) == 4


def test_crossover_detect1():
    """  all the jumps make sense, but many are due to same underlying reason...

cross [(0: -0.04), (3: 0.04), (5: 0.09), (8: -0.03), (9: 0.17), (14: -0.40), (15: 0.51), (16: -0.36), (19: 0.47),
       (22: -0.58), (23: 0.66), (24: -0.57),              (29: 0.35),             (40: -0.62), (43: 0.67),
                   (48: -0.55), (49: 0.55), (52: -0.75), (53: 0.76), (54: -0.70), (55: 0.69), (56: -0.79), (57: 0.80)]
jumps [                                              (9: 0.17), (14: -0.40), (15: 0.51), (16: -0.36), (19: 0.47),
       (22: -0.58), (23: 0.66), (24: -0.57),?(26: -0.14), (29: 0.35),*(31: 0.40), (40: -0.62), (43: 0.67),*(46: -0.46),  hereafter all due to the same missed move
      *(47: 0.47), (48: -0.55), (49: 0.55), (52: -0.75), (53: 0.76), (54: -0.70), (55: 0.69), (56: -0.79), (57: 0.80),
       (58: -0.40), (59: 0.40), (60: -0.20), (61: 0.19), (62: -0.13), (63: 0.12)]
    """
    winrates = [0.54, 0.50, 0.49, 0.48, 0.52, 0.50, 0.59, 0.59, 0.53, 0.50,
                0.67, 0.68, 0.67, 0.66, 0.70, 0.30, 0.81, 0.45, 0.42, 0.40,
                0.87, 0.82, 0.84, 0.26, 0.92, 0.35, 0.40, 0.26, 0.24, 0.19,
                0.54, 0.56, 0.96, 0.96, 0.97, 0.96, 0.99, 0.95, 0.98, 0.94,
                0.96, 0.34, 0.33, 0.32, 0.99, 0.99, 0.99, 0.53, 1.00, 0.45,
                1.00, 0.97, 0.99, 0.24, 1.00, 0.30, 0.99, 0.20, 1.00, 0.60,
                1.00, 0.80, 0.99, 0.86, 0.98, 0.91, 0.96, 0.93, 0.97, 0.92,
                0.98, 0.92, 1.00, 0.93, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
                1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
                1.00]
    cps = ExpertReviewer.detect_crossover(winrates)
    print()
    print('cross', cps)

    # does it capture all jumps? No
    warr = np.array(winrates)
    diff = np.diff(warr)
    JUMP_THRESH = 0.1
    jumps = np.where(np.abs(diff) > JUMP_THRESH)
    cp_jumps = [ChangePoint(i, diff[i]) for i in jumps[0]]
    print('jumps', cp_jumps)


def test_crossover_detect2():
    """ TODO JIGO w/ wiggle room """
    winrates = [0.86, 0.80, 0.85, 0.31, 0.91, 0.40, 0.39, 0.29, 0.26, 0.23,
                0.48, 0.53, 0.94]
    cps = ExpertReviewer.detect_crossover(winrates)
    print(cps)
    black_flips = [x for x in cps if x.i % 2 == 0]
    white_flips = [x for x in cps if x.i % 2 == 1]
    assert(all(x.delta < 0 for x in black_flips))
    assert(all(x.delta > 0 for x in white_flips))


def test_review_a_game():
    sgf_fname = f'{myconf.EXP_HOME}/eval_bots-model2/sgfs-model1-vs-model2/model1_epoch5#100-vs-model2_epoch2#200-15-65075018704.sgf'

    reviewer = ExpertReviewer()
    cps = reviewer.review_a_game(sgf_fname)

    print(cps)
    black_flips = [x for x in cps if x.i % 2 == 0]
    white_flips = [x for x in cps if x.i % 2 == 1]
    print('black', black_flips)
    print('white', white_flips)
    assert(all(x.delta < 0 for x in black_flips))
    assert(all(x.delta > 0 for x in white_flips))


def test_review_games():
    sgfs_dir = ''
    game_blunder_review(sgfs_dir)
