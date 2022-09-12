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
        return f'Change(i={self.i}, {self.delta:.2f})'


class ExpertReviewer:
    CROSSOVER_MARGIN_THRESH = 0.03
    JIGO = 0.5

    def __init__(self):
        self.kata = KataEngine().start()

    @staticmethod
    def detect_crossover(black_winrates: List[float]):
        """ do we want a hard cutoff at 0.5? some wiggle room?
        a crossover with a diff of at least 0.2?
        """

        # cps = []  # type: List[ChangePoint]
        # for i in range(len(black_winrates) - 1):
        #     wr1, wr2 = black_winrates[i], black_winrates[i+1]
        #     delta = wr2 - wr1
        #     did_cross = np.sign(wr1 - ExpertReviewer.JIGA) != np.sign(wr2 - ExpertReviewer.JIGA)
        #     if did_cross and abs(delta) > ExpertReviewer.CROSSOVER_MARGIN_THRESH:
        #         cps.append(ChangePoint(i, delta))

        wr_arr = np.array(black_winrates)
        wr_signs = np.sign(wr_arr - ExpertReviewer.JIGO)
        wr_diff = np.diff(wr_arr)
        # note this includes positive surprises as well
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
        arequest = ARequest(moves, turns_to_analyze, 200, komi=reader.komi())
        responses = self.kata.analyze(arequest)

        black_winrates = [RootInfo.from_dict(x.rootInfo).winrate for x in responses]
        # print(', '.join([f'{x:.2f}' for x in black_winrates]))
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


def test_crossover_detect():
    winrates = [0.54, 0.51, 0.52, 0.51, 0.59, 0.48, 0.65, 0.64, 0.57, 0.53,
                0.71, 0.69, 0.66, 0.69, 0.68, 0.32, 0.79, 0.48, 0.48, 0.46,
                0.86, 0.81, 0.85]
    crosses = ExpertReviewer.detect_crossover(winrates)
    print(crosses)


def test_review_a_game():
    sgf_fname = f'{myconf.EXP_HOME}/eval_bots-model2/sgfs-model1-vs-model2/model1_epoch5#100-vs-model2_epoch2#200-15-65075018704.sgf'

    reviewer = ExpertReviewer()
    cps = reviewer.review_a_game(sgf_fname)

    print(cps)
    black_flips = [x.delta for x in cps if x.i % 2 == 0]
    white_flips = [x.delta for x in cps if x.i % 2 == 1]
    print('black', black_flips)
    print('white', white_flips)


def test_review_games():
    sgfs_dir = ''
    game_blunder_review(sgfs_dir)