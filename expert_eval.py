""" expert (kata) eval of models, including eval games, selfplay games
"""
import logging
import os
import glob
from typing import List
import attr
import numpy as np
import pandas as pd

import coords
import go
import myconf
import sgf_wrapper
import utils
from go import PlayerMove
from katago.analysis_engine import KataEngine, ARequest, RootInfo, assemble_comments
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
    JUMP_THRESH = 0.10

    def __init__(self, annotated_sgf_dir='/tmp/kata_reviewed_sgfs'):
        self.kata = KataEngine().start()
        self._annotated_sgf_dir = annotated_sgf_dir
        utils.ensure_dir_exists(annotated_sgf_dir)

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

    @staticmethod
    def find_jumps(black_winrates: List[float]):
        """ """
        warr = np.array(black_winrates)
        diff = np.diff(warr)
        jumps = np.where(np.abs(diff) > ExpertReviewer.JUMP_THRESH)
        return [ChangePoint(i, diff[i]) for i in jumps[0]]

    def review_a_game(self, sgf_fname):
        """ whenever winrate crosses 0.5, one side has made a mistake:
        - goes up: white made a mistake
        - down: black made a mistake
        """
        reader = SGFReader.from_file_compatible(sgf_fname)

        player_moves = [PlayerMove(pwc.position.to_play, pwc.next_move)
                        for pwc in reader.iter_pwcs()]
        turns_to_analyze = list(range(len(player_moves)))
        arequest = ARequest(ARequest.format_moves(player_moves), turns_to_analyze, 500, komi=reader.komi())
        responses = self.kata.analyze(arequest)

        # write out annotated game, for later review
        comments = assemble_comments(arequest, responses)
        sgf_str = sgf_wrapper.make_sgf(player_moves, reader.result_str(), komi=arequest.komi,
                                       white_name=reader.white_name(),
                                       black_name=reader.black_name(),
                                       game_comment=f'analyzed by: {self.kata.model_id()}',
                                       comments=comments)
        out_sgfname = f'{self._annotated_sgf_dir}/annotate.%s' % os.path.basename(sgf_fname)
        logging.info(f'Writing review to {out_sgfname}')
        with open(out_sgfname, 'w') as f:
            f.write(sgf_str)

        black_winrates = [RootInfo.from_dict(x.rootInfo).winrate for x in responses]
        # print(', '.join([f'{i}:{x:.2f}' for i, x in enumerate(black_winrates)]))
        # cps = self.detect_crossover(black_winrates)
        cps = self.find_jumps(black_winrates)
        surprises = find_surprise_changes(cps)
        if len(surprises) > 0:
            logging.info(f'%s has %d/%d surprises: %s', reader.name, len(surprises), len(cps), surprises)
        return cps, reader


def find_surprise_changes(cps: List[ChangePoint]) -> List[ChangePoint]:
    """ find all positive surprises in change-points, i.e. black moves that improves black winrate """
    surprises = [x for x in cps if (x.i % 2 == 0 and x.delta > 0) or (x.i % 2 == 1 and x.delta < 0)]
    return surprises


def game_blunder_review(sgfs_dir, model_ids: List[str]):
    """ kata review #blunders stats by both sides, ordered by pos.n

    model_ids: if specified, check sgf_fname contains all models in the list;
        and tabulate stats for each model (both playing white and black)
    """
    MAX_MOVE_NUMBER = 80 - 1
    reviewer = ExpertReviewer()

    num_games = 0
    stats_by_model = {m: np.zeros(MAX_MOVE_NUMBER + 1, dtype=int) for m in model_ids}
    model_id_set = set(model_ids)
    for sgf_fname in glob.glob(f'{sgfs_dir}/*.sgf'):
        if any(x not in sgf_fname for x in model_ids):
            continue

        num_games += 1
        logging.info(f'reviewing #{num_games}')
        cps, reader = reviewer.review_a_game(sgf_fname)

        black_id = reader.black_name()
        white_id = reader.white_name()
        if black_id in model_id_set:
            black_mistakes = [x.i for x in cps if x.i % 2 == 0]
            if black_mistakes:
                stats_by_model[black_id][np.minimum(black_mistakes, MAX_MOVE_NUMBER)] += 1
        if white_id in model_id_set:
            white_mistakes = [x.i for x in cps if x.i % 2 == 1]
            if white_mistakes:
                stats_by_model[white_id][np.minimum(white_mistakes, MAX_MOVE_NUMBER)] += 1

        # if num_games > 1:
        #     break

    df = pd.DataFrame(stats_by_model)
    pickle_fpath = '/tmp/df-blunder-stats.pkl'
    df.to_pickle(pickle_fpath)
    print(f'Total {num_games} games')
    print(df)


def test_crossover_detect0():
    logging.info('simple test')
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
    # from f'{myconf.EXP_HOME}/eval_bots-model2/sgfs-model1-vs-model2/model1_epoch5#100-vs-model2_epoch2#200-15-65075018704.sgf'
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
    JUMP_THRESH = ExpertReviewer.JUMP_THRESH
    jumps = np.where(np.abs(diff) > JUMP_THRESH)
    cp_jumps = [ChangePoint(i, diff[i]) for i in jumps[0]]
    print('jumps', cp_jumps)
    cp_surprises = find_surprise_changes(cps)
    assert len(cp_surprises) == 0


def test_crossover_detect2():
    """ TODO JIGO w/ wiggle room """
    winrates = [0.86, 0.80, 0.85, 0.31, 0.91, 0.40, 0.39, 0.29, 0.26, 0.23,
                0.48, 0.53, 0.94]
    cps = ExpertReviewer.detect_crossover(winrates)
    print(cps)
    cp_surprises = find_surprise_changes(cps)
    assert len(cp_surprises) == 0


def test_review_a_game():
    # sgf_fname = f'{myconf.EXP_HOME}/eval_bots-model2/sgfs-model1-vs-model2/model1_epoch5#100-vs-model2_epoch2#200-15-65075018704.sgf'
    sgf_fname = f'{myconf.EXP_HOME}/eval_bots-model2/sgfs-model1-vs-model2/model1_epoch5#200-vs-model2_epoch2#300-8-63909912641.sgf'

    reviewer = ExpertReviewer()
    cps, reader = reviewer.review_a_game(sgf_fname)

    print(cps)
    black_flips = [x for x in cps if x.i % 2 == 0]
    white_flips = [x for x in cps if x.i % 2 == 1]
    print('black', len(black_flips), black_flips)
    print('white', len(white_flips), white_flips)


def test_review_games():
    sgfs_dir = f'{myconf.EXP_HOME}/eval_bots-model2/sgfs-model1-vs-model2'
    game_blunder_review(sgfs_dir, ['model1_epoch5#200', 'model2_epoch2#300'])
