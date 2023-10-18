import itertools
import resource
from typing import List, Optional

import numpy as np

import coords
import go
import mcts
import myconf
from puzzle.dataset import Puzzle9DataSet1, GameInfo
from selfplay import run_game, create_dir_if_needed
import k2net as dual_net
from katago.analysis_engine import KataDualNetwork, KataModels
from absl import logging, app, flags

from utils import grouper, format_game_summary

flags.DEFINE_integer('num_games', 5, '#games to play')
flags.DEFINE_integer('num_games_share_tree', 1, '#games that shares a tree')

FLAGS = flags.FLAGS


class InitPositions:
    """
    9x9: limit to lower-left-down-center triangle: C3, D3, E3, D4, E4, E5
            open_moves = ['C3', 'D3', 'E3', 'D4', 'E4', 'E5']
    5x5:    open_moves = ['C2', 'B2']
    """
    def __init__(self, open_moves: Optional[List], open_probs: Optional[List] = None):
        if not open_moves:
            self.init_positions = [go.Position()]
            self.open_probs = np.ones(1)
        else:
            if not open_probs:
                n = len(open_moves)
                open_probs = np.ones(n) / n
            assert len(open_moves) == len(open_probs)
            self.init_positions = [go.Position().play_move(coords.from_gtp(move)) for move in open_moves]
            self.open_probs = open_probs
        self.rng = np.random.default_rng()

    def sample(self) -> go.Position:
        return self.rng.choice(self.init_positions, p=self.open_probs)


def play_games(num_games=500):
    """ full games """
    network = dual_net.load_net(FLAGS.load_file)
    # network = DNNStub(model_file=FLAGS.load_file)
    # network = load_kata_network(KataModels.MODEL_B6_4k)

    create_dir_if_needed(selfplay_dir=FLAGS.selfplay_dir, holdout_dir=FLAGS.holdout_dir,
                         sgf_dir=FLAGS.sgf_dir)

    init_position_sampler = InitPositions(['C3', 'D3', 'E3', 'D4', 'E4', 'E5'])
    for i in range(num_games):
        init_position = init_position_sampler.sample()
        game_info = GameInfo(f'{i}', init_position, myconf.FULL_BOARD_FOCUS, full_game=True)
        player, sgf_fname = run_game(network, game_info,
                                     selfplay_dir=FLAGS.selfplay_dir,
                                     holdout_dir=FLAGS.holdout_dir,
                                     holdout_pct=FLAGS.holdout_pct,
                                     sgf_dir=FLAGS.sgf_dir,
                                     )

        moves = [coords.to_gtp(x.move) for x in player.root.position.recent]
        history_str = format_game_summary(moves, player.result_string, sgf_fname=sgf_fname, first_n=6)

        logging.info(f'game {i}: %s', history_str)

    logging.info(f'Done with {num_games} games')


def play_puzzles(num_games=500, shuffle=True):
    """ """
    network = dual_net.load_net(FLAGS.load_file)
    # network = DNNStub(model_file=FLAGS.load_file)
    # network = load_kata_network(KataModels.MODEL_B6_4k)

    create_dir_if_needed(selfplay_dir=FLAGS.selfplay_dir, holdout_dir=FLAGS.holdout_dir,
                         sgf_dir=FLAGS.sgf_dir)

    puzzle_set = Puzzle9DataSet1(Puzzle9DataSet1.EASY_COLLECTIONS)
    for i, game_info in enumerate(puzzle_set.game_iter(stop=num_games, shuffle=shuffle)):
        player, sgf_fname = run_game(network, game_info,
                                     selfplay_dir=FLAGS.selfplay_dir,
                                     holdout_dir=FLAGS.holdout_dir,
                                     holdout_pct=FLAGS.holdout_pct,
                                     sgf_dir=FLAGS.sgf_dir,
                                     )

        moves = [coords.to_gtp(x.move) for x in player.root.position.recent]
        history_str = format_game_summary(moves, player.result_string, sgf_fname=sgf_fname, first_n=6)

        # ru_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        logging.info(f'game {i} score: %s', history_str)

    logging.info(f'Done with {num_games} games')


def _examine_tree(root: mcts.MCTSNode, thresh: int):
    """ see how many nodes have enough visits """
    n = 0
    for child in root.children.values():
        if child.N >= thresh:
            n += 1
            n += _examine_tree(child, thresh)
    return n


def main(argv):
    assert len(argv) > 1
    play_puzzle = argv[1].lower().startswith('puzzle')
    logging.info(f'model={FLAGS.load_file} {FLAGS.selfplay_dir} play_puzzle={play_puzzle}')
    logging.info(f'config: #readouts={FLAGS.num_readouts} softpick_move_cutoff={FLAGS.softpick_move_cutoff}'
                 f' dir_noise={FLAGS.dirichlet_noise_weight} reduce_symmetry_before_move={FLAGS.reduce_symmetry_before_move}'
                 f' resign_thresh={FLAGS.resign_threshold}')
    if play_puzzle:
        play_puzzles(num_games=FLAGS.num_games)
    else:
        play_games(num_games=FLAGS.num_games)


def main_local(argv):
    FLAGS.load_file = '/Users/hyu/PycharmProjects/a0-jax/exp-go9/tfmodel/model-218'
    FLAGS.load_file = f'{myconf.MODELS_DIR}/model10_4.mlpackage'

    FLAGS.sgf_dir = f'{myconf.SELFPLAY_DIR}/sgf'
    FLAGS.selfplay_dir = f'{myconf.SELFPLAY_DIR}/train'
    FLAGS.holdout_dir = f'{myconf.SELFPLAY_DIR}/val'
    FLAGS.num_readouts = 200
    FLAGS.parallel_readouts = 16
    FLAGS.holdout_pct = 0
    FLAGS.softpick_move_cutoff = 6
    FLAGS.dirichlet_noise_weight = 0.25
    FLAGS.resign_threshold = -1.0
    FLAGS.reduce_symmetry_before_move = 0
    FLAGS.verbose = 0
    play_games(num_games=2)
    play_puzzles(5, shuffle=False)


if __name__ == '__main__':
    # app.run(main_local)
    app.run(main)
