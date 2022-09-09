import resource
from typing import List, Optional

import numpy as np

import coords
import go
import mcts
import myconf
from selfplay import run_game, create_dir_if_needed
import k2net as dual_net
from katago.analysis_engine import KataDualNetwork, KataModels
from absl import logging, app, flags

from utils import grouper

flags.DEFINE_integer('num_games', 5, '#games to play')
flags.DEFINE_integer('num_games_share_tree', 1, '#games that shares a tree')

FLAGS = flags.FLAGS


class InitPositions:
    """
    9x9: limit to lower-left-down-center triangle: C3, D3, E3, D4, E4, E5
    5x5:    open_moves, open_probs = ['C2', 'B2'], np.ones(2) / 2
    """
    def __init__(self, open_moves: Optional[List], open_probs: Optional[List]):
        if not open_moves:
            open_moves, open_probs = ['C3', 'D3', 'E3', 'D4', 'E4', 'E5'], np.ones(6) / 6
        assert len(open_moves) == len(open_probs)

        self.init_positions = [go.Position().play_move(coords.from_gtp(move)) for move in open_moves]
        self.open_probs = open_probs

    def sample(self) -> go.Position:
        return np.random.choice(self.init_positions, p=self.open_probs)


def load_kata_network(model_file):
    logging.info('loading Kata %s', model_file)
    return KataDualNetwork(model_file)


def load_k2net(model_file):
    if model_file:
        logging.info('loading %s', model_file)
        network = dual_net.DualNetwork(model_file)
    else:
        logging.info('use DummyNetwork')
        network = dual_net.DummyNetwork()
    return network


def play_games(num_games=500):
    """ """
    network = load_k2net(FLAGS.load_file)
    # network = load_kata_network(KataModels.MODEL_B6_4k)

    create_dir_if_needed(selfplay_dir=FLAGS.selfplay_dir, holdout_dir=FLAGS.holdout_dir,
                         sgf_dir=FLAGS.sgf_dir)

    init_position_sampler = InitPositions(None, None)
    for i_batch in grouper(FLAGS.num_games_share_tree, iter(range(num_games))):
        if FLAGS.num_games_share_tree > 1:
            logging.info(f'\nStarting new batch : %d games', len(i_batch))
        init_position = init_position_sampler.sample()
        shared_tree = mcts.MCTSNode(init_position)
        for i in i_batch:
            player = run_game(network,
                              init_position=init_position,
                              init_root=shared_tree,
                              selfplay_dir=FLAGS.selfplay_dir,
                              holdout_dir=FLAGS.holdout_dir,
                              holdout_pct=FLAGS.holdout_pct,
                              sgf_dir=FLAGS.sgf_dir,
                              game_id=str(i)
                              )
            margin_est = player.black_margin_no_komi
            result_str = player.result_string
            moves_history = player.root.position.recent
            history_str = ' '.join([coords.to_gtp(x.move) for x in moves_history[:8]])
            # history_str = ' '.join(player.move_infos[:8])

            # ru_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

            logging.info(f'game {i}: %d moves, score %s\t%s', player.root.position.n,
                         result_str, history_str)

        del shared_tree


def _examine_tree(root: mcts.MCTSNode, thresh: int):
    """ see how many nodes have enough visits """
    n = 0
    for child in root.children.values():
        if child.N >= thresh:
            n += 1
            n += _examine_tree(child, thresh)
    return n


def main(argv):
    play_games(num_games=FLAGS.num_games)


def main_local(argv):
    FLAGS.load_file = f'{myconf.MODELS_DIR}/model1_epoch5.h5'
    FLAGS.sgf_dir = f'{myconf.SELFPLAY_DIR}/sgf'
    FLAGS.selfplay_dir = f'{myconf.SELFPLAY_DIR}/train'
    FLAGS.holdout_dir = f'{myconf.SELFPLAY_DIR}/val'
    FLAGS.num_readouts = 200
    FLAGS.parallel_readouts = 16
    FLAGS.holdout_pct = 0
    FLAGS.softpick_move_cutoff = 6
    FLAGS.dirichlet_noise_weight = 0.125
    FLAGS.resign_threshold = -1.0
    play_games(num_games=2)


if __name__ == '__main__':
    app.run(main)
