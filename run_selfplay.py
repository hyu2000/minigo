import resource
import numpy as np

import coords
import go
import mcts
import myconf
from selfplay import run_game, create_dir_if_needed
import k2net as dual_net
from absl import logging, app, flags

from utils import grouper

flags.DEFINE_integer('num_games', 5, '#games to play')
flags.DEFINE_integer('num_games_share_tree', 1, '#games that shares a tree')

FLAGS = flags.FLAGS


def play_games(num_games=500):
    """ """
    model_file = FLAGS.load_file
    if model_file:
        logging.info('loading %s', model_file)
        network = dual_net.DualNetwork(model_file)
    else:
        logging.info('use DummyNetwork')
        network = dual_net.DummyNetwork()

    create_dir_if_needed(selfplay_dir=FLAGS.selfplay_dir, holdout_dir=FLAGS.holdout_dir,
                         sgf_dir=FLAGS.sgf_dir)

    # limit to lower-left-down-center triangle: C3, D3, E3, D4, E4, E5
    open_moves, open_probs = ['C3', 'D3', 'E3', 'D4', 'E4', 'E5'], np.ones(6) / 6
    for i_batch in grouper(FLAGS.num_games_share_tree, iter(range(num_games))):
        if FLAGS.num_games_share_tree > 1:
            logging.info(f'\nStarting new batch : %d games', len(i_batch))
        open_move = np.random.choice(open_moves, p=open_probs)
        init_position = go.Position().play_move(
            coords.from_gtp(open_move)
        )
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
            # moves_history = player.root.position.recent
            # history_str = ' '.join([coords.to_gtp(x.move) for x in moves_history[:8]])
            history_str = ' '.join(player.move_infos[:8])

            ru_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

            logging.info(f'game {i}: %d moves, score %.1f l1n=%d rss=%.1f\t%s', player.root.position.n,
                         margin_est, len(shared_tree.children), ru_rss / 1e6,
                         history_str)

        del shared_tree


def main(argv):
    play_games(num_games=FLAGS.num_games)


if __name__ == '__main__':
    app.run(main)
