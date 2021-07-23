import numpy as np

import coords
import myconf
from selfplay import run_game, create_dir_if_needed
from tar_dataset import GameStore
import k2net as dual_net
from absl import logging, app, flags


flags.DEFINE_integer('num_games', 5, '#games to play')


FLAGS = flags.FLAGS


def play_games(num_games=500):
    """ """
    model_file = FLAGS.load_file
    if model_file:
        network = dual_net.DualNetwork(model_file)
    else:
        network = dual_net.DummyNetwork()

    create_dir_if_needed(selfplay_dir=FLAGS.selfplay_dir, holdout_dir=FLAGS.holdout_dir,
                         sgf_dir=FLAGS.sgf_dir)

    for i in range(num_games):
        player = run_game(network,
                          selfplay_dir=FLAGS.selfplay_dir,
                          holdout_dir=FLAGS.holdout_dir,
                          holdout_pct=FLAGS.holdout_pct,
                          sgf_dir=FLAGS.sgf_dir
                          )
        margin_est = player.black_margin_no_komi
        moves_history = player.root.position.recent
        history_str = ' '.join([coords.to_gtp(moves_history[i].move) for i in range(10)])
        logging.info(f'game {i}: %d moves, final margin %.1f \t%s', player.root.position.n, margin_est, history_str)


def main(argv):
    play_games(num_games=FLAGS.num_games)


if __name__ == '__main__':
    app.run(main)
