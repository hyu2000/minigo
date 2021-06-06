""" selfplay on endgames """
import numpy as np
import myconf
from selfplay import run_game, create_dir_if_needed
from tar_dataset import GameStore
import k2net as dual_net
from absl import logging, app, flags


flags.DEFINE_string('tar_dir', None, 'Where to find TarDataSets.')

FLAGS = flags.FLAGS


def play_endgames(start_idx=0):
    """ use DNN to score endgame, measure how accurate it is to the RE record
    some MCTS lookahead might help
    """
    store = GameStore(data_dir=FLAGS.tar_dir)
    game_iter = store.game_iter([store.ds_nngs], filter_game=True)

    # model_file = f'{myconf.MODELS_DIR}/model3_epoch_5.h5'
    model_file = FLAGS.load_file
    network = dual_net.DualNetwork(model_file)
    create_dir_if_needed(selfplay_dir=FLAGS.selfplay_dir, holdout_dir=FLAGS.holdout_dir,
                         sgf_dir=FLAGS.sgf_dir)

    for i, (game_id, reader) in enumerate(game_iter):
        if i > 500:
            break
        if i < start_idx:
            continue
        pos = reader.last_pos(ignore_final_pass=True)

        tromp0 = pos.score() + pos.komi
        _, val0 = network.run(pos)
        if True:
            player = run_game(network, init_position=pos, game_id=game_id,
                              selfplay_dir=FLAGS.selfplay_dir,
                              holdout_dir=FLAGS.holdout_dir,
                              holdout_pct=FLAGS.holdout_pct,
                              sgf_dir=FLAGS.sgf_dir
                              )
            margin_est = player.black_margin_no_komi
            margin_rec = reader.black_margin_adj(adjust_komi=True)
            if margin_rec is None:
                margin_rec = np.nan
            final_pos = player.root.position
            _, val1 = network.run(final_pos)
            final_q = player.root.Q
            logging.info(f'{i} scoring {game_id}: RE=( %.1f %.1f ) %.1f => %.1f \t\t%.1f %.1f',
                         reader.komi(), margin_rec, tromp0, margin_est, val0, val1)


def main(argv):
    play_endgames(start_idx=0)


if __name__ == '__main__':
    app.run(main)
