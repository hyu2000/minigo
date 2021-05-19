""" selfplay on endgames """
import myconf
from selfplay import run_game
from tar_dataset import GameStore
import k2net as dual_net
from absl import logging, app, flags


flags.DEFINE_string('tar_dir', None, 'Where to find TarDataSets.')

FLAGS = flags.FLAGS


def play_endgame():
    """ use DNN to score endgame, measure how accurate it is to the RE record
    some MCTS lookahead might help
    """
    store = GameStore(data_dir=FLAGS.tar_dir)
    game_iter = store.game_iter([store.ds_pro, store.ds_top], filter_game=True)

    # model_file = f'{myconf.MODELS_DIR}/model3_epoch_5.h5'
    model_file = FLAGS.load_file
    network = dual_net.DualNetwork(model_file)
    for i, (game_id, reader) in enumerate(game_iter):
        pos = reader.last_pos(ignore_final_pass=True)

        _, val0 = network.run(pos)
        if True:
            player = run_game(network, init_position=pos,
                              selfplay_dir=FLAGS.selfplay_dir,
                              holdout_dir=FLAGS.holdout_dir,
                              holdout_pct=FLAGS.holdout_pct,
                              sgf_dir=FLAGS.sgf_dir
                              )
            margin_est = player.black_margin_no_komi
            margin_rec = reader.black_margin_adj(adjust_komi=True)
            final_pos = player.root.position
            _, val1 = network.run(final_pos)
            final_q = player.root.Q
            logging.info(f'scoring {game_id}: RE=( %.1f %.1f ) vs %.1f \t{pos.n} => {final_pos.n} \t%.1f %.1f %.2f %d',
                         reader.komi(), margin_rec, margin_est, val0, val1, final_q, player.root.N)


def main(argv):
    play_endgame()


if __name__ == '__main__':
    app.run(main)
