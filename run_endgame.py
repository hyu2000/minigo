""" selfplay on endgames """
import myconf
from selfplay import run_game
from tar_dataset import GameStore
import k2net as dual_net
from absl import logging, app


def score_endgame():
    """ use DNN to score endgame, measure how accurate it is to the RE record
    some MCTS lookahead might help
    """
    store = GameStore()
    game_iter = store.game_iter([store.ds_pro], filter_game=True)

    model_file = f'{myconf.MODELS_DIR}/model3_epoch_5.h5'
    network = dual_net.DualNetwork(model_file)
    for game_id, reader in game_iter:
        pos = reader.last_pos()

        for i in range(3):
            player = run_game(network, init_position=pos)
            margin_est = player.black_margin_no_komi
            margin_rec = reader.black_margin_adj(adjust_komi=True)
            logging.info('%s: komi=%.1f RE=%.1f -> %.1f', game_id, reader.komi(), margin_rec, margin_est)
        break


def play_endgame():
    """ """


def main(argv):
    score_endgame()


if __name__ == '__main__':
    app.run(main)
