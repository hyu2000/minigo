""" selfplay on endgames on Top50, to learn life-and-death

Top50 does have early pass-pass, e.g.
go9/2015-02-10T00\:46\:35.809Z_2tswv91qi3h0.sgf
"""
import glob
import os

import numpy as np
import myconf
from selfplay import run_game, create_dir_if_needed
from tar_dataset import GameStore
import k2net as dual_net
from absl import logging, app, flags


flags.DEFINE_integer('num_games', 5, '#games to play')
flags.DEFINE_string('tar_dir', None, 'Where to find TarDataSets.')

FLAGS = flags.FLAGS


def check_game_processed(sgf_dir: str, base_game_id: str) -> bool:
    """ check game_id is already processed
    """
    if glob.glob(f'{sgf_dir}/full/{base_game_id}-*.sgf'):
        logging.info(f'Already processed, skipping {base_game_id} ...')
        return True
    return False


def play_endgames():
    """ to parallelize selfplay (and handle M1 TF random hang), we use a rudimentary mechanism:
    - game_iter(shuffle=True)
    - check whether the game has already been processed before proceeding
    """
    store = GameStore(data_dir=FLAGS.tar_dir)
    game_iter = store.game_iter([store.ds_top], filter_game=True, shuffle=True)

    # model_file = f'{myconf.MODELS_DIR}/model3_epoch_5.h5'
    model_file = FLAGS.load_file
    network = dual_net.DualNetwork(model_file)
    create_dir_if_needed(selfplay_dir=FLAGS.selfplay_dir, holdout_dir=FLAGS.holdout_dir,
                         sgf_dir=FLAGS.sgf_dir)

    i = 0
    for game_id, reader in game_iter:
        # this might only work in top50, where game_id is always 'go9/2015-*.sgf'
        base_game_id = os.path.splitext(os.path.basename(game_id))[0]
        if check_game_processed(FLAGS.sgf_dir, base_game_id):
            continue

        init_pos = reader.last_pos(ignore_final_pass=True)

        player = run_game(network, init_position=init_pos,
                          game_id=game_id,
                          selfplay_dir=FLAGS.selfplay_dir,
                          holdout_dir=FLAGS.holdout_dir,
                          holdout_pct=FLAGS.holdout_pct,
                          sgf_dir=FLAGS.sgf_dir
                          )

        logging.info(f'{i} %s %d -> %d  RE %s  final %s', base_game_id,
                     init_pos.n, player.root.position.n,
                     reader.result_str(), player.result_string)
        i += 1
        if i >= FLAGS.num_games:
            break


def main(argv):
    play_endgames()


if __name__ == '__main__':
    app.run(main)
