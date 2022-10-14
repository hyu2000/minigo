# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Python implementation of selfplay worker.

This worker is used to set up many parallel selfplay instances."""
import itertools
import random
import os
import socket
import time
from typing import Tuple

from absl import app, flags, logging
import numpy as np
import tensorflow as tf

import coords
import go
import k2net as dual_net
import mcts
import preprocessing
from sgf_wrapper import SGFReader
from strategies import MCTSPlayer
import utils
import myconf

flags.DEFINE_string('load_file', None, 'Path to model save files.')
flags.DEFINE_string('selfplay_dir', None, 'Where to write game data.')
flags.DEFINE_string('holdout_dir', None, 'Where to write held-out game data.')
flags.DEFINE_string('sgf_dir', None, 'Where to write human-readable SGFs.')
flags.DEFINE_float('holdout_pct', 0.05, 'What percent of games to hold out.')
flags.DEFINE_float('resign_disable_pct', 0.05, 'What percent of games to disable resign for.')

# From strategies.py
flags.declare_key_flag('verbose')
flags.declare_key_flag('num_readouts')


FLAGS = flags.FLAGS
PI_CONST = np.zeros(go.N * go.N + 1, dtype=np.float32)


def _format_move_info(move, best_move):
    move_info = coords.to_gtp(move)
    if move == best_move:
        return move_info
    return '%s(%s)' % (move_info, coords.to_gtp(best_move))


def should_full_search(player: MCTSPlayer) -> Tuple[bool, int]:
    """
    I don't understand this, but we retain this minigo logic anyways
    # we want to do "X additional readouts", rather than "up to X readouts".
    """
    current_readouts = player.root.N
    if FLAGS.full_readout_prob >= 1.:
        return True, current_readouts + FLAGS.num_readouts

    # if we have already searched the node enough, just push it over so we can record pi for policy training
    if FLAGS.num_readouts - current_readouts - FLAGS.num_fast_readouts < 50:
        return False, FLAGS.num_readouts

    # allow more full search at early games, rather than late games
    # range: 0.05 around full_readout_prob. Assume base=0.25, decrease from 0.30 to 0.20 over the game
    n = player.root.position.n
    accept_thresh = FLAGS.full_readout_prob + (50 - n) / 50 * 0.05
    if random.random() < accept_thresh:
        return True, current_readouts + FLAGS.num_readouts

    return False, current_readouts + FLAGS.num_fast_readouts


def play(network, init_position=None, init_root=None):
    """Plays out a self-play match, returning a MCTSPlayer object containing:
        - the final position
        - the n x 362 tensor of floats representing the mcts search probabilities
        - the n-ary tensor of floats representing the original value-net estimate
          where n is the number of moves in the game
    """
    # Disable resign in 5% of games
    if random.random() < FLAGS.resign_disable_pct:
        resign_threshold = -1.0
    else:
        resign_threshold = None

    player = MCTSPlayer(network, resign_threshold=resign_threshold)
    player.initialize_game(position=init_position, root=init_root)

    # Must run this once at the start to expand the root node.
    if not player.root.is_expanded:
        first_node = player.root.select_leaf()
        prob, val = network.run(first_node.position)
        first_node.incorporate_results(prob, val, first_node)

    while True:
        start = time.time()

        # play-cap randomization. We inject noise only when doing a full search
        inject_noise, total_readouts = should_full_search(player)
        if inject_noise:
            player.root.inject_noise()

        while player.root.N < total_readouts:
            player.tree_search()

        if FLAGS.verbose >= 3:
            print(player.root.position)
            print(player.root.describe())

        if player.should_resign():
            pos = player.root.position
            player.set_result(-1 * pos.to_play, was_resign=True)
            break

        move, best_move = player.pick_move(player.root.position.n < FLAGS.softpick_move_cutoff)
        orig_root = player.root  # play_move() dumps info, and it changes root
        player.play_move(move)
        player.add_move_info(_format_move_info(move, best_move))
        orig_root.uninject_noise()

        benson_score_details = player.root.position.score_benson()
        if benson_score_details.final:  # end the game when score is final
            player.set_result(np.sign(benson_score_details.score), was_resign=False)
            break
        if player.root.position.is_game_over():  # pass-pass
            if np.sign(player.root.Q) != np.sign(benson_score_details.score):
                logging.warning(f'Benson score {benson_score_details.score:.1f} is non-final. root.Q={player.root.Q:.1f}')
            player.set_result(np.sign(benson_score_details.score), was_resign=False)
            break
        if player.root.position.n >= FLAGS.max_game_length:
            # this is likely super-ko, should ignore game
            logging.warning(f'game exceeds {FLAGS.max_game_length}, void')
            player.set_result(0, was_resign=False)
            break

        if (FLAGS.verbose >= 2) or (FLAGS.verbose >= 1 and player.root.position.n % 10 == 7):
            dur = time.time() - start
            logging.info(f"move %d: Q=%.3f, {total_readouts} readouts ({dur:.2f} sec), %.3f sec/100",
                         player.root.position.n, player.root.Q, dur / total_readouts * 100.0)
        if FLAGS.verbose >= 3:
            print("Played >>",
                  coords.to_gtp(coords.from_flat(player.root.fmove)))

    if FLAGS.verbose >= 2:
        utils.dbg("%s: %.3f" % (player.result_string, player.root.Q))
        utils.dbg(player.root.position, player.root.position.score())

    return player


def create_dir_if_needed(selfplay_dir=None, holdout_dir=None, sgf_dir=None):
    """ run this only once """
    if sgf_dir is not None:
        # minimal_sgf_dir = os.path.join(sgf_dir, 'clean')
        full_sgf_dir = os.path.join(sgf_dir, 'full')
        # utils.ensure_dir_exists(minimal_sgf_dir)
        utils.ensure_dir_exists(full_sgf_dir)
    if selfplay_dir is not None:
        utils.ensure_dir_exists(selfplay_dir)
        utils.ensure_dir_exists(holdout_dir)


def run_game(dnn, init_position: go.Position=None, init_root: mcts.MCTSNode=None,
             game_id=None,
             selfplay_dir=None, holdout_dir=None,
             sgf_dir=None, holdout_pct=0.05) -> Tuple[MCTSPlayer, str]:
    """Takes a played game and record results and game data."""
    with utils.logged_timer(f"Playing game {game_id}"):
        player = play(dnn, init_position=init_position, init_root=init_root)

    if game_id:
        output_name = '{}-{}'.format(os.path.splitext(os.path.basename(game_id))[0], utils.microseconds_since_midnight())
    else:
        output_name = '{}-{}'.format(utils.microseconds_since_midnight(), socket.gethostname())
    game_data = player.extract_data()
    sgf_name = ''
    if sgf_dir is not None:
        sgf_name = output_name
        # with tf.io.gfile.GFile(os.path.join(minimal_sgf_dir, '{}.sgf'.format(sgf_name)), 'w') as f:
        #     f.write(player.to_sgf(use_comments=False))
        with tf.io.gfile.GFile(os.path.join(sgf_dir, 'full', f'{sgf_name}.sgf'), 'w') as f:
            f.write(player.to_sgf())

    if player.result == 0:  # void
        return player, sgf_name
    if selfplay_dir is None:
        return player, sgf_name

    # separate out data where we have policy target vs those we don't
    iter1, iter2 = itertools.tee(game_data)
    missing_pi = lambda x: x[1] is None
    iter1 = itertools.filterfalse(missing_pi, iter1)
    iter2 = filter(missing_pi, iter2)
    iter2 = map(lambda x: (x[0], PI_CONST, x[2]), iter2)
    tf_full_examples = preprocessing.make_dataset_from_selfplay(iter1)
    tf_nopi_examples = preprocessing.make_dataset_from_selfplay(iter2)

    tf_full_examples, tf_nopi_examples = list(tf_full_examples), list(tf_nopi_examples)
    # logging.info(f'{game_id}: %d full examples, %d value only', len(tf_full_examples), len(tf_nopi_examples))
    for sample_type, samples in [('full', tf_full_examples), ('nopi', tf_nopi_examples)]:
        if len(samples) == 0:
            continue

        # Hold out 5% of games for validation.
        dir_path = holdout_dir if random.random() < holdout_pct else selfplay_dir

        fname = os.path.join(dir_path, f'{output_name}.tfrecord.{sample_type}.zz')
        preprocessing.write_tf_examples(fname, samples)

    return player, sgf_name


def main9(argv):
    """Entry point for running one selfplay game."""
    del argv  # Unused
    flags.mark_flag_as_required('load_file')

    # B+0.5, KM5.5 (B+1.5 by Chinese scoring): one dead white group (1 eye); two empty spots
    init_sgf = '/Users/hyu/PycharmProjects/dlgo/9x9/games/Pro/9x9/Minigo/890826.sgf'
    # init_sgf = '/Users/hyu/PycharmProjects/dlgo/9x9/games/Pro/9x9/Minigo/001203.sgf'
    # W+3.5, KM6.5: well defined. only J9 up for grab, but black needs to protect G8 first. Also no need for B:D2
    # init_sgf = '/Users/hyu/PycharmProjects/dlgo/9x9/games/tmp/2.sgf'
    # B+9.5, KM5.5: finalized. two dead white stones
    # init_sgf = '/Users/hyu/PycharmProjects/dlgo/9x9/games/tmp/jrd-tromp-07-17-29.sgf'

    init_position = None
    if init_sgf:
        reader = SGFReader.from_file_compatible(init_sgf)
        init_position = reader.last_pos(ignore_final_pass=True)

    load_file = f'{myconf.MODELS_DIR}/endgame2_epoch_1.h5'
    with utils.logged_timer("Loading weights from %s ... " % load_file):
        network = dual_net.DualNetwork(load_file)

    create_dir_if_needed(selfplay_dir=FLAGS.selfplay_dir, holdout_dir=FLAGS.holdout_dir,
                         sgf_dir=FLAGS.sgf_dir)
    run_game(
        network,
        init_position=init_position, game_id=init_sgf,
        selfplay_dir=FLAGS.selfplay_dir,
        holdout_dir=FLAGS.holdout_dir,
        # selfplay_dir=f'{myconf.EXP_HOME}/selfplay/tfdata',
        # holdout_dir= f'{myconf.EXP_HOME}/selfplay/tfdata',
        holdout_pct=0.0,
        sgf_dir=f'{myconf.SELFPLAY_DIR}'
        # sgf_dir=FLAGS.sgf_dir
    )


def main(argv):
    """Entry point for running one selfplay game."""
    del argv  # Unused
    flags.mark_flag_as_required('load_file')

    init_sgf = None
    # init_sgf = '/Users/hyu/PycharmProjects/dlgo/5x5/games/mcts-study1.sgf'

    init_position = None
    init_position = go.Position().play_move(coords.from_gtp('C2'))
    if init_sgf:
        reader = SGFReader.from_file_compatible(init_sgf)
        init_position = reader.last_pos(ignore_final_pass=True)

    load_file = f'{myconf.MODELS_DIR}/model8_epoch_5.h5'
    with utils.logged_timer("Loading weights from %s ... " % load_file):
        network = dual_net.DualNetwork(load_file)

    create_dir_if_needed(selfplay_dir=FLAGS.selfplay_dir, holdout_dir=FLAGS.holdout_dir,
                         sgf_dir=myconf.SELFPLAY_DIR)

    shared_tree = mcts.MCTSNode(init_position)
    player, sgf_fname = run_game(
        network,
        init_position=init_position, game_id=init_sgf,
        init_root=shared_tree,
        selfplay_dir=FLAGS.selfplay_dir,
        holdout_dir=FLAGS.holdout_dir,
        # selfplay_dir=f'{myconf.EXP_HOME}/selfplay/tfdata',
        # holdout_dir= f'{myconf.EXP_HOME}/selfplay/tfdata',
        holdout_pct=0.0,
        sgf_dir=f'{myconf.SELFPLAY_DIR}'
        # sgf_dir=FLAGS.sgf_dir
    )
    print(len(shared_tree.children))


if __name__ == '__main__':
    app.run(main)
