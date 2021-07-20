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

"""Evalation plays games between two neural nets."""

import os
import time
from absl import app, flags
from tensorflow.python import gfile

import coords
import k2net as dual_net
from strategies import MCTSPlayer
import sgf_wrapper
import utils
import myconf

flags.DEFINE_string('eval_sgf_dir', None, 'Where to write evaluation results.')

flags.DEFINE_integer('num_evaluation_games', 16, 'How many games to play')

# From strategies.py
flags.declare_key_flag('num_readouts')
flags.declare_key_flag('verbose')

FLAGS = flags.FLAGS


def play_tournament(black_model, white_model, num_games, sgf_dir):
    """Plays matches between two neural nets.

    Args:
        black_model: Path to the model for black player
        white_model: Path to the model for white player
    """
    with utils.logged_timer("Loading weights"):
        black_net = dual_net.DualNetwork(black_model)
        white_net = dual_net.DualNetwork(white_model)

    readouts = FLAGS.num_readouts

    black = MCTSPlayer(black_net, two_player_mode=True, num_readouts=200)
    white = MCTSPlayer(white_net, two_player_mode=True, num_readouts=200)

    black_name = os.path.basename(black_net.save_file)
    white_name = os.path.basename(white_net.save_file)

    for i in range(num_games):
        active = play_game(black, white)

        fname = "{:d}-{:s}-vs-{:s}-{:d}.sgf".format(int(time.time()),
                                                    white_name, black_name, i)
        if active.result == 0:
            active.set_result(active.root.position.result(), was_resign=False)
        game_history = active.position.recent
        with gfile.GFile(os.path.join(sgf_dir, fname), 'w') as _file:
            sgfstr = sgf_wrapper.make_sgf(game_history,
                                          active.result_string, komi=active.position.komi,
                                          black_name=black_name, white_name=white_name)
            _file.write(sgfstr)
        move_history_head = ' '.join([coords.to_gtp(game_history[i].move) for i in range(5)])
        print(f'Finished game {i}: #moves=%d %d %d {active.result_string} %s' % (
            len(game_history), black.num_readouts, white.num_readouts, move_history_head))


def play_game(black, white) -> MCTSPlayer:
    num_move = 0  # The move number of the current game
    for player in [black, white]:
        player.initialize_game()
        first_node = player.root.select_leaf()
        prob, val = player.network.run(first_node.position)
        first_node.incorporate_results(prob, val, first_node)
    while True:
        start = time.time()
        active = white if num_move % 2 else black
        inactive = black if num_move % 2 else white

        current_readouts = active.root.N
        while active.root.N < current_readouts + active.num_readouts:
            active.tree_search()

        # print some stats on the search
        if FLAGS.verbose >= 3:
            print(active.root.position)

        # First, check the roots for hopeless games.
        if active.should_resign():  # Force resign
            active.set_result(-1 *
                              active.root.position.to_play, was_resign=True)
            inactive.set_result(
                active.root.position.to_play, was_resign=True)

        if active.is_done():
            return active

        move = active.pick_move()
        active.play_move(move)
        inactive.play_move(move)

        dur = time.time() - start
        num_move += 1

        if (FLAGS.verbose > 1):  # or (FLAGS.verbose == 1 and num_move % 10 == 9):
            timeper = (dur / active.num_readouts) * 100.0
            print(active.root.position)
            print("%d: %d readouts, %.3f s/100. (%.2f sec)" % (num_move,
                                                               active.num_readouts,
                                                               timeper,
                                                               dur))


def main(argv):
    """Play matches between two neural nets."""
    _, black_model, white_model = argv
    utils.ensure_dir_exists(FLAGS.eval_sgf_dir)
    play_tournament(black_model, white_model, FLAGS.num_evaluation_games, FLAGS.eval_sgf_dir)
    # play_match(f'{myconf.MODELS_DIR}/endgame1_epoch_2.h5', f'{myconf.MODELS_DIR}/endgame1_epoch_2.h5',
    #            FLAGS.num_evaluation_games, FLAGS.eval_sgf_dir)


if __name__ == '__main__':
    flags.mark_flag_as_required('eval_sgf_dir')
    app.run(main)
