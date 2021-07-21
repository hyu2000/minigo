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
import attr
from typing import Tuple, Dict, Sequence

from absl import app, flags, logging
from tensorflow.python import gfile

import coords
import k2net as dual_net
import go
from strategies import MCTSPlayer
import sgf_wrapper
import utils
import myconf

flags.DEFINE_string('eval_sgf_dir', None, 'Where to write evaluation results.')

flags.DEFINE_integer('num_evaluation_games', 16, 'How many games to play')

# From strategies.py
flags.declare_key_flag('verbose')

FLAGS = flags.FLAGS

NUM_OPEN_MOVES = 2


@attr.s
class Outcome(object):
    moves = attr.ib()
    result = attr.ib()
    count = attr.ib(default=1)


class RedundancyChecker(object):
    """ In evaluation mode, bot runs in a more-deterministic mode.
    If the first two moves are the same, game will most likely end up the same.
    Use this to avoid repetitive work.

    We can also allow to play a couple extra games, just to validate that this is indeed the case.
    """
    def __init__(self, num_open_moves):
        self.num_open_moves = num_open_moves
        self._result_map = dict()  # type: Dict[Tuple, Outcome]

    @staticmethod
    def _player_moves_to_gtp(moves: Sequence[go.PlayerMove]) -> Sequence[str]:
        return tuple(coords.to_gtp(x.move) for x in moves)

    def should_continue(self, initial_moves: Sequence[go.PlayerMove]) -> bool:
        """ client calls this to check whether it should continue the current game
        Note client might call this multiple times in a game
        """
        if len(initial_moves) < self.num_open_moves:
            return True
        
        gtp_moves = self._player_moves_to_gtp(initial_moves)
        key = gtp_moves[:self.num_open_moves]
        outcome = self._result_map.get(key)
        if outcome is None:
            return True

        if outcome.count == 1:
            logging.info('found opening %s, rerun', ' '.join(gtp_moves))
            return True
        logging.info('dup opening: %s, should skip', ' '.join(gtp_moves))
        return False

    def record_game(self, move_history: Sequence[go.PlayerMove], result_str):
        """ client calls this to log a finished game """
        move_history = self._player_moves_to_gtp(move_history)
        
        key = move_history[:self.num_open_moves]
        outcome = self._result_map.get(key)
        if outcome is None:
            self._result_map[key] = Outcome(move_history, result_str)
            return
        if outcome.moves != move_history or outcome.result != result_str:
            logging.warning('Different results for same opening: %s %s Moves=\n%s\n%s',
                            outcome.result, result_str,
                            ' '.join(outcome.moves), ' '.join(move_history))
        else:
            # dup game with the same result
            outcome.count += 1

    def record_aborted_game(self, initial_moves: Sequence[go.PlayerMove]):
        """ client log a game that's considered dup """
        gtp_moves = self._player_moves_to_gtp(initial_moves)
        assert len(gtp_moves) >= self.num_open_moves

        key = gtp_moves[:self.num_open_moves]
        assert key in self._result_map
        outcome = self._result_map[key]
        outcome.count += 1

    def report(self):
        print('Tournament Stats:')
        print('open\tresult\tcount\t#moves')
        for k, outcome in self._result_map.items():
            print('%s \t%s\t %d\t %d' % (' '.join(k), outcome.result, outcome.count, len(outcome.moves)))


def play_game(black: MCTSPlayer, white: MCTSPlayer, redundancy_checker: RedundancyChecker) -> MCTSPlayer:
    """ return None if game is not played """
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
            break

        move = active.pick_move()
        active.play_move(move)
        inactive.play_move(move)

        if num_move < NUM_OPEN_MOVES:
            history = active.root.position.recent
            assert history[-1].move == move
            keep_play = redundancy_checker.should_continue(history)
            if not keep_play:
                redundancy_checker.record_aborted_game(history)
                return None

        dur = time.time() - start
        num_move += 1

        if (FLAGS.verbose > 1):  # or (FLAGS.verbose == 1 and num_move % 10 == 9):
            timeper = (dur / active.num_readouts) * 100.0
            print(active.root.position)
            print("%d: %d readouts, %.3f s/100. (%.2f sec)" % (num_move,
                                                               active.num_readouts,
                                                               timeper,
                                                               dur))

    if active.result == 0:
        active.set_result(active.root.position.result(), was_resign=False)

    redundancy_checker.record_game(active.root.position.recent, active.result_string)
    return active


def play_tournament(black_model: str, white_model: str, num_games, sgf_dir):
    """Plays matches between two neural nets.

    Args:
        black_model: Path to the model for black player
        white_model: Path to the model for white player
    """
    with utils.logged_timer("Loading weights"):
        black_net = dual_net.DualNetwork(black_model)
        white_net = dual_net.DualNetwork(white_model)

    black = MCTSPlayer(black_net, two_player_mode=True, num_readouts=200)
    white = MCTSPlayer(white_net, two_player_mode=True, num_readouts=200)
    redundancy_checker = RedundancyChecker(num_open_moves=NUM_OPEN_MOVES)

    black_name = os.path.basename(black_net.save_file)
    white_name = os.path.basename(white_net.save_file)

    for i in range(num_games):
        active = play_game(black, white, redundancy_checker)
        if active is None:
            continue

        fname = "{:d}-{:s}-vs-{:s}-{:d}.sgf".format(int(time.time()), white_name, black_name, i)
        game_history = active.position.recent
        with gfile.GFile(os.path.join(sgf_dir, fname), 'w') as _file:
            sgfstr = sgf_wrapper.make_sgf(game_history,
                                          active.result_string, komi=active.position.komi,
                                          black_name=black_name, white_name=white_name)
            _file.write(sgfstr)
        move_history_head = ' '.join([coords.to_gtp(game_history[i].move) for i in range(5)])
        print(f'Finished game {i}: #moves=%d %d %d {active.result_string} %s' % (
            len(game_history), black.num_readouts, white.num_readouts, move_history_head))

    redundancy_checker.report()


def main(argv):
    """Play matches between two neural nets."""
    _, black_model, white_model = argv
    utils.ensure_dir_exists(FLAGS.eval_sgf_dir)
    # play_tournament(black_model, white_model, FLAGS.num_evaluation_games, FLAGS.eval_sgf_dir)
    play_tournament(f'{myconf.MODELS_DIR}/model5_epoch_3.h5', f'{myconf.MODELS_DIR}/model_epoch_2.h5',
                    12, f'{myconf.EXP_HOME}/eval')


if __name__ == '__main__':
    flags.mark_flag_as_required('eval_sgf_dir')
    app.run(main)
