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

"""Evaluation plays games between two neural nets."""
import os
import random
import time
from collections import defaultdict

import attr
from typing import Tuple, Dict, Sequence, List

import pandas as pd
from absl import app, flags, logging
from tensorflow.python import gfile

import coords
import k2net as dual_net
import go
from run_selfplay import InitPositions
from strategies import MCTSPlayer
import sgf_wrapper
import utils
import myconf

# None = unlimited, 0 = auto-detect
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

flags.DEFINE_string('eval_sgf_dir', None, 'Where to write evaluation results.')
flags.DEFINE_integer('num_eval_games', 16, 'How many games to play')
# From strategies.py
flags.declare_key_flag('verbose')

FLAGS = flags.FLAGS

NUM_OPEN_MOVES = 6


class Ledger:
    """ keep tab on win/loss records of two players where they take equal black/white role

    No longer check for redundancy, as 9x9 games with soft-pick rarely repeats
    """
    def __init__(self):
        self._wins_by_black = defaultdict(int)
        self._games_by_black = defaultdict(int)
        self._model_ids = set()

    def _parse_winner_side(self, result_str: str) -> int:
        winner = result_str.upper()[0]
        return 1 if winner == 'B' else -1

    def record_game(self, black_model_id: str, white_model_id: str, result_str):
        winner = self._parse_winner_side(result_str)
        if winner == 1:
            self._wins_by_black[black_model_id] += 1
        self._games_by_black[black_model_id] += 1

        self._model_ids.add(black_model_id)
        self._model_ids.add(white_model_id)
        assert len(self._model_ids) == 2

    def report(self):
        results_dict = {}
        for model in self._model_ids:
            opponent = (self._model_ids - {model}).pop()
            wins_as_white = self._games_by_black[opponent] - self._wins_by_black[opponent]
            d = {'Black': self._wins_by_black[model],
                 'White': wins_as_white,
                 'Total': (self._wins_by_black[model] + wins_as_white)}
            results_dict[model] = pd.Series(d)
        df = pd.DataFrame(results_dict).sort_index(axis=1)
        # df.index.name = '#wins'
        print(df)
        return df


def test_ledger(argv):
    ledger = Ledger()
    models = list('AB')
    results = ['B+R', 'W+R']
    for i in range(5):
        result = random.choice(results)
        print(f'A as black: {result}')
        ledger.record_game('A', 'B', result)
        if i < 3:
            result = random.choice(results)
            print(f'B as black: {result}')
            ledger.record_game('B', 'A', result)
    ledger.report()


class RunOneSided:
    def __init__(self, black_model: str, white_model: str, sgf_dir: str):
        self.black_model = black_model
        self.white_model = white_model
        self.sgf_dir = sgf_dir

        self.black_model_id, self.white_model_id = get_model_id(black_model), get_model_id(white_model)

        with utils.logged_timer("Loading weights"):
            self.black_net = dual_net.DualNetwork(black_model)
            self.white_net = dual_net.DualNetwork(white_model)
        self.black_player = MCTSPlayer(self.black_net, two_player_mode=True, num_readouts=400)
        self.white_player = MCTSPlayer(self.white_net, two_player_mode=True, num_readouts=400)

        self.init_positions = InitPositions(None, None)

        self._num_games_so_far = 0

    def play(self, init_position: go.Position) -> MCTSPlayer:
        """ return None if game is not played """
        black, white = self.black_player, self.white_player
        for player in [black, white]:
            player.initialize_game(init_position)
            first_node = player.root.select_leaf()
            prob, val = player.network.run(first_node.position)
            first_node.incorporate_results(prob, val, first_node)

        for num_move in range(500):
            start = time.time()
            if num_move % 2:
                active, inactive = white, black
            else:
                active, inactive = black, white

            current_readouts = active.root.N
            while active.root.N < current_readouts + active.num_readouts:
                active.tree_search()

            # print some stats on the search
            if FLAGS.verbose >= 3:
                print(active.root.position)

            # First, check the roots for hopeless games.
            if active.should_resign():  # Force resign
                active.set_result(-1 * active.root.position.to_play, was_resign=True)
                inactive.set_result(active.root.position.to_play, was_resign=True)
                break

            if active.is_done():
                break

            move, best_move = active.pick_move(active.root.position.n < FLAGS.softpick_move_cutoff)
            active.play_move(move)
            inactive.play_move(move)

            if num_move < NUM_OPEN_MOVES:
                history = active.root.position.recent
                assert history[-1].move == move

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
            inactive.set_result(active.root.position.result(), was_resign=False)

        return active

    def _create_sgf(self, ith_game: int, init_position_n: int):
        """ merge comments from black and white """
        fname = "{:d}-{:s}-vs-{:s}-{:d}.sgf".format(int(time.time()), self.white_model_id, self.black_model_id, ith_game)
        game_history = self.black_player.position.recent
        if len(game_history) < len(self.white_player.position.recent):
            game_history = self.white_player.position.recent
        black_comments = self.black_player.comments
        white_comments = self.white_player.comments
        assert len(black_comments) == len(white_comments) and len(black_comments) + init_position_n == len(game_history)
        comments = ['' for i in range(init_position_n)]
        comments.extend([black_comments[i] if i % 2 == 0 else white_comments[i] for i in range(len(black_comments))])

        with gfile.GFile(os.path.join(self.sgf_dir, fname), 'w') as _file:
            sgfstr = sgf_wrapper.make_sgf(game_history,
                                          self.black_player.result_string, komi=self.black_player.position.komi,
                                          comments=comments,
                                          black_name=self.black_model_id, white_name=self.white_model_id)
            _file.write(sgfstr)

    def play_a_game(self):
        game_idx = self._num_games_so_far
        self._num_games_so_far += 1

        init_position = self.init_positions.sample()
        active = self.play(init_position)
        assert active is not None
        result_str = active.result_string

        self._create_sgf(game_idx, init_position.n)
        game_history = active.position.recent
        move_history_head = ' '.join([coords.to_gtp(game_history[i].move) for i in range(5)])
        logging.info(f'Finished game {game_idx}: %d moves, {result_str} \t%s', len(game_history), move_history_head)
        return result_str


def get_model_id(model_path: str) -> str:
    basename = os.path.basename(model_path)
    model_id, _ = os.path.splitext(basename)
    return model_id


def main(argv):
    """Play matches between two neural nets."""
    _, black_model, white_model = argv
    models_dir = f'{myconf.EXP_HOME}/pbt'
    if not black_model.startswith('/'):
        black_model = f'{models_dir}/{black_model}'
    if not white_model.startswith('/'):
        white_model = f'{models_dir}/{white_model}'
    utils.ensure_dir_exists(FLAGS.eval_sgf_dir)

    ledger = Ledger()
    runner1 = RunOneSided(black_model, white_model, FLAGS.eval_sgf_dir)
    runner2 = RunOneSided(white_model, black_model, FLAGS.eval_sgf_dir)
    logging.info('Tournament: %s vs %s, %d games', runner1.black_model_id, runner1.white_model_id, FLAGS.num_eval_games)
    for i in range(FLAGS.num_eval_games):
        result_str = runner1.play_a_game()
        ledger.record_game(runner1.black_model_id, runner1.white_model_id, result_str)
        result_str = runner2.play_a_game()
        ledger.record_game(runner2.black_model_id, runner2.white_model_id, result_str)
        if i % 2 == 1:
            ledger.report()

    ledger.report()


if __name__ == '__main__':
    flags.mark_flag_as_required('eval_sgf_dir')
    app.run(main)
    # app.run(test_ledger)
