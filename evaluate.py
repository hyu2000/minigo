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
flags.DEFINE_integer('num_evaluation_games', 16, 'How many games to play')
# From strategies.py
flags.declare_key_flag('verbose')

FLAGS = flags.FLAGS

NUM_OPEN_MOVES = 6


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
    def __init__(self, num_open_moves, max_verify_games=4):
        self.num_open_moves = num_open_moves
        self._result_map = dict()  # type: Dict[Tuple, Outcome]
        self._num_verify_games = 0
        self._max_verify_games = max_verify_games

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

        if outcome.count == 1 and self._num_verify_games < self._max_verify_games:
            logging.info('found opening %s, rerun', ' '.join(gtp_moves))
            self._num_verify_games += 1
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

    def to_df(self) -> pd.DataFrame:
        """ format result_map as a DataFrame """
        def format_outcome(outcome: Outcome) -> Dict:
            d = attr.asdict(outcome)
            d['moves'] = len(d['moves'])
            return d

        result_dict = {' '.join(k): format_outcome(v) for k, v in self._result_map.items()}
        df = pd.DataFrame.from_dict(result_dict, orient='index')
        return df

    def report(self):
        print('Tournament Stats:')
        df = self.to_df()
        print(df.sort_values('count', ascending=False))


class RunTournament:
    def __init__(self, black_model: str, white_model: str):
        self.black_model = black_model
        self.white_model = white_model
        self.black_model_id, self.white_model_id = get_model_id(black_model), get_model_id(white_model)

        with utils.logged_timer("Loading weights"):
            self.black_net = dual_net.DualNetwork(black_model)
            self.white_net = dual_net.DualNetwork(white_model)
        self.black_player = MCTSPlayer(self.black_net, two_player_mode=True, num_readouts=400)
        self.white_player = MCTSPlayer(self.white_net, two_player_mode=True, num_readouts=400)

        self.init_positions = InitPositions(None, None)

    def play_game(self, init_position: go.Position, redundancy_checker: RedundancyChecker) -> MCTSPlayer:
        """ return None if game is not played """
        black, white = self.black_player, self.white_player
        num_move = 0  # The move number of the current game
        for player in [black, white]:
            player.initialize_game(init_position)
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

            move, best_move = active.pick_move(active.root.position.n < FLAGS.softpick_move_cutoff)
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

    def play_tournament(self, num_games, sgf_dir):
        """Plays matches between two neural nets.

        Args:
            black_model: Path to the model for black player
            white_model: Path to the model for white player
        """
        redundancy_checker = RedundancyChecker(num_open_moves=NUM_OPEN_MOVES)

        for i in range(num_games):
            init_position = self.init_positions.sample()
            active = self.play_game(init_position, redundancy_checker)
            if active is None:
                continue

            fname = "{:d}-{:s}-vs-{:s}-{:d}.sgf".format(int(time.time()), self.white_model_id, self.black_model_id, i)
            game_history = active.position.recent
            with gfile.GFile(os.path.join(sgf_dir, fname), 'w') as _file:
                sgfstr = sgf_wrapper.make_sgf(game_history,
                                              active.result_string, komi=active.position.komi,
                                              black_name=self.black_model_id, white_name=self.white_model_id)
                _file.write(sgfstr)
            move_history_head = ' '.join([coords.to_gtp(game_history[i].move) for i in range(5)])
            logging.info(f'Finished game {i}: #moves=%d %d %d {active.result_string} %s',
                         len(game_history), self.black_player.num_readouts, self.white_player.num_readouts, move_history_head)

        return redundancy_checker


def get_model_id(model_path: str) -> str:
    basename = os.path.basename(model_path)
    model_id, _ = os.path.splitext(basename)
    return model_id


def main(argv):
    """Play matches between two neural nets."""
    _, black_model, white_model = argv
    if not black_model.startswith('/'):
        black_model = f'{myconf.MODELS_DIR}/{black_model}'
    if not white_model.startswith('/'):
        white_model = f'{myconf.MODELS_DIR}/{white_model}'
    utils.ensure_dir_exists(FLAGS.eval_sgf_dir)

    runner = RunTournament(black_model, white_model)
    black_model_id, white_model_id = runner.black_model_id, runner.white_model_id

    logging.info('Tournament: %s vs %s', black_model_id, white_model_id)
    ledger1 = runner.play_tournament(FLAGS.num_evaluation_games, FLAGS.eval_sgf_dir)
    df1 = ledger1.to_df()
    print(df1)
    logging.info('Tournament: %s vs %s', white_model_id, black_model_id)
    ledger2 = runner.play_tournament(FLAGS.num_evaluation_games, FLAGS.eval_sgf_dir)
    df2 = ledger2.to_df()
    print(df2)

    logging.info('Combining both runs')
    df = join_and_format(df1, df2, black_model_id, white_model_id)
    print(df.fillna('-'))


def join_and_format(df1: pd.DataFrame, df2: pd.DataFrame, black_id: str, white_id: str) -> pd.DataFrame:
    df1.columns = pd.MultiIndex.from_product([[black_id], df1.columns])
    df2.columns = pd.MultiIndex.from_product([[white_id], df2.columns])
    df = df1.join(df2, how='outer')
    df['count_max'] = df.xs('count', axis=1, level=1).max(axis=1)
    df = df.sort_values('count_max', ascending=False).drop('count_max', axis=1)
    return df


def test_report(argv):
    d1 = {
        ('C3', 'D3'): Outcome(tuple('abcd'), 'B+7', 3),
        ('C3', 'D2'): Outcome(tuple('abcdef'), 'W+2', 1),  # common
        ('B3', 'D2'): Outcome(tuple('abcdefg'), 'B+3', 2),
    }
    ledger1 = RedundancyChecker(2)
    ledger1._result_map = d1
    df1 = ledger1.to_df()
    d2 = {
        ('C3', 'D3'): Outcome(tuple('abcde'), 'B+5', 4),
        ('C3', 'C4'): Outcome(tuple('abcdef'), 'B+2', 2),
        ('C3', 'D2'): Outcome(tuple('abcdeg'), 'W+1', 1),  # common
    }
    ledger2 = RedundancyChecker(2)
    ledger2._result_map = d2
    df2 = ledger2.to_df()

    df = join_and_format(df1, df2, get_model_id('/models/m1.h5'), get_model_id('m2.h5'))
    print(df.fillna('-'))


if __name__ == '__main__':
    flags.mark_flag_as_required('eval_sgf_dir')
    app.run(main)
    # app.run(test_report)
