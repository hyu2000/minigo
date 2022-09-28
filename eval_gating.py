""" revamp: attempt at a more accurate eval of two bots (including kata engine), using new dnn cache

- control randomness in each bot: open move, soft-picks (temperature?), noise
  my bot is controlled by configs
- measure game redundancy:
- bot uses a simpler interface than gtp: just move, and some debug info

PlayerInterface: ok, but doesn't feel quite right
"""
import logging
import time
from typing import List

import attr
import numpy as np
from absl import flags

import coords
import go
import utils
from run_selfplay import InitPositions

FLAGS = flags.FLAGS


class BasicPlayerInterface:
    """ simplest interface for playing a game """
    def initialize_game(self, position=None):
        """Initializes a new game. May start from a setup position
        """

    def play_move(self, c) -> bool:
        """ play the given move, to advance the game
        """

    def suggest_moves(self, position=None) -> List:
        """ return a list of ranked moves, together with probs, for the current position (can be overridden)

        returns [] if it's ready to resign
        """

    def set_result(self, winner, was_resign):
        """ Sets the game result. This ends the game
        """

    def get_game_comments(self) -> List[str]:
        """ player's comment on each move of the game
        """


@attr.s
class GameResult:
    black_margin: float = attr.ib()
    was_resign: bool = attr.ib()

    def winner(self):
        return 'B' if self.black_margin > 0 else 'W' if self.black_margin < 0 else '-'

    def sgf_str(self):
        winner = self.winner()
        if self.was_resign:
            return f'{winner}+R'
        if self.black_margin == 0:  # tie is always 'B+T'
            return 'B+T'
        margin = abs(self.black_margin)
        return f'{winner}+{margin}'


def test_result():
    result = GameResult(0, False)
    assert result.winner() == '-'
    assert result.sgf_str() == 'B+T'

    result = GameResult(1, True)
    assert result.sgf_str() == 'B+R'
    result = GameResult(-2, True)
    assert result.sgf_str() == 'W+R'

    result = GameResult(12, False)
    assert result.sgf_str() == 'B+12'
    result = GameResult(-2.5, False)
    assert result.sgf_str() == 'W+2.5'


class EvaluateOneSide:
    """ run evaluation, control for randomness """
    def __init__(self, black_config, white_config, sgf_dir: str):
        # self.black_model = black_model
        # self.white_model = white_model
        self.sgf_dir = sgf_dir

        utils.ensure_dir_exists(sgf_dir)
        utils.ensure_dir_exists(FLAGS.eval_data_dir)

        self.black_model_id = black_config.model_id()
        self.white_model_id = white_config.model_id()

        with utils.logged_timer("Loading weights"):
            self.black_net = None
            self.white_net = None  # dual_net.DualNetwork(white_config.model_path())
        self.black_player = None  # type: BasicPlayerInterface
        self.white_player = None  # type: BasicPlayerInterface

        self.init_positions = InitPositions(None, None)  #['C2'], [1.0])

        self._num_games_so_far = 0

    def _end_game(self, result: GameResult):
        for player in (self.black_player, self.white_player):
            player.set_result(result.black_margin, was_resign=result.was_resign)

    def _pick_move(self, cur_pos: go.Position, moves_with_probs: List):
        """ evaluator may soft-pick to increase game variety """
        # active.pick_move(active.root.position.n < FLAGS.softpick_move_cutoff)
        top_move = moves_with_probs[0][0]
        return top_move

    def _play_one_game(self, init_position: go.Position):
        """ """
        black, white = self.black_player, self.white_player
        for player in [black, white]:
            player.initialize_game(init_position)

        cur_pos = init_position  # type: go.Position
        max_game_length = FLAGS.max_game_length
        game_result = None
        for move_idx in range(init_position.n, max_game_length):
            if move_idx % 2:
                active, inactive = white, black
            else:
                active, inactive = black, white

            moves_with_probs = active.suggest_moves()

            if len(moves_with_probs) == 0:  # resigned
                game_result = GameResult(-1 * cur_pos.to_play, was_resign=True)
                break

            # evaluator picks the next move
            move, best_move = self._pick_move(cur_pos, moves_with_probs)

            # advance game
            active.play_move(move)
            inactive.play_move(move)
            cur_pos = cur_pos.play_move(move)

            benson_score_details = cur_pos.score_benson()
            if benson_score_details.final:  # end the game when score is final
                game_result = GameResult(benson_score_details.score, was_resign=False)
                break
            if cur_pos.n >= max_game_length or cur_pos.is_game_over():
                # score not final, but pass-pass or too long
                # todo Tromp score, or tie?
                logging.warning('ending game without a definite winner: %s', benson_score_details.score)
                game_result = GameResult(0, was_resign=False)
                break

        self._end_game(game_result)
        return cur_pos, game_result

    def _create_sgf(self, ith_game: int):
        """ merge comments from black and white """

    def _accumulate_stats(self):
        """ """

    def play_a_game(self):
        game_idx = self._num_games_so_far
        self._num_games_so_far += 1

        init_position = self.init_positions.sample()
        final_pos, game_result = self._play_one_game(init_position)

        # accu game stats
        self._accumulate_stats(final_pos, game_result)

        # generate sgf
        self._create_sgf(game_idx)

        game_history = final_pos.recent
        move_history_head = ' '.join([coords.to_gtp(game_history[i].move) for i in range(12)])
        logging.info(f'Finished game %3d: %3d moves, %-7s   %s', game_idx, len(game_history), game_result.sgf_str(), move_history_head)
        return game_result
