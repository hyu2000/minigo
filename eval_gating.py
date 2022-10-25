""" revamp: attempt at a more accurate eval of two bots (including kata engine), using new dnn cache

- control randomness in each bot: open move, soft-picks (temperature?), noise
  my bot is controlled by configs
- measure game redundancy:
- bot uses a simpler interface than gtp: just move, and some debug info

PlayerInterface: ok, but doesn't feel quite right
"""
import logging
import random
import time
from typing import List, Tuple

import attr
import numpy as np
from absl import flags, app

import coords
import k2net as dual_net
import go
import myconf
import sgf_wrapper
import utils
from model_config import ModelConfig
from katago.analysis_engine import ARequest, KataEngine, MoveInfo, RootInfo, assemble_comment, KataModels
from run_selfplay import InitPositions
from sgfs_stats import run_tournament_report
from strategies import MCTSPlayer

FLAGS = flags.FLAGS
# flags.mark_flags_as_required(['softpick_move_cutoff', 'softpick_topn_cutoff'])


class BasicPlayerInterface:
    """ simplest interface for playing an eval game
    """
    def id(self) -> str:
        """ id for the model config """

    def initialize_game(self, position: go.Position):
        """Initializes a new game. May start from a setup position
        """

    def play_move(self, c: tuple):
        """ play the given move, to advance the game
        """

    def suggest_moves(self, position: go.Position) -> List[Tuple]:
        """ return a list of ranked moves, together with probs, for the current position.

        position should match the internal state if player chooses to track position.
        play_move(c) will be called after this, which may not be the top move bot considered.

        returns [] if it's ready to resign
        """

    def set_result(self, winner, was_resign):
        """ Sets the game result. This ends the game
        """

    def get_game_comments(self) -> List[str]:
        """ player's comment on each move of the game
        """


class KataPlayer(BasicPlayerInterface):
    """ backed by kata analysis engine

    we don't really maintain game state here
    """
    def __init__(self, kata_engine: KataEngine, max_readouts=500):
        self.kata_engine = kata_engine
        self.max_readouts = max_readouts

    def id(self):
        return f'{self.kata_engine.model_id()}#{self.max_readouts}'

    def initialize_game(self, position):
        self.moves = [coords.to_gtp(move.move)
                      for i, move in enumerate(position.recent)]
        self.comments = ['init' for x in self.moves]
        self._resp1 = None
        self.win_rate = 0.5

    def play_move(self, c):
        """ this gets called whether or not it's our turn to move """
        move = coords.to_gtp(c)
        comment = 'not katas turn'
        if self._resp1 is not None:
            comment = assemble_comment(move, self._resp1)

        self.comments.append(comment)
        self.moves.append(move)

        self._resp1 = None

    def suggest_moves(self, position: go.Position) -> List:
        assert position.n == len(self.moves)
        kmoves = [['B' if i % 2 == 0 else 'W', move]
                  for i, move in enumerate(self.moves)]
        arequest = ARequest(kmoves, [len(kmoves)], maxVisits=self.max_readouts)
        responses = self.kata_engine.analyze(arequest)

        assert len(responses) == 1
        resp1 = responses[0]
        self._resp1 = resp1

        rinfo = RootInfo.from_dict(resp1.rootInfo)
        self.win_rate = rinfo.winrate
        top_moves = []
        top_n = FLAGS.softpick_topn_cutoff
        for minfo_dict in resp1.moveInfos[:top_n]:
            minfo = MoveInfo.from_dict(minfo_dict)
            # this may not sum up to 1
            prob = minfo.visits / rinfo.visits
            top_moves.append((minfo.move, prob))
        return top_moves

    def get_game_comments(self) -> List[str]:
        return self.comments


def arg_top_k_moves(pi: np.array, k) -> np.array:
    indices = np.argpartition(pi, -k)[-k:]
    return indices[np.argsort(-pi[indices])]


def notest_arg_top_k_moves():
    arr = np.arange(0, 1, .1)
    np.random.shuffle(arr)
    print(arr)
    top_k_indices = arg_top_k_moves(arr, 3)
    print(arr[top_k_indices])
    # assert all(arr[top_k_indices] == np.array([0.9, 0.8, 0.7]))


class K2Player(BasicPlayerInterface):
    """ k2net w/ mcts """
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.dnn = dual_net.load_net(model_config.model_path())
        self.mcts_player = MCTSPlayer(self.dnn, num_readouts=model_config.num_readouts)

    def id(self):
        return self.model_config.model_id()

    def initialize_game(self, position: go.Position):
        self.mcts_player.initialize_game(position)
        first_node = self.mcts_player.root.select_leaf()
        prob, val = self.mcts_player.network.run(first_node.position)
        first_node.incorporate_results(prob, val, first_node)

        self.mcts_player.comments = ['init' for x in range(position.n)]
        self.win_rate = 0.5

    def play_move(self, c: tuple):
        self.mcts_player.play_move(c, record_pi=False)

    def suggest_moves(self, position: go.Position) -> List:
        """ return a list of ranked moves, together with probs, for the current position.
        """
        active = self.mcts_player
        assert active.root.position.n == position.n
        current_readouts = active.root.N
        while active.root.N < current_readouts + active.num_readouts:
            active.tree_search()

        if active.should_resign():  # Force resign
            return []

        self.win_rate = (active.root.Q + 1) / 2   # transform my Q value to winrate
        pi = active.root.children_as_pi(squash=False).flatten()
        # restrict soft-pick to only top 5 moves
        move_indices = arg_top_k_moves(pi, FLAGS.softpick_topn_cutoff)
        top_moves = [(coords.flat_to_gtp(x), pi[x]) for x in move_indices]
        return top_moves

    def set_result(self, winner, was_resign):
        """ """

    def get_game_comments(self) -> List[str]:
        return self.mcts_player.comments


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


def notest_game_result():
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
    def __init__(self, black_player: BasicPlayerInterface, white_player: BasicPlayerInterface, sgf_dir: str):
        self.black_player = black_player
        self.white_player = white_player

        self.sgf_dir = sgf_dir
        utils.ensure_dir_exists(sgf_dir)

        self.init_positions = InitPositions(None, None)  #['C2'], [1.0])
        self._num_games_so_far = 0

    def _end_game(self, result: GameResult):
        for player in (self.black_player, self.white_player):
            player.set_result(result.black_margin, was_resign=result.was_resign)

    def _pick_move(self, cur_pos: go.Position, moves_with_probs: List):
        """ evaluator may soft-pick to increase game variety """
        # active.pick_move(active.root.position.n < FLAGS.softpick_move_cutoff)
        top_move = moves_with_probs[0][0]
        if cur_pos.n < FLAGS.softpick_move_cutoff:
            chosen_move = utils.choose_moves_with_probs(moves_with_probs, softpick_topn_cutoff=FLAGS.softpick_topn_cutoff)
        else:
            chosen_move = top_move
        return chosen_move, top_move

    def _check_benson_score(self, benson_score: float):
        """ see if both agrees with Benson """
        black_winrate, white_winrate = self.black_player.win_rate, self.white_player.win_rate
        if np.sign(black_winrate - 0.5) == np.sign(white_winrate - 0.5) == np.sign(benson_score):
            return
        logging.warning(f'Benson score {benson_score:.1f} is non-final. Black winrate={black_winrate:.1f}, white {white_winrate:.1f}')

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

            moves_with_probs = active.suggest_moves(cur_pos)

            if len(moves_with_probs) == 0:  # resigned
                game_result = GameResult(-1 * cur_pos.to_play, was_resign=True)
                break

            # evaluator picks the next move
            move, best_move = self._pick_move(cur_pos, moves_with_probs)

            # advance game
            c = coords.from_gtp(move)
            active.play_move(c)
            inactive.play_move(c)
            cur_pos = cur_pos.play_move(c)

            benson_score_details = cur_pos.score_benson()
            if benson_score_details.final:  # end the game when score is final
                game_result = GameResult(benson_score_details.score, was_resign=False)
                break
            if cur_pos.is_game_over():  # pass-pass: use Benson score, same as self-play
                self._check_benson_score(benson_score_details.score)
                game_result = GameResult(benson_score_details.score, was_resign=False)
                break
            if cur_pos.n >= max_game_length:
                logging.warning('ending game without a definite winner: %s', benson_score_details.score)
                game_result = GameResult(0, was_resign=False)
                break

        self._end_game(game_result)
        return cur_pos, game_result

    def _create_sgf(self, final_pos: go.Position, game_result: GameResult, sgf_fname: str):
        """ merge comments from black and white """
        black_comments = self.black_player.get_game_comments()
        white_comments = self.white_player.get_game_comments()
        assert len(black_comments) == len(white_comments) and len(black_comments) == final_pos.n
        comments = [black_comments[i] if i % 2 == 0 else white_comments[i] for i in range(final_pos.n)]

        with open(sgf_fname, 'w') as _file:
            sgfstr = sgf_wrapper.make_sgf(final_pos.recent,
                                          game_result.sgf_str(), komi=final_pos.komi,
                                          comments=comments,
                                          black_name=self.black_player.id(), white_name=self.white_player.id())
            _file.write(sgfstr)

    def _accumulate_stats(self):
        """ this needs to be done post-game, in case we run more games later """

    def play_a_game(self) -> GameResult:
        game_idx = self._num_games_so_far
        self._num_games_so_far += 1

        init_position = self.init_positions.sample()
        final_pos, game_result = self._play_one_game(init_position)

        # accu game stats
        # self._accumulate_stats(final_pos, game_result)
        sgf_fname = f'{self.sgf_dir}/{self.black_player.id()}-vs-{self.white_player.id()}-{game_idx}-%s.sgf' % (
                    utils.microseconds_since_midnight())
        self._create_sgf(final_pos, game_result, sgf_fname)

        game_history = final_pos.recent
        all_moves = [coords.to_gtp(x.move) for x in game_history]
        line = utils.format_game_summary(all_moves, game_result.sgf_str(), sgf_fname=sgf_fname)
        logging.info(f'Finished game %3d: %s', game_idx, line)
        return game_result

    def play_games(self, n: int):
        logging.info(f'eval between {self.black_player.id()} vs {self.white_player.id()}: {n} games')
        black_wins = 0
        for i in range(n):
            game_result = self.play_a_game()
            black_wins += game_result.black_margin > 0
        logging.info(f'  {self.black_player.id()} wins {black_wins} / {n} games')


def load_player(model_config: ModelConfig) -> BasicPlayerInterface:
    if model_config.is_kata_model():
        kata_engine = KataEngine(model_config.model_path()).start()
        return KataPlayer(kata_engine, model_config.num_readouts)

    return K2Player(model_config)


def run_one_side(black_player, white_player, sgf_dir, num_games: int):
    evaluator = EvaluateOneSide(black_player, white_player, f'{sgf_dir}')
    evaluator.play_games(num_games)


def main(argv):
    logging.info('softpick_move_cutoff = %d, softpick_topn_cutoff = %d', FLAGS.softpick_move_cutoff, FLAGS.softpick_topn_cutoff)
    _, black_id, white_id, eval_sgf_dir, num_games = argv
    num_games = int(num_games)
    black_player = load_player(ModelConfig(black_id))
    white_player = load_player(ModelConfig(white_id))
    return run_one_side(black_player, white_player, eval_sgf_dir, num_games)


def main_kata(argv):
    """ eval against kata remains serial, as it's likely not a good idea to run multiple KataEngine
    """
    num_readouts = 400
    logging.info('softpick_move_cutoff = %d, softpick_topn_cutoff = %d', FLAGS.softpick_move_cutoff, FLAGS.softpick_topn_cutoff)
    sgf_dir_root = f'{myconf.EXP_HOME}/eval_bots-model9/model9_4-vs-elo5k#400'

    player1id = f'{KataModels.MODEL_B6_5k}#{num_readouts}'
    # player1id = f'model11_2#{num_readouts}'
    player2id = f'model9_4#{num_readouts}'

    num_games_per_side = 4
    FLAGS.softpick_move_cutoff = 8
    FLAGS.reduce_symmetry_before_move = 3
    logging.info('softpick_move_cutoff = %d, softpick_topn_cutoff = %d, reduce_symmetry_before_move = %d',
                 FLAGS.softpick_move_cutoff, FLAGS.softpick_topn_cutoff, FLAGS.reduce_symmetry_before_move)

    # share KataEngine between two runs
    player1 = load_player(ModelConfig(player1id))
    player2 = load_player(ModelConfig(player2id))
    run_one_side(player1, player2, f'{sgf_dir_root}', num_games_per_side)
    run_one_side(player2, player1, f'{sgf_dir_root}', num_games_per_side)

    run_tournament_report(f'{sgf_dir_root}/*')


if __name__ == '__main__':
    # app.run(main_kata)
    app.run(main)
