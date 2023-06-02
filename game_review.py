"""
given a game, annotate what bot will do, same way a human reviews a game

Similar to test_kata analyze_game()
"""
from typing import List, Iterable

from absl import logging
from absl import app as abslapp
import numpy as np
import go
import coords
import sgf_wrapper
from k2net import DualNetwork, CoreMLNet
import myconf
from sgf_wrapper import replay_sgf, SGFReader
from strategies import MCTSPlayer
import go
import coords


BEST_C2_GAME = "B[cd];W[cc];B[dc];W[dd];B[de];W[bd];B[ed];W[cb];B[be];W[ad];B[db];W[ca];B[ab];W[bb];B[ce];W[ac];B[da];W[aa];B[ae]"
COLOR_ARRAY = ' BW'


class BotAnalyzer(object):
    """ """
    def __init__(self, dnn: DualNetwork):
        self.dnn = dnn
        self.pos = go.Position()
        self.player = MCTSPlayer(dnn, resign_threshold=-1)

    def sample_move(self, probs) -> int:
        return np.random.choice(len(probs), 1, p=probs)[0]

    def select_move_no_mcts(self, pos: go.Position):
        """ """
        probs, value = self.dnn.run(pos)

        # explore / exploit balance: higher -> max
        probs_sample = probs ** 3
        probs_sample = probs_sample / probs_sample.sum()
        move = self.sample_move(probs_sample)
        logging.info('value=%.1f, turn %d: best-move=%s (%.2f), chosen=%s (%.2f)', value, pos.n,
                     coords.from_flat(np.argmax(probs)), np.max(probs), coords.from_flat(move), probs[move])
        return coords.flat_to_gtp(move)

    def evaluate_move(self, pos: go.Position, move: tuple, outcome: float):
        """ we typically evaluate a sequence, better to reuse tree
        """
        player = self.player
        player.initialize_game(pos)

        # Must run this once at the start to expand the root node.
        first_node = player.root.select_leaf()
        prob, val = player.network.run(first_node.position)
        first_node.incorporate_results(prob, val, first_node)
        # carry out MCTS searches
        move_bot = player.suggest_move(pos)

        root = player.root
        idx_move = coords.to_flat(move)
        comment = root.describe_less_details(target_move=idx_move)
        return comment

        # q_move = root.child_Q[idx_move]
        # s = '' if np.sign(q_move) == np.sign(outcome) else ' *'
        # if move_bot == move:
        #     print('move #%d: %s %.1f%s' % (1 + pos.n, coords.to_gtp(move), q_move, s))
        # else:
        #     idx_bot_move = coords.to_flat(move_bot)
        #     q_bot = root.child_Q[idx_bot_move]
        #     print('move #%d: mcts best: %s %.1f, \tactual: %s %.1f%s' % (1 + pos.n,
        #           coords.to_gtp(move_bot), q_bot, coords.to_gtp(move), q_move, s))
        #     # print(root.describe(max_children=5))
        # return

    def construct_game_state(self, moves_history: List[str]) -> go.Position:
        """ moves are in gtp format """
        pos = go.Position()
        try:
            for i, move in enumerate(moves_history):
                if move == 'resign':
                    logging.info('%s resigned', pos.to_play)
                    break

                c = coords.from_gtp(move)
                pos = pos.play_move(c)
        except:
            logging.error('error reconstructing %d', pos.n)
        return pos


def probe_along_game(bot: BotAnalyzer, pwcs: Iterable[go.PositionWithContext], outcome: float):
    comments = []

    cur_pos = go.Position()
    for i, pwc in enumerate(pwcs):
        assert cur_pos.n == pwc.position.n
        move = pwc.next_move
        comment = bot.evaluate_move(cur_pos, move, outcome)
        comments.append(comment)

        cur_pos = cur_pos.play_move(move)

    player_moves = cur_pos.recent
    return player_moves, comments


def run_hand_crafted_game():
    model_id = 'model4_epoch2'
    logging.info('loading %s', model_id)
    dnn = DualNetwork(f'{myconf.EXP_HOME}/checkpoints/{model_id}.h5')
    player = BotAnalyzer(dnn)

    sgf_moves_str = BEST_C2_GAME
    sgf_str = sgf_wrapper.make_sgf_from_move_str(sgf_moves_str, 'B+R')
    reader = SGFReader.from_string(sgf_str)
    player_moves, comments = probe_along_game(player, reader.iter_pwcs(), 1.0)


def run_game():
    model_id = 'model12_2'
    logging.info('loading %s', model_id)
    dnn = CoreMLNet(f'{myconf.EXP_HOME}/../9x9-exp2/checkpoints/{model_id}.mlpackage')
    player = BotAnalyzer(dnn)

    sgf_fname = '/Users/hyu/PycharmProjects/dlgo/5x5/selfplay4/sgf/full/133-54927012350.sgf'
    sgf_fname = '/Users/hyu/Downloads/230601-hyu2001-GnuGo.sgf'
    logging.info('Reviewing %s', sgf_fname)
    reader = SGFReader.from_file_compatible(sgf_fname)
    player_moves, comments = probe_along_game(player, reader.iter_pwcs(), reader.result())
    sgf_str = sgf_wrapper.make_sgf(player_moves, reader.result_str(), komi=reader.komi(),
                                   white_name=reader.white_name(),
                                   black_name=reader.black_name(),
                                   game_comment=f'analyzed by: {model_id}',
                                   comments=comments)
    with open(f'/Users/hyu/Downloads/test_annotate.sgf', 'w') as f:
        f.write(sgf_str)
    return


def main(argv):
    run_game()


if __name__ == '__main__':
    abslapp.run(main)
