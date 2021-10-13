"""
given a game, annotate what bot will do, same way a human reviews a game
"""
from typing import List

from absl import logging
from absl import app as abslapp
import numpy as np
import go
import coords
from k2net import DualNetwork
import myconf
from sgf_wrapper import replay_sgf
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

        move_bot = player.suggest_move(pos)

        root = player.root
        idx_move = coords.to_flat(move)
        q_move = root.child_Q[idx_move]
        s = '' if np.sign(q_move) == np.sign(outcome) else ' *'
        if move_bot == move:
            print('move #%d: %s %.1f%s' % (1 + pos.n, coords.to_gtp(move), root.child_Q[idx_move], s))
        else:
            idx_bot_move = coords.to_flat(move_bot)
            q_bot = root.child_Q[idx_bot_move]
            print('move #%d: mcts best: %s %.1f, \tactual: %s %.1f%s' % (1 + pos.n,
                  coords.to_gtp(move_bot), q_bot, coords.to_gtp(move), q_move, s))
            # print(root.describe(max_children=5))
        return

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


def probe_along_game(bot: BotAnalyzer, sgf_moves_str, outcome: float):
    moves = sgf_moves_str.split(';')
    cur_pos = go.Position()

    for i, sgf_move in enumerate(moves):
        color, sgf_coords = sgf_move[0], sgf_move[2:4]
        assert color == COLOR_ARRAY[cur_pos.to_play]
        move = coords.from_sgf(sgf_coords)
        bot.evaluate_move(cur_pos, move, outcome)

        cur_pos = cur_pos.play_move(move)

    return cur_pos


def probe_along_sgf(bot: BotAnalyzer, sgf_fname: str):
    with open(sgf_fname) as f:
        for position_w_context in replay_sgf(f.read()):
            pos = position_w_context.position
            result = position_w_context.result
            if pos.n > 0:
                move = pos.recent[-1]
                # print(cur_pos.n, coords.to_gtp(move.move))
                bot.evaluate_move(cur_pos, move.move, result)
            cur_pos = pos

        return cur_pos


def run_game():
    model_id = 'model13_epoch2'
    logging.info('loading %s', model_id)
    dnn = DualNetwork(f'{myconf.EXP_HOME}/checkpoints/{model_id}.h5')
    player = BotAnalyzer(dnn)
    # probe_along_game(player, BEST_C2_GAME, 1.0)
    sgf_fname = '/Users/hyu/PycharmProjects/dlgo/5x5/logs/selfplay7-sgfs/0-1633047668.sgf'
    logging.info('Reviewing %s', sgf_fname)
    probe_along_sgf(player, sgf_fname)


def main(argv):
    run_game()


if __name__ == '__main__':
    abslapp.run(main)
