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

    def evaluate_move(self, pos: go.Position, move):
        """ we typically evaluate a sequence, better to reuse tree
        """
        player = self.player
        player.initialize_game(pos)

        # Must run this once at the start to expand the root node.
        first_node = player.root.select_leaf()
        prob, val = player.network.run(first_node.position)
        first_node.incorporate_results(prob, val, first_node)

        move_bot = player.suggest_move(pos)
        logging.info('move #%d, mcts chose: %s, got %s', pos.n, coords.to_gtp(move_bot), coords.to_gtp(move))
        print(player.root.describe(max_children=5))
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


def probe_along_game(bot: BotAnalyzer, sgf_moves_str):
    moves = sgf_moves_str.split(';')
    cur_pos = go.Position()

    for i, sgf_move in enumerate(moves):
        color, sgf_coords = sgf_move[0], sgf_move[2:4]
        assert color == COLOR_ARRAY[cur_pos.to_play]
        move = coords.from_sgf(sgf_coords)
        bot.evaluate_move(cur_pos, move)

        cur_pos = cur_pos.play_move(move)
    print(cur_pos.n)
    return cur_pos


def run_game():
    dnn = DualNetwork(f'{myconf.EXP_HOME}/checkpoints/model13_epoch2.h5')
    player = BotAnalyzer(dnn)
    probe_along_game(player, BEST_C2_GAME)


def main(argv):
    run_game()


if __name__ == '__main__':
    abslapp.run(main)
