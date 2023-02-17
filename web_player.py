from typing import List

from absl import logging
from absl import app as abslapp
import numpy as np
import go
import coords
from k2net import DualNetwork
import myconf
from strategies import MCTSPlayer


class WebPlayer(object):
    def __init__(self, dnn: DualNetwork):
        self.dnn = dnn
        self.pos = go.Position()
        self.player = MCTSPlayer(dnn, resign_threshold=-1)

    def sample_move(self, probs) -> int:
        return np.random.choice(len(probs), 1, p=probs)[0]

    def select_move_no_mcts(self, moves_history):
        """ """
        pos = self.construct_game_state(moves_history)
        probs, value = self.dnn.run(pos)

        # explore / exploit balance: higher -> max
        probs_sample = probs ** 3
        probs_sample = probs_sample / probs_sample.sum()
        move = self.sample_move(probs_sample)
        logging.info('value=%.1f, turn %d: best-move=%s (%.2f), chosen=%s (%.2f)', value, pos.n,
                     coords.from_flat(np.argmax(probs)), np.max(probs), coords.from_flat(move), probs[move])
        return coords.flat_to_gtp(move)

    def select_move(self, moves_history):
        """
        quick job: we would benefit from reuse the tree...
        """
        pos = self.construct_game_state(moves_history)
        player = self.player
        player.initialize_game(pos)

        # Must run this once at the start to expand the root node.
        first_node = player.root.select_leaf()
        prob, val = player.network.run(first_node.position)
        first_node.incorporate_results(prob, val, first_node)

        move = player.suggest_move(pos)
        # why is root.Q np.array for minigo model, float for converted jax model
        root_winrate = '%s' % player.root.Q
        child_q = player.root.child_Q[coords.to_flat(move)]
        gtp_move = coords.to_gtp(move)
        info_dict = {'winrate': root_winrate, 'move': gtp_move}
        logging.info('root winrate=%s, mcts chose: %s, child_Q=%.2f', info_dict['winrate'], gtp_move, child_q)
        return gtp_move, info_dict

    def construct_game_state(self, moves_history: List[str]):
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


def test_player():
    dnn = DualNetwork(f'{myconf.EXP_HOME}/checkpoints/model3_epoch_5.h5')
    player = WebPlayer(dnn)
    moves_history = []
    for i in range(5):
        move, info = player.select_move(moves_history)
        moves_history.append(move)


def main(argv):
    test_player()


if __name__ == '__main__':
    abslapp.run(main)
