from typing import List

from absl import logging
import numpy as np
import go
import coords
from k2net import DualNetwork
import myconf


class WebPlayer(object):
    def __init__(self, dnn: DualNetwork):
        self.dnn = dnn
        self.pos = go.Position()

    def sample_move(self, probs) -> int:
        return np.random.choice(len(probs), 1, p=probs)[0]

    def select_move(self, moves_history):
        # entry point
        pos = self.construct_game_state(moves_history)
        probs, value = self.dnn.run(pos)

        # explore / exploit balance: higher -> max
        probs_sample = probs ** 3
        probs_sample = probs_sample / probs_sample.sum()
        move = self.sample_move(probs_sample)
        logging.info('value=%.1f, turn %d: best-move=%s (%.2f), chosen=%s (%.2f)', value, pos.n,
                     coords.from_flat(np.argmax(probs)), np.max(probs), coords.from_flat(move), probs[move])
        return coords.to_gtp(coords.from_flat(move))

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
        move = player.select_move(moves_history)
        moves_history.append(move)


if __name__ == '__main__':
    test_player()
