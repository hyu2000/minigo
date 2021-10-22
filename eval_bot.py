""" measure strength of a bot / DNN, to track training progress

"""

from typing import List

import pandas as pd
from absl import logging
from absl import app as abslapp
import numpy as np
from k2net import DualNetwork
import myconf
from sgf_wrapper import replay_sgf, SGFReader
from strategies import MCTSPlayer
import go
import k2net as dual_net
from tar_dataset import GameStore

from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='cpu')  # Available options are 'cpu', 'gpu', and ‘any'.

# evaluate every 5 moves
START_POS = 1
POS_STEP = 5
MAX_NUM_STEPS = 40


class ScoreStats(object):
    def __init__(self):
        self.num_games = np.zeros(MAX_NUM_STEPS)
        self.num_correct = np.zeros(MAX_NUM_STEPS)
        self.max_steps = 0

    def add(self, win_loss, scores: np.array):
        num_steps = len(scores)
        scores = np.sign(scores)
        self.num_games[:num_steps] += np.ones(len(scores))
        self.num_correct[:num_steps] += scores == win_loss
        self.max_steps = max(self.max_steps, len(scores))

    def summary(self):
        print('ScoreStats:')
        df = pd.DataFrame({
                'move': range(START_POS, 200, POS_STEP)[:self.max_steps],
                'count': self.num_games[:self.max_steps].astype(int),
                'accu': self.num_correct[:self.max_steps] / self.num_games[:self.max_steps]
               })
        df = df.set_index('move')
        return df


class BotAnalyzer(object):
    """ """
    def __init__(self, dnn: DualNetwork):
        self.dnn = dnn
        self.pos = go.Position()
        self.player = MCTSPlayer(dnn, resign_threshold=-1)

    def predict(self, pos: go.Position):
        """ no mcts right now, just vnet """
        prob, val = self.dnn.run(pos)
        return val


def run_game(dnn: DualNetwork, game_id, reader: SGFReader, stats: ScoreStats):
    win_loss = reader.result()
    if win_loss == 0:  # no RE
        return

    pos_to_eval = []  # type: List[go.Position]
    for pwc in reader.iter_pwcs():
        position = pwc.position
        if position.n % POS_STEP == START_POS:
            pos_to_eval.append(position)

    if len(pos_to_eval) == 0:
        print(f'empty pos for {game_id}')
        return

    _, vnet_scores = dnn.run_many(pos_to_eval)
    stats.add(win_loss, vnet_scores)


def run_games():
    """ """
    store = GameStore(data_dir=f'{myconf.DATA_DIR}')

    dfdict = {}
    for model_id in [f'model{i}_epoch2' for i in range(8)]:
        model_file = f'{myconf.MODELS_DIR}/{model_id}.h5'
        dnn = dual_net.DualNetwork(model_file)

        stats = ScoreStats()

        game_iter = store.game_iter([store.ds_pro], filter_game=True, shuffle=False)
        for game_idx, (game_id, reader) in enumerate(game_iter):
            run_game(dnn, game_id, reader, stats)

        df = stats.summary()
        dfdict[model_id] = df

    df0 = df
    df = pd.DataFrame({k: df['accu'] for k, df in dfdict.items()})
    # df.index = df0.index
    df['count'] = df0['count']
    df.reset_index().to_csv('/tmp/eval_vnet.csv')
    print(df)


def main(argv):
    run_games()


if __name__ == '__main__':
    # abslapp.run(main)
    main({})

