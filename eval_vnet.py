""" evaluate value net on labeled games

Motivation: in self-play, it might be better to score end-position w/ value net. So we want to monitor
how value net performs over time.

For simplicity, just look at win/loss accuracy (since many games has no margin result)
- accuracy at different game stages: every 10 moves?
- also run Tromp scoring for comparison
- win/loss accuracy after MCTS search on final position
"""
from strategies import MCTSPlayer

"""  Result:
Pro:
I0525 14:10:27.492900 4547440128 eval_vnet.py:63] Total 462 games, final acc: Tromp 0.51, vnet 0.62
 position.n    0      10      20      30      40      50      60      70      80
	Count [' 462', ' 462', ' 461', ' 416', ' 302', ' 159', '  53', '  15', '   5']
	Tromp ['0.53', '0.53', '0.53', '0.52', '0.53', '0.53', '0.47', '0.47', '0.40']
	vnet  ['0.53', '0.52', '0.55', '0.59', '0.61', '0.65', '0.62', '0.60', '0.60']

Top50 (random 500)
I0525 17:06:07.369480 4763217408 eval_vnet.py:75] Total 500 games, final acc: Tromp 0.62, vnet 0.73
I0525 17:06:07.369722 4763217408 eval_vnet.py:79] By turns:
	Count [' 500', ' 500', ' 500', ' 497', ' 444', ' 359', ' 205', '  79', '  20', '   6', '   1', '   1', '   1', '   1', '   1']
	Tromp ['0.53', '0.53', '0.53', '0.53', '0.51', '0.56', '0.52', '0.63', '0.75', '1.00', '1.00', '1.00', '1.00', '1.00', '0.00']
	vnet  ['0.53', '0.52', '0.55', '0.61', '0.59', '0.65', '0.65', '0.66', '0.85', '1.00', '1.00', '1.00', '1.00', '1.00', '1.00']

NNGS (random 500):
I0525 17:00:48.143626 4747976192 eval_vnet.py:69] Total 500 games, final acc: Tromp 0.60, vnet 0.81
I0525 17:00:48.143833 4747976192 eval_vnet.py:73] By turns:
	Count [' 500', ' 500', ' 495', ' 460', ' 369', ' 231', ' 130', '  50', '  18', '   7', '   3', '   3', '   2', '   1', '   1', '   1']
	Tromp ['0.60', '0.60', '0.59', '0.56', '0.57', '0.60', '0.65', '0.72', '0.83', '0.57', '0.67', '0.67', '1.00', '1.00', '1.00', '1.00']
	vnet  ['0.41', '0.59', '0.69', '0.73', '0.79', '0.81', '0.80', '0.72', '0.89', '0.71', '0.67', '0.67', '1.00', '1.00', '1.00', '1.00']
NNGS (first 500):
I0602 14:42:21.079003 4412198400 eval_vnet.py:91] Total 500 games, final acc: Tromp 0.53, vnet 0.79, mcts 0.00
I0602 14:42:21.079193 4412198400 eval_vnet.py:96] By turns:
	Count [' 500', ' 500', ' 499', ' 472', ' 364', ' 220', '  95', '  19', '   3', '   1']
	Tromp ['0.58', '0.57', '0.54', '0.56', '0.55', '0.50', '0.60', '0.63', '1.00', '1.00']
	vnet  ['0.43', '0.64', '0.69', '0.76', '0.79', '0.79', '0.82', '0.63', '1.00', '1.00']

epoch2
I0602 14:40:39.793861 4696673792 eval_vnet.py:83] Total 500 games, final acc: Tromp 0.53, vnet 0.78, mcts 0.00
I0602 14:40:39.794044 4696673792 eval_vnet.py:88] By turns:
	Count [' 500', ' 500', ' 499', ' 472', ' 364', ' 220', '  95', '  19', '   3', '   1']
	Tromp ['0.58', '0.57', '0.54', '0.56', '0.55', '0.50', '0.60', '0.63', '1.00', '1.00']
	vnet  ['0.43', '0.51', '0.65', '0.73', '0.80', '0.77', '0.80', '0.63', '1.00', '1.00']	
"""
from typing import List

import numpy as np
import myconf
import go
from sgf_wrapper import SGFReader
from tar_dataset import GameStore
import k2net as dual_net
from absl import logging, app, flags

flags.DEFINE_string('tar_dir', None, 'Where to find TarDataSets.')

FLAGS = flags.FLAGS
NUM_TURNS_COUNTED = 40


class ScoreStats(object):

    def __init__(self):
        self.num_games = 0
        self.tromp_final = 0
        self.vnet_final = 0
        self.mcts_final = 0
        self.tromp_counters = np.zeros(NUM_TURNS_COUNTED)
        self.vnet_counters  = np.zeros(NUM_TURNS_COUNTED)
        self.total_counters = np.zeros(NUM_TURNS_COUNTED)

    @staticmethod
    def _pad_np_1darray(arr: np.array, n: int, const_val=0):
        pad_right = n - len(arr)
        assert len(arr.shape) == 1 and pad_right >= 0
        return np.pad(arr, [(0, pad_right)], constant_values=const_val)

    def add_game(self, win_loss: int, mcts_final: int, tromp_final: int, vnet_final: int,
                 tromp_binary_by_turns: List[int], vnet_binary_by_turns: List[int]):
        assert len(tromp_binary_by_turns) <= NUM_TURNS_COUNTED
        self.num_games += 1
        num_steps = len(tromp_binary_by_turns)

        self.tromp_counters += self._pad_np_1darray(np.array(tromp_binary_by_turns) == win_loss, NUM_TURNS_COUNTED)
        self.vnet_counters  += self._pad_np_1darray(np.array( vnet_binary_by_turns) == win_loss, NUM_TURNS_COUNTED)
        self.total_counters += self._pad_np_1darray(np.ones(num_steps), NUM_TURNS_COUNTED)
        self.tromp_final += tromp_final == win_loss
        self.vnet_final  +=  vnet_final == win_loss
        self.mcts_final  +=  mcts_final == win_loss

    def report(self):
        logging.info('Total %d games, final acc: Tromp %.2f, vnet %.2f, mcts %.2f',
                     self.num_games, self.tromp_final / self.num_games,
                     self.vnet_final / self.num_games, self.mcts_final / self.num_games)
        max_steps = (self.total_counters > 0).sum()
        total_counts = self.total_counters[:max_steps]
        logging.info('By turns:\n\tCount %s\n\tTromp %s\n\tvnet  %s\n',
                     ['%4d' % x for x in total_counts],
                     ['%.2f' % x for x in self.tromp_counters[:max_steps] / total_counts],
                     ['%.2f' % x for x in self.vnet_counters[ :max_steps] / total_counts])


def run_tree_search(network, init_position):
    NUM_READOUTS = 200

    player = MCTSPlayer(network, resign_threshold=-1)  # no resign
    player.initialize_game(position=init_position)

    # Must run this once at the start to expand the root node.
    first_node = player.root.select_leaf()
    prob, val = network.run(first_node.position)
    first_node.incorporate_results(prob, val, first_node)

    while player.root.N < NUM_READOUTS:
        player.tree_search()

    q = player.root.Q
    return np.sign(q - 0.5)


def run_game(network, game_id, reader: SGFReader, stats: ScoreStats):
    win_loss = reader.result()
    if win_loss == 0:
        # no RE
        return

    komi = reader.komi()

    pos_to_eval = []  # type: List[go.Position]
    for pwc in reader.iter_pwcs():
        position = pwc.position
        if position.n % 10 == 0:
            pos_to_eval.append(position)
    # last position
    position = pwc.position
    # position = pos_to_eval[2]
    pos_to_eval.append(position)
    mcts_binary = 0  # run_tree_search(network, position)

    tromp_scores = [p.score() for p in pos_to_eval]
    tromp_binary = [np.sign(s) for s in tromp_scores]
    _, vnet_scores = network.run_many(pos_to_eval)
    vnet_binary = [np.sign(s - komi) for s in vnet_scores]
    assert len(tromp_scores) == len(vnet_scores)

    tromp_final = tromp_binary.pop()
    vnet_final  = vnet_binary.pop()
    stats.add_game(win_loss, mcts_binary, tromp_final, vnet_final, tromp_binary, vnet_binary)


def run_games(start_idx=0):
    """
    """
    # store = GameStore(data_dir=FLAGS.tar_dir)
    store = GameStore(data_dir=f'{myconf.DATA_DIR}')
    game_iter = store.game_iter([store.ds_nngs], filter_game=True, shuffle=False)

    # model_file = FLAGS.load_file
    model_file = f'{myconf.MODELS_DIR}/endgame_epoch_2.h5'
    network = dual_net.DualNetwork(model_file)
    stats = ScoreStats()
    for game_idx, (game_id, reader) in enumerate(game_iter):
        if game_idx < start_idx:
            continue

        run_game(network, game_id, reader, stats)
        if stats.num_games >= 500: break
        if stats.num_games % 50 == 0:
            print('.', end='')
    print()
    stats.report()


def main(argv):
    run_games(start_idx=0)


if __name__ == '__main__':
    app.run(main)
