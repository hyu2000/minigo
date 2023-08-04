import math

from absl import app, flags
import numpy as np
import coords
import go
import mcts

import myconf
from absl import logging
from k2net import DualNetwork
import k2net as dual_net
from sgf_wrapper import SGFReader
from strategies import MCTSPlayer


FLAGS = flags.FLAGS


def test1():
    num_readouts = 400
    dnn = DualNetwork(f'{myconf.EXP_HOME}/checkpoints/model2_epoch_3.h5')
    player = MCTSPlayer(dnn)

    sgf_fname = '/Users/hyu/PycharmProjects/dlgo/9x9/games/Pro/9x9/Minigo/890826.sgf'
    # sgf_fname = '/Users/hyu/PycharmProjects/dlgo/9x9/games/Top50/go9/2015-09-12T04:44:34.331Z_c2aer20lnwa7.sgf'
    reader = SGFReader.from_file_compatible(sgf_fname)
    pos = reader.last_pos()

    player.initialize_game(pos)
    # Must run this once at the start to expand the root node.
    first_node = player.root.select_leaf()
    assert first_node == player.root
    prob, val = dnn.run(first_node.position)
    first_node.incorporate_results(prob, val, first_node)

    for i in range(10):
        # dnn prediction
        move_probs, val_estimate = dnn.run(pos)
        move_dnn = coords.flat_to_gtp(move_probs.argmax())

        # player.initialize_game(pos)
        active = player

        while active.root.N < num_readouts:
            # value output is garbage right now; also not in the expected +/-1 form
            active.tree_search()

        move = active.pick_move()
        logging.info('%d %d: dnn picks %s, val=%.1f, mcts picks %s', pos.n, pos.to_play,
                     move_dnn, val_estimate, coords.to_gtp(move))

        active.play_move(move)
        pos = pos.play_move(move)
        if player.root.is_done():
            player.set_result(player.root.position.result(), was_resign=False)
            break


def test_puzzle_play():
    """ see how my models work on puzzles
    Similar to what KataGo would do
    """
    num_readouts = 400
    model_id = 'model12_2'
    model_fname = f'{myconf.EXP_HOME}/../9x9-exp2/checkpoints/{model_id}.mlpackage'
    dnn = dual_net.load_net(model_fname)
    player = MCTSPlayer(dnn)

    sgf_fname = '/Users/hyu/PycharmProjects/dlgo/9x9/games/Pro/9x9/Minigo/890826.sgf'
    sgf_fname = '/Users/hyu/Downloads/go-puzzle9/Amigo no igo - 詰碁2023 - Life and Death/総合問題４級.sgf'
    reader = SGFReader.from_file_compatible(sgf_fname)
    pos = reader.first_pos()

    player.initialize_game(pos)
    # Must run this once at the start to expand the root node.
    first_node = player.root.select_leaf()
    assert first_node == player.root
    prob, val = dnn.run(first_node.position)
    first_node.incorporate_results(prob, val, first_node)

    for i in range(1):
        # dnn prediction
        move_probs, val_estimate = dnn.run(pos)
        move_dnn = coords.flat_to_gtp(move_probs.argmax())

        # player.initialize_game(pos)
        active = player

        while active.root.N < num_readouts:
            # value output is garbage right now; also not in the expected +/-1 form
            active.tree_search()

        move = active.pick_move()[1]
        logging.info('%d %d: dnn picks %s, val=%.1f, mcts picks %s', pos.n, pos.to_play,
                     move_dnn, val_estimate, coords.to_gtp(move))
        pi = active.root.children_as_pi()
        print(pi[:81].reshape((9, 9)))

        active.play_move(move)
        pos = pos.play_move(move)
        if player.root.is_done():
            player.set_result(player.root.position.result(), was_resign=False)
            break


def test_upper_bound_confidence(self):
    probs = np.array([.02] * (go.N * go.N + 1))
    root = mcts.MCTSNode(go.Position())
    leaf = root.select_leaf()
    assert root == leaf
    leaf.incorporate_results(probs, 0.5, up_to=root)

    # 0.02 are normalized to 1/82
    # self.assertAlmostEqual(root.child_prior[0], 1/82)
    # self.assertAlmostEqual(root.child_prior[1], 1/82)
    puct_policy = lambda n: 2.0 * (math.log((1.0 + n + FLAGS.c_puct_base)
                   / FLAGS.c_puct_base) + FLAGS.c_puct_init) * 1/82
    assert root.N == 1
    # self.assertAlmostEqual(
    #     root.child_U[0], puct_policy(root.N) * math.sqrt(1) / (1 + 0))

    leaf = root.select_leaf()
    assert root != leaf

    # With the first child expanded.
    assert root.N == 1
    # self.assertAlmostEqual(
    #     root.child_U[0], puct_policy(root.N) * math.sqrt(1) / (1 + 0))
    # self.assertAlmostEqual(
    #     root.child_U[1], puct_policy(root.N) * math.sqrt(1) / (1 + 0))

    leaf.add_virtual_loss(up_to=root)
    leaf2 = root.select_leaf()

    assert leaf2 not in {root, leaf}

    leaf.revert_virtual_loss(up_to=root)
    leaf.incorporate_results(probs, 0.3, root)
    leaf2.incorporate_results(probs, 0.3, root)

    # With the 2nd child expanded.
    self.assertEqual(root.N, 3)
    self.assertAlmostEqual(
        root.child_U[0], puct_policy(root.N) * math.sqrt(2) / (1 + 1))
    self.assertAlmostEqual(
        root.child_U[1], puct_policy(root.N) * math.sqrt(2) / (1 + 1))
    self.assertAlmostEqual(
        root.child_U[2], puct_policy(root.N) * math.sqrt(2) / (1 + 0))


def main(argv):
    # test_upper_bound_confidence(argv)
    test_puzzle_play()


if __name__ == '__main__':
    app.run(main)
