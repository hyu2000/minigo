import coords
import k2net as dual_net
import myconf
from sgf_wrapper import SGFReader
from strategies import MCTSPlayer
from absl import logging


def play_puzzle():
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


if __name__ == '__main__':
    play_puzzle()
