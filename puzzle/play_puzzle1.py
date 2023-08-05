import glob
import os
from collections import namedtuple
from typing import Tuple

import numpy as np

import coords
import go
import k2net as dual_net
import myconf
from sgf_wrapper import SGFReader
from strategies import MCTSPlayer
from absl import logging, app


class BBox(namedtuple('BBox', ['row0', 'col0', 'row1', 'col1'])):
    """ top-left, bottom-right in minigo coord style """
    def grow(self, delta: int):
        return BBox(max(0, self.row0 - delta), max(0, self.col0 - delta),
                    min(go.N-1, self.row1 + delta), min(go.N-1, self.col1 + delta))


class LnDPuzzle:
    """ helper to figure out basics of an LnD puzzle """
    # tolerance to the edge of the board
    SNAP_EDGE_THRESH = 1

    @staticmethod
    def snap(k: int, to: int):
        return to if abs(k - to) <= LnDPuzzle.SNAP_EDGE_THRESH else k

    @classmethod
    def find_bound_for(cls, board: np.array, side: int) -> BBox:
        """
        """
        stone_coords = np.argwhere(board == side)
        imin, jmin = stone_coords.min(axis=0)
        imax, jmax = stone_coords.max(axis=0)
        imin, jmin = cls.snap(imin, 0), cls.snap(jmin, 0)
        imax, jmax = cls.snap(imax, go.N), cls.snap(jmax, go.N-1)
        return BBox(imin, jmin, imax, jmax)

    @staticmethod
    def enclosing_box(a: BBox, b: BBox) -> BBox:
        return BBox(min(a.row0, b.row0), min(a.col0, b.col0), max(a.row1, b.row1), max(a.col1, b.col1))

    @classmethod
    def solve_boundary(cls, board: np.array) -> tuple[BBox, BBox, int]:
        """ given a puzzle board, figure out who's surrounding, corner/center area (for the surrounded)

        Attacker is more on the peripheral, or has more stones there (maybe by a weighted sum)
        """
        black_boundary = cls.find_bound_for(board, go.BLACK)
        white_boundary = cls.find_bound_for(board, go.WHITE)
        if black_boundary == white_boundary:
            print('white_boundary == black_boundary: cannot figure out attacker/defender')
            return black_boundary, white_boundary, 0
        enclosure = cls.enclosing_box(black_boundary, white_boundary)
        attack_side = 0
        if enclosure == black_boundary:
            attack_side = go.BLACK
        elif enclosure == white_boundary:
            attack_side = go.WHITE
        if attack_side == 0:
            print('cannot figure out which side is bigger')
            return black_boundary, white_boundary, attack_side
        return black_boundary, white_boundary, attack_side


def rect_mask(i0, i1, j0, j1) -> np.array:
    """ all inclusive """
    mask = np.zeros((go.N, go.N), dtype=np.int8)
    mask[i0:i1+1, j0:j1+1] = 1
    return mask


def mask_to_policy(board: np.array) -> np.array:
    """ return a flattened policy, always allow pass """
    return np.append(board.flatten(), 1)


def test_mask():
    print()
    mask0 = rect_mask(0, 3, 3, 8)
    print(mask0)


def test_puzzle_helper():
    sgf_fname = '/Users/hyu/Downloads/go-puzzle9/Amigo no igo - 詰碁2023 - Life and Death/総合問題４級.sgf'
    sgf_fname = '/Users/hyu/Downloads/go-puzzle9/Amigo no igo - 詰碁2023 - Life and Death/総合問題９級.sgf'
    sgf_fname = '/Users/hyu/Downloads/go-puzzle9/Amigo no igo - 詰碁2023 - Life and Death/総合問題４級(３).sgf'
    reader = SGFReader.from_file_compatible(sgf_fname)
    pos = reader.first_pos()
    black_box, white_box, attack_side = LnDPuzzle.solve_boundary(pos.board)
    assert attack_side == go.BLACK
    print("black", black_box)
    print("white", white_box)
    white_enlarged = white_box.grow(1)
    # white policy area: 0..3, 3..8
    assert white_enlarged.col0 == 3 and white_enlarged.row1 == 3


def test_puzzle_helper_bulk():
    sgf_dir = '/Users/hyu/Downloads/go-puzzle9/Amigo no igo - 詰碁2023 - Life and Death'
    sgf_fnames = glob.glob(f'{sgf_dir}/総合問題５級*.sgf')
    print('found', len(sgf_fnames))
    for sgf_fname in sgf_fnames:
        basename = os.path.split(sgf_fname)[-1]
        print(f'Processing {basename}')
        reader = SGFReader.from_file_compatible(sgf_fname)
        pos = reader.first_pos()
        black_box, white_box, attack_side = LnDPuzzle.solve_boundary(pos.board)
        if attack_side == 0:
            continue
        print(go.color_str(attack_side), black_box, white_box)
        # assert attack_side == go.BLACK


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
    white_mask = mask_to_policy(rect_mask(0, 3, 3, 8))

    player.initialize_game(pos)
    # Must run this once at the start to expand the root node.
    first_node = player.root.select_leaf()
    assert first_node == player.root
    prob, val = dnn.run(first_node.position)
    first_node.incorporate_results(prob, val, first_node)

    for i in range(6):
        # dnn prediction
        move_probs, val_estimate = dnn.run(pos)
        move_dnn = coords.flat_to_gtp(move_probs.argmax())

        # player.initialize_game(pos)
        active = player

        while active.root.N < num_readouts:
            # value output is garbage right now; also not in the expected +/-1 form
            active.tree_search()

        move_global = active.pick_move()[1]
        pi = active.root.children_as_pi()
        print(pi[:81].reshape((9, 9)))
        pi *= white_mask
        move = coords.from_flat(pi.argmax())
        logging.info('%d %s: dnn picks %s, val=%.1f, global %s -> masked %s', pos.n, go.color_str(pos.to_play),
                     move_dnn, val_estimate, coords.to_gtp(move_global), coords.to_gtp(move))

        active.play_move(move)
        pos = pos.play_move(move)
        if player.root.is_done():
            player.set_result(player.root.position.result(), was_resign=False)
            break


def main(argv):
    play_puzzle()


if __name__ == '__main__':
    app.run(main)

