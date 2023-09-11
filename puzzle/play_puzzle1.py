import os
from typing import Tuple

import numpy as np

import coords
import go
import k2net as dual_net
import myconf
from puzzle.lnd_puzzle import BBox, LnDPuzzle, rect_mask
from sgf_wrapper import SGFReader
from strategies import MCTSPlayer
from absl import logging, app


class LnDFramer0:
    """ first cut to fill the rest of the board """
    # -----------------------------------------------------------------------
    # methods below are incomplete wip
    # -----------------------------------------------------------------------
    @staticmethod
    def verify_standardize_to_top_right(black_box: BBox, white_box: BBox) -> bool:
        """ """
        if white_box.row0 == 0:
            if white_box.col1 == 8:
                # corner puzzle
                return True
            # edge puzzle
            if white_box.col0 == 0:
                logging.error("puzzle is at the top-left corner, need standardize")
                return False
            return True

        logging.error("puzzle is either not standardized or is in the center (not supported)")
        return False

    @staticmethod
    def easy_oppo_area_case_heuristic1(black_box: BBox, white_box: BBox):
        """ Assuming Black is on the attack.
        white_box is more accurately the contested area; black may have extra stones on the outside for safety

        Easiest way to carve out safe area for the defender: entire rows/cols, or a combination (L-shaped).
        Ideally with 1 extra row separated from white_box.grow(1).
        """
        area_needed = (go.N * go.N - white_box.area()) / 2
        max_gap_from_white_box = max(black_box.nrows() - white_box.nrows(), black_box.ncols() - white_box.ncols())
        white_box_grown = white_box.grow(1)

        # case 1: L-shape is big enough
        area_L_shape = go.N * go.N - white_box_grown.area()
        easy_area_L_shape = go.N * go.N - black_box.area()

        n_fullrows = min(go.N - black_box.nrows(), area_needed // go.N)
        area_to_satisfy = area_needed - n_fullrows * go.N
        if area_to_satisfy == 0:
            return True
        if area_to_satisfy < go.N and go.N - black_box.nrows() - n_fullrows > 0:
            assert n_fullrows >= 2
            # add a partial row on top of the bottom rows. Done
            pass
        else:
            # must allocate from the L-shape
            pass

        if easy_area_L_shape >= area_needed:
            #
            return 1
        elif area_L_shape >= area_needed:
            # now make sure we have enough gap from white box
            gap = max_gap_from_white_box + (area_L_shape - area_needed) / go.N
            if gap >= 3:
                return 1
            else:
                logging.error('not enough gap %s ', black_box, max_gap_from_white_box, area_L_shape, area_needed, gap)
                return
        logging.error('L-shape not big enough:', black_box, area_L_shape, area_needed)
        return 0

    @staticmethod
    def easy_oppo_area_case2(black_box: BBox, white_box: BBox):
        """ Second try over the above routine (heuristic-heavy)

        Start off existing puzzle area (both sides combined, or inside white_box.grow(1)), grow a couple steps to avoid
        changing the puzzle dynamics (the growth area belong to black). If the rest is big
        enough, we then carve out a portion for white.
        """
        pass


    @staticmethod
    def find_oppo_area(black_box: BBox, white_box: BBox) -> np.array:
        """ Assuming Black is on the attack.
        :return: board with white area marked
        """
        LnDFramer0.verify_standardize_to_top_right(black_box, white_box)


    @classmethod
    def fill_rest_of_board(cls, board: np.array):
        """
        Away from the puzzle area, fill an area with defender stones, about half the size.
        Surround it with one layer of attacker stones.
        """
        black_box, white_box, attack_side = LnDPuzzle.solve_boundary(board)
        # todo standardize: if a corner puzzle, make it top-right; otherwise make it near top

        assert attack_side != 0
        # see if away from puzzle by at least delta, whether we have enough space
        if attack_side == go.BLACK:
            oppo_area = cls.find_oppo_area(black_box, white_box)
        else:
            oppo_area = cls.find_oppo_area(white_box, black_box)

        # add border to oppo_area



def mask_to_policy(board: np.array) -> np.array:
    """ return a flattened policy, always allow pass """
    return np.append(board.flatten(), 1)


def test_fill_board():
    sgf_fname = '/Users/hyu/Downloads/go-puzzle9/Amigo no igo - 詰碁2023 - Life and Death/総合問題5級(9).sgf'
    reader = SGFReader.from_file_compatible(sgf_fname)
    pos = reader.first_pos()
    black_box, white_box, attack_side = LnDPuzzle.solve_boundary(pos.board)
    # assert attack_side == go.BLACK

    LnDFramer0.fill_rest_of_board(pos.board)


def play_puzzle():
    """ see how my models work on puzzles (after filling rest of the board)
    Similar to what KataGo would do

    Seems raw policy is pretty flat --> MCTS is doing most of the work. Masked MCTS?
    """
    num_readouts = 400
    model_id = 'model12_3'
    model_fname = f'{myconf.EXP_HOME}/../9x9-exp2/checkpoints/{model_id}.mlpackage'
    dnn_underlyer = dual_net.load_net(model_fname)

    sgf_fname = '/Users/hyu/Downloads/go-puzzle9/Amigo no igo - 詰碁2023 - Life and Death/総合問題5級.sgf'
    # sgf_fname = '/Users/hyu/Downloads/test_framer4(4).sgf'
    reader = SGFReader.from_file_compatible(sgf_fname)
    pos = reader.first_pos()
    black_box, white_box, attack_side = LnDPuzzle.solve_boundary(pos.board)
    bbox_defender = white_box if attack_side == go.BLACK else black_box
    policy_mask = mask_to_policy(rect_mask(bbox_defender.grow(1)))
    print(f'Solving %s, mask size=%d:' % (os.path.basename(sgf_fname), np.sum(policy_mask)))
    print(policy_mask[:81].reshape((9, 9)))

    # dnn = dual_net.MaskedNet(dnn_underlyer, policy_mask)
    dnn = dnn_underlyer
    player = MCTSPlayer(dnn)

    player.initialize_game(pos, focus_area=policy_mask)

    for i in range(10):
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
        print(pi[:81].reshape((9, 9)), active.root.Q)
        # pi *= policy_mask
        # move = coords.from_flat(pi.argmax())
        logging.info('%d %s: dnn picks %s, val=%.1f, global %s ', pos.n, go.color_str(pos.to_play),
                     move_dnn, val_estimate, coords.to_gtp(move_global))

        active.play_move(move_global)
        pos = pos.play_move(move_global)
        if player.root.is_done():
            player.set_result(player.root.position.result(), was_resign=False)
            break


def main(argv):
    play_puzzle()


if __name__ == '__main__':
    app.run(main)

