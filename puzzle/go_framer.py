""" utility to fill a board around a Tsumego

Basically we want to fill the rest of the board, so that the only focus is on the Tsumego.

We don't need to be super-exact, just enough to discourage a bot to play outside Tsumego.
"""
import itertools
import logging
from typing import Tuple

import numpy as np

import myconf
import coords
import go
from puzzle.lnd_puzzle import LnDPuzzle
from sgf_wrapper import SGFReader, add_init_stones_file


def find_chainlike(board: np.ndarray, c: Tuple, include_diagonal=True) -> Tuple[set, set]:
    """ generalizing from go.find_reached(), which is equivalent include_diagonal=False.
    boundary is similarly affected: diagonal spot will be included by default. The other change is
    only empty spots are included here.
    """
    color = board[c]
    chain = {c}
    boundary = set()
    frontier = [c]
    while frontier:
        current = frontier.pop()
        chain.add(current)
        neighbors = go.NEIGHBORS[current]
        if include_diagonal:
            neighbors = itertools.chain(go.NEIGHBORS[current], go.DIAGONALS[current])
        for n in neighbors:
            if board[n] == color:
                if n not in chain:
                    frontier.append(n)
            else:
                if board[n] == go.EMPTY:
                    boundary.add(n)
    return chain, boundary


def grow(chain: set):
    """ conv """


class Framer:
    def __init__(self, sgf_fname):
        self.sgf_fname = sgf_fname

    @staticmethod
    def surround(board: np.array, coord: tuple, opposite_stone=True):
        """ add a one-stone wide fringe around area(*) identified by coord.
        *) If two chains are touching each other, they should be considered the same area.
        conv
        """
        chain, boundary = find_chainlike(board, coord)
        color = board[coord]
        if opposite_stone:
            color = -color
        for p in boundary:
            # check p is not taken?
            board[p] = color
        return board

    @staticmethod
    def grow_to(board: np.array, area_left: int) -> Tuple:
        """ Grow puzzle area layer by layer, until the size of the rest of the board is area_left.

        Without loss of generality, assume black surrounds white in the puzzle. The idea is to grow
        the black area outward, so that we don't disturb the puzzle itself; add a black border at
        the peripheral, next to a white border which surrounds the empty white space that is of size area_left.

        Returns the final boundaries, both black and white.

        Comment:
        1. puzzle area is dense. The first (or several) iteration typically fills it
        2. in general, when we grow two iterations, black layer should have enough eye-space
        """
        board_size = np.prod(board.shape)
        board = abs(board)  # make it all black
        coord_init = tuple(np.argwhere(board)[0])

        MAX_ITER = 5
        for i in range(MAX_ITER):
            chain, boundary = find_chainlike(board, coord_init)
            area_before = np.sum(board)
            area_after = area_before + len(boundary)
            if area_after + area_left > board_size:
                break
            for c in boundary:
                board[c] = go.BLACK

        area_before = np.sum(board)
        can_grow = board_size - area_left - area_before
        print(f'grow_to({area_left}): {i+1} iter: needs {can_grow} blacks')
        assert len(boundary) >= can_grow
        for c in sorted(boundary)[:can_grow]:
            board[c] = go.BLACK

        # simple frame the white boundary
        chain, white_boundary = find_chainlike(board, coord_init)
        # derive final black boundary
        board = board - 1  # flip
        coord_white = tuple(np.argwhere(board)[0])
        chain, black_boundary = find_chainlike(board, coord_white)

        return black_boundary, white_boundary

    def frame1(self, out_sgf_fname):
        reader = SGFReader.from_file_compatible(self.sgf_fname)
        pos = reader.first_pos()
        board = pos.board

        # figure out who encircles whom
        black_bbox, white_bbox, attack_side = LnDPuzzle.solve_boundary(board)
        contested_area = LnDPuzzle.contested_area(board, white_bbox if attack_side == go.BLACK else black_bbox, attack_side)
        corner_size = np.sum(contested_area)
        defender_area_other = (np.prod(board.shape) - corner_size) // 2
        logging.info(f'attacker %s, corner size={corner_size}, area_left={defender_area_other}', go.color_str(attack_side))
        attacker_boundary, defender_boundary = Framer.grow_to(board, defender_area_other)

        if attack_side == go.BLACK:
            add_init_stones_file(self.sgf_fname, attacker_boundary, defender_boundary, out_sgf_fname)
        else:
            add_init_stones_file(self.sgf_fname, defender_boundary, attacker_boundary, out_sgf_fname)


def test_chainlike():
    sgf_fname = '/Users/hyu/Downloads/go-puzzle9/Amigo no igo - 詰碁2023 - Life and Death/総合問題4級(3).sgf'
    reader = SGFReader.from_file_compatible(sgf_fname)
    pos = reader.first_pos()
    board = pos.board
    white_coord = coords.from_gtp('F9')
    chain, boundary = find_chainlike(board, white_coord, include_diagonal=False)
    assert len(chain) == 4
    print(len(boundary))
    chain, boundary = find_chainlike(board, white_coord)
    assert len(chain) == 7
    # white boundary reflected nicely the surrounded area in this case
    print('boundary:', ' '.join(sorted([coords.to_gtp(c) for c in boundary])))
    assert len(chain) + len(boundary) == 16

    # boundary is also affected
    black_coord = coords.from_gtp('E7')
    chain, boundary = find_chainlike(board, black_coord)
    assert len(chain) == 10
    print('boundary:', ' '.join(sorted([coords.to_gtp(c) for c in boundary])))


def test_grow_puzzle_margin():
    """ take a puzzle, grow a couple rounds, see if the outer boundary makes sense """
    sgf_fname = '/Users/hyu/Downloads/go-puzzle9/Amigo no igo - 詰碁2023 - Life and Death/総合問題4級(4).sgf'
    reader = SGFReader.from_file_compatible(sgf_fname)
    pos = reader.first_pos()
    board = pos.board

    board = abs(board)  # make it all black
    coord_init = coords.from_gtp('D7')
    coord_init = tuple(np.argwhere(board)[0])  # any stone in the original puzzle
    for i in range(2):
        chain, boundary = find_chainlike(board, coord_init)
        for c in boundary:
            board[c] = go.BLACK

    boundary_black = boundary
    # just find boundary in the end
    chain, boundary = find_chainlike(board, coord_init)
    print('boundary:', ' '.join(sorted([coords.to_gtp(c) for c in boundary])))

    out_fname = '/Users/hyu/Downloads/test_framer.sgf'
    add_init_stones_file(sgf_fname, boundary_black, boundary, out_fname)


def test_grow_to():
    """ grow a puzzle a few rounds, until just a certain space is left """
    sgf_fname = '/Users/hyu/Downloads/go-puzzle9/Amigo no igo - 詰碁2023 - Life and Death/総合問題4級(3).sgf'
    reader = SGFReader.from_file_compatible(sgf_fname)
    pos = reader.first_pos()
    board = pos.board
    white_corner_size = 16
    white_area_other = (np.prod(board.shape) - white_corner_size) // 2
    black_boundary, white_boundary = Framer.grow_to(board, white_area_other)
    out_fname = '/Users/hyu/Downloads/test_framer.sgf'
    add_init_stones_file(sgf_fname, black_boundary, white_boundary, out_fname)


def test_frame1():
    sgf_fname = '/Users/hyu/Downloads/go-puzzle9/Amigo no igo - 詰碁2023 - Life and Death/総合問題4級(4).sgf'
    framer = Framer(sgf_fname)
    out_fname = '/Users/hyu/Downloads/test_framer4(4).sgf'
    framer.frame1(out_fname)


def test_surround():
    sgf_fname = '/Users/hyu/Downloads/go-puzzle9/Amigo no igo - 詰碁2023 - Life and Death/総合問題4級(3).sgf'
    reader = SGFReader.from_file_compatible(sgf_fname)
    pos = reader.first_pos()
    board = pos.board
    board[2, 8] = go.BLACK
    print()
    print(board)
    nboard = Framer.surround(board.copy(), (0, 3))
    print(nboard - board)
