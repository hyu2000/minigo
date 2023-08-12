""" utility to fill a board around a Tsumego

Basically we want to fill the rest of the board, so that the only focus is on the Tsumego.

We don't need to be super-exact, just enough to discourage a bot to play outside Tsumego.
"""
import itertools
from typing import Tuple

import numpy as np

import coords
import go
from sgf_wrapper import SGFReader


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
    def __init__(self, board: np.array):
        self.puzzle = board

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
    sgf_fname = '/Users/hyu/Downloads/go-puzzle9/Amigo no igo - 詰碁2023 - Life and Death/総合問題4級(2).sgf'
    reader = SGFReader.from_file_compatible(sgf_fname)
    pos = reader.first_pos()
    board = pos.board

    board = abs(board)
    coord_init = coords.from_gtp('D7')
    for i in range(1):
        chain, boundary = find_chainlike(board, coord_init)
        for c in boundary:
            board[c] = go.BLACK
    # just find boundary in the end
    chain, boundary = find_chainlike(board, coord_init)
    print('boundary:', ' '.join(sorted([coords.to_gtp(c) for c in boundary])))


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
