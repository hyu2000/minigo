""" utility to fill a board around a Tsumego

Basically we want to fill the rest of the board, so that the only focus is on the Tsumego.

We don't need to be super-exact, just enough to discourage a bot to play outside Tsumego.
"""
import numpy as np

import go
from sgf_wrapper import SGFReader


class Framer:
    def __init__(self, board: np.array):
        self.puzzle = board

    @staticmethod
    def surround(board: np.array, coord: tuple, opposite_stone = True):
        """ add a one-stone wide fringe around area/chain identified by coord """
        # chain is not quite what we want?
        chain, boundary = go.find_reached(board, coord)
        color = board[coord]
        if opposite_stone:
            color = -color
        for p in boundary:
            # check p is not taken?
            board[p] = color
        return board


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
