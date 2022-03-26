"""
compute a hash of go.Position board + to_play, to
- detect dups under symmetries, so that MCTS will explore just one variant
- identify effective #states visited in training runsQ

It should have low collision rate.
Zobrist hash should be ok as a baseline. We only need symmetry-dup detection at earlier stage of the game,
where we can use a min of Zobrist hash over 8 symmetries. After this stage we don't worry about it anymore,
as dups are less likely.
"""

import numpy as np
import go


def hash_under_symmetry(pos: go.Position):
    """ """


def hash_position(pos: go.Position):
    """ """
    pass


def hash_board(board: np.array):
    """ log(3) bit per spot:  * 81 = 128.4 bits
    """
