""" Benson's algorithm to determine pass-alive chains

Let X be the set of all Black chains and R be the set of all Black-enclosed regions of X.
Then Benson's algorithm requires iteratively applying the following two steps until neither
is able to remove any more chains or regions:

Remove from X all Black chains with less than two vital Black-enclosed regions in R, where a Black-enclosed region
is **vital** to a Black chain in X if *all* its empty intersections are also liberties of the chain.
Remove from R all Black-enclosed regions with a surrounding stone in a chain not in X.
The final set X is the set of all unconditionally alive Black chains.

Implementation:
- LibertyTracker tracks groups
- Black-enclosed regions: start from empty spots, find the max region (include white) that's surrounded by black
  or wall.  This is similar to a chain, just that it's the maximal region of empty+white

"""
from typing import Tuple, Dict
import numpy as np

import coords
import go
from go import LibertyTracker, Group, N
from tests import test_utils
from tests.test_go import coords_from_gtp_set


class PassAliveTracker:
    """
    - how we will use it:
    - is it easy to incrementally update its status?
    """
    def __init__(self):
        self.group_index = -np.ones([go.N, go.N], dtype=np.int32)  # type: np.ndarray
        self.groups = dict()  # type: Dict[int, Group]
        self.max_group_id = 0

    @staticmethod
    def from_board(board: np.ndarray, color_bound) -> 'PassAliveTracker':
        board = np.copy(board)
        curr_group_id = 0
        tracker = PassAliveTracker()
        for color in (go.EMPTY,):
            while color in board:
                curr_group_id += 1
                found_color = np.where(board == color)
                coord = found_color[0][0], found_color[1][0]
                chain, reached = go.find_maximal_region_with_no(board, coord, color_bound)
                liberties = frozenset(r for r in reached if board[r] == go.EMPTY)
                new_group = go.Group(curr_group_id, frozenset(chain), liberties, color)
                tracker.groups[curr_group_id] = new_group
                for s in chain:
                    tracker.group_index[s] = curr_group_id
                go.place_stones(board, go.FILL, chain)

        tracker.max_group_id = curr_group_id
        return tracker


def find_all_enclosed_regions(board, player):
    """ find black (player) enclosed regions
    """
    lib_tracker = LibertyTracker.from_board(board)
    # for all empty spots, find maximal region which will be bounded by black
    # pretty much from_board() logic



class TestLibertyTracker(test_utils.MinigoUnitTest):
    EMPTY_ROW = '.' * go.N + '\n'

    def test_basic(self):
        """ from test_capture_many """
        board = test_utils.load_board('''
            .XX......
            X..X.....
            XXXX..O..
        ''' + self.EMPTY_ROW * 6)
        lib_tracker = LibertyTracker.from_board(board)
        self.assertEqual(len(lib_tracker.groups), 3)
        self.assertEqual(lib_tracker.group_index[coords.from_gtp('B8')], go.MISSING_GROUP_ID)

        for gid, group in lib_tracker.groups.items():
            print(gid, group.color, len(group.stones), len(group.liberties))

        found_color = np.where(board == go.WHITE)
        coord = found_color[0][0], found_color[1][0]
        print(coord)

    def test_maximal_region(self):
        board = test_utils.load_board('''
            .XX......
            X..X.....
            XXXX..O..
        ''' + self.EMPTY_ROW * 6)
        tracker = PassAliveTracker.from_board(board, go.BLACK)
        assert len(tracker.groups) == 3
        for gid, group in tracker.groups.items():
            print(gid, group.color, len(group.stones), len(group.liberties))

    def test_not_pass_alive(self):
        """ two black chains, two eyes: one vital to both, one (two spaces) vital to one only
        With either black or white(!) in one of the two spaces, it becomes vital
        """
