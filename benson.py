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
from collections import namedtuple
from typing import Tuple, Dict
import numpy as np

import coords
import go
from go import LibertyTracker, Group, N
from tests import test_utils
from tests.test_go import coords_from_gtp_set


class Region(namedtuple('Group', ['id', 'stones', 'liberties', 'chains', 'color'])):
    """
    stones: a frozenset of Coordinates belonging to this group
    liberties: a frozenset of Coordinates that are empty and adjacent to this group.
    color: color of this group
    """


class PassAliveTracker:
    """
    - how we will use it:
    - is it easy to incrementally update its status?
    """
    def __init__(self):
        self.region_index = -np.ones([go.N, go.N], dtype=np.int32)  # type: np.ndarray
        self.regions = dict()  # type: Dict[int, Region]
        self.max_region_id = 0
        self.lib_tracker = None

    @staticmethod
    def from_board(board: np.ndarray, color_bound) -> 'PassAliveTracker':
        tracker = PassAliveTracker()
        lib_tracker = LibertyTracker.from_board(board)

        board = np.copy(board)
        curr_region_id = 0
        for color in (go.EMPTY,):
            while color in board:
                curr_region_id += 1
                found_color = np.where(board == color)
                coord = found_color[0][0], found_color[1][0]
                region, reached = go.find_maximal_region_with_no(board, coord, color_bound)
                liberties = frozenset(r for r in region if board[r] == go.EMPTY)
                # reached -> set of bordering chains?
                chains = frozenset(lib_tracker.group_index[s] for s in reached)
                new_region = Region(curr_region_id, frozenset(region), liberties, chains, color)
                tracker.regions[curr_region_id] = new_region
                for s in region:
                    tracker.region_index[s] = curr_region_id
                go.place_stones(board, go.FILL, region)

        tracker.lib_tracker = lib_tracker
        tracker.max_region_id = curr_region_id
        return tracker

    def eliminate(self, color):
        """ find pass-alive chains for color
        Benson's algorithm
        """
        for gid, group in self.lib_tracker.regions.items():
            if group.color != color:
                continue
            # see if it has at least two (small) vital regions



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
        assert len(tracker.regions) == 3
        for rid, region in tracker.regions.items():
            print(rid, region.color, len(region.stones), len(region.liberties), region.chains)

    def test_not_pass_alive(self):
        """ two black chains, two eyes: one vital to both, one (two spaces) vital to one only
        With either black or white(!) in one of the two spaces, it becomes vital
        """
