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
- Black-enclosed regions: start from empty spots, find the max region (include white) that's surrounded by black or wall
"""
import coords
import go
from go import LibertyTracker
from tests import test_utils
from tests.test_go import coords_from_gtp_set


class TestLibertyTracker(test_utils.MinigoUnitTest):
    EMPTY_ROW = '.' * go.N + '\n'

    def test_capture_many(self):
        board = test_utils.load_board('''
            .XX......
            XOO......
            .XX...O..
        ''' + self.EMPTY_ROW * 6)
        lib_tracker = LibertyTracker.from_board(board)
        captured = lib_tracker.add_stone(go.BLACK, coords.from_gtp('D8'))
        self.assertEqual(len(lib_tracker.groups), 5)
        self.assertEqual(lib_tracker.group_index[coords.from_gtp('B8')], go.MISSING_GROUP_ID)
        self.assertEqual(captured, coords_from_gtp_set('B8 C8'))
