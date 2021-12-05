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
from collections import namedtuple, defaultdict
from typing import Tuple, Dict, Set
import numpy as np

import go
from go import LibertyTracker


class Region(namedtuple('Region', ['id', 'stones', 'liberties', 'chains', 'color'])):
    """
    stones: a frozenset of Coordinates belonging to this group
    liberties: subset of stones that are empty
    chains: enclosing chains
    color: empty if all empty, otherwise color of the present stones
    """

    def __eq__(self, other):
        return self.stones == other.stones and self.liberties == other.liberties and \
               self.chains == other.chains and self.color == other.color


class PassAliveTracker:
    """
    - is it easy to incrementally update its status?
    """
    def __init__(self):
        self.region_index = -np.ones([go.N, go.N], dtype=np.int32)  # type: np.ndarray
        self.regions = dict()  # type: Dict[int, Region]
        self.max_region_id = 0
        self.lib_tracker = None  # type: LibertyTracker

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
                # reached -> set of bordering chains
                chains = frozenset(lib_tracker.group_index[s] for s in reached)
                assert all(lib_tracker.groups[i].color == color_bound for i in chains)
                # region color indicates presence of enemy stone
                region_color = -color_bound if len(liberties) < len(region) else color

                new_region = Region(curr_region_id, frozenset(region), liberties, chains, region_color)
                tracker.regions[curr_region_id] = new_region
                for s in region:
                    tracker.region_index[s] = curr_region_id
                go.place_stones(board, go.FILL, region)

        tracker.lib_tracker = lib_tracker
        tracker.max_region_id = curr_region_id
        return tracker

    def eliminate(self, color) -> Set[int]:
        """ find pass-alive chains for color, using Benson's algorithm
        """
        chains_current = set(idx for idx, chain in self.lib_tracker.groups.items() if chain.color == color)
        regions_current = [r for r in self.regions.values()]

        for i in range(100):
            print(f"Benson iter {i}: %d chains, %d regions" % (len(chains_current), len(regions_current)))

            num_vital_regions = defaultdict(int)
            for region in regions_current:
                # see which chains this is vital for
                for chain_idx in region.chains:
                    chain = self.lib_tracker.groups[chain_idx]
                    if len(chain.liberties) < len(region.liberties):
                        continue
                    if region.liberties.issubset(chain.liberties):
                        num_vital_regions[chain_idx] += 1

            # see if it has at least two (small) vital regions
            chains_pruned = set(idx for idx in chains_current if num_vital_regions[idx] >= 2)
            # prune regions
            regions_pruned = [r for r in regions_current if all(chain_idx in chains_pruned for chain_idx in r.chains)]

            if len(chains_pruned) == 0:
                return chains_pruned
            if len(chains_pruned) == len(chains_current) and len(regions_pruned) == len(regions_current):
                return chains_pruned

            chains_current, regions_current = chains_pruned, regions_pruned
