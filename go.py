# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A board is a NxN numpy array.
A Coordinate is a tuple index into the board.
A Move is a (Coordinate c | None).
A PlayerMove is a (Color, Move) tuple

(0, 0) is considered to be the upper left corner of the board, and (18, 0) is the lower left.
"""
from collections import namedtuple, defaultdict
import copy
import itertools
from typing import Iterable, Sequence, Dict, Tuple, Set

import numpy as np
import os

import coords
import zobrist

N = int(os.environ.get('BOARD_SIZE', 9))  # was 19
zobrist_hasher = zobrist.ZobristHash(N)

# Position.score() switches from Tromp to Benson after this, for speed reasons
NUM_MOVES_BEFORE_BENSON = N * N // 2  # this is sensitive to go.N.  40 for 9x9

# Represent a board as a numpy array, with 0 empty, 1 is black, -1 is white.
# This means that swapping colors is as simple as multiplying array by -1.
WHITE, EMPTY, BLACK, FILL, KO, UNKNOWN = range(-1, 5)


def color_str(color: int) -> str:
    if color == WHITE:
        return 'WHITE'
    if color == BLACK:
        return 'BLACK'
    if color == EMPTY:
        return 'EMPTY'
    return str(color)

# Represents "group not found" in the LibertyTracker object
MISSING_GROUP_ID = -1

ALL_COORDS = [(i, j) for i in range(N) for j in range(N)]
EMPTY_BOARD = np.zeros([N, N], dtype=np.int8)


def _check_bounds(c):
    return 0 <= c[0] < N and 0 <= c[1] < N


NEIGHBORS = {(x, y): list(filter(_check_bounds, [
    (x+1, y), (x-1, y), (x, y+1), (x, y-1)])) for x, y in ALL_COORDS}
DIAGONALS = {(x, y): list(filter(_check_bounds, [
    (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)])) for x, y in ALL_COORDS}


class IllegalMove(Exception):
    pass


class PlayerMove(namedtuple('PlayerMove', ['color', 'move'])):
    pass


class PositionWithContext(namedtuple('SgfPosition', ['position', 'next_move', 'result'])):
    pass


def place_stones(board, color, stones):
    for s in stones:
        board[s] = color


def replay_position(position, result, initial_position=None):
    """
    Wrapper for a go.Position which replays its history.
    # Assumes an empty start position! (i.e. no handicap, and history must be exhaustive.)

    Result must be passed in, since a resign cannot be inferred from position
    history alone.

    for position_w_context in replay_position(position):
        print(position_w_context.position)
    """
    pos = initial_position
    if pos is None:
        pos = Position(komi=position.komi)
    assert position.n == len(position.recent), "Position history is incomplete"
    for player_move in position.recent[pos.n:]:
        color, next_move = player_move
        yield PositionWithContext(pos, next_move, result)
        pos = pos.play_move(next_move, color=color)


def find_reached(board: np.ndarray, c: Tuple) -> Tuple[set, set]:
    color = board[c]
    chain = {c}
    reached = set()
    frontier = [c]
    while frontier:
        current = frontier.pop()
        chain.add(current)
        for n in NEIGHBORS[current]:
            if board[n] == color:
                if n not in chain:
                    frontier.append(n)
            else:
                reached.add(n)
    return chain, reached


def find_maximal_region_with_no(board: np.ndarray, c: Tuple, color_bound) -> Tuple[set, set]:
    """ similar to find_reached, assume color_bound is black, find maximal region starting from c,
    reachable thru empty or white stones
    """
    assert board[c] != color_bound
    chain = {c}
    border = set()
    frontier = [c]
    while frontier:
        current = frontier.pop()
        chain.add(current)
        for n in NEIGHBORS[current]:
            if board[n] != color_bound:
                if n not in chain:
                    frontier.append(n)
            else:
                border.add(n)
    return chain, border


def is_koish(board, c):
    'Check if c is surrounded on all sides by 1 color, and return that color'
    if board[c] != EMPTY:
        return None
    neighbors = {board[n] for n in NEIGHBORS[c]}
    if len(neighbors) == 1 and EMPTY not in neighbors:
        return list(neighbors)[0]
    else:
        return None


def is_eyeish(board, c):
    'Check if c is an eye, for the purpose of restricting MC rollouts.'
    # pass is fine.
    if c is None:
        return
    color = is_koish(board, c)
    if color is None:
        return None
    diagonal_faults = 0
    diagonals = DIAGONALS[c]
    if len(diagonals) < 4:
        diagonal_faults += 1
    for d in diagonals:
        if not board[d] in (color, EMPTY):
            diagonal_faults += 1
    if diagonal_faults > 1:
        return None
    else:
        return color


class Group(namedtuple('Group', ['id', 'stones', 'liberties', 'color'])):
    """
    stones: a frozenset of Coordinates belonging to this group
    liberties: a frozenset of Coordinates that are empty and adjacent to this group.
    color: color of this group
    """

    def __eq__(self, other):
        return self.stones == other.stones and self.liberties == other.liberties and self.color == other.color


class LibertyTracker():
    @staticmethod
    def from_board(board: np.ndarray) -> 'LibertyTracker':
        board = np.copy(board)
        curr_group_id = 0
        lib_tracker = LibertyTracker()
        for color in (WHITE, BLACK):
            while color in board:
                curr_group_id += 1
                found_color = np.where(board == color)
                coord = found_color[0][0], found_color[1][0]
                chain, reached = find_reached(board, coord)
                liberties = frozenset(r for r in reached if board[r] == EMPTY)
                new_group = Group(curr_group_id, frozenset(chain), liberties, color)
                lib_tracker.groups[curr_group_id] = new_group
                for s in chain:
                    lib_tracker.group_index[s] = curr_group_id
                place_stones(board, FILL, chain)

        lib_tracker.max_group_id = curr_group_id

        liberty_counts = np.zeros([N, N], dtype=np.uint8)
        for group in lib_tracker.groups.values():
            num_libs = len(group.liberties)
            for s in group.stones:
                liberty_counts[s] = num_libs
        lib_tracker.liberty_cache = liberty_counts

        return lib_tracker

    def __init__(self, group_index=None, groups=None, liberty_cache=None, max_group_id=1):
        # group_index: a NxN numpy array of group_ids. -1 means no group
        # groups: a dict of group_id to groups
        # liberty_cache: a NxN numpy array of liberty counts
        self.group_index = group_index if group_index is not None else -np.ones([N, N], dtype=np.int32)  # type: np.ndarray
        self.groups = groups or {}  # type: Dict[int, Group]
        self.liberty_cache = liberty_cache if liberty_cache is not None else np.zeros([N, N], dtype=np.uint8)  # type: np.ndarray
        self.max_group_id = max_group_id

    def __deepcopy__(self, memodict={}):
        new_group_index = np.copy(self.group_index)
        new_lib_cache = np.copy(self.liberty_cache)
        # shallow copy
        new_groups = copy.copy(self.groups)
        return LibertyTracker(new_group_index, new_groups, liberty_cache=new_lib_cache, max_group_id=self.max_group_id)

    def add_stone(self, color, c) -> set:
        assert self.group_index[c] == MISSING_GROUP_ID
        captured_stones = set()
        opponent_neighboring_group_ids = set()
        friendly_neighboring_group_ids = set()
        empty_neighbors = set()

        for n in NEIGHBORS[c]:
            neighbor_group_id = self.group_index[n]
            if neighbor_group_id != MISSING_GROUP_ID:
                neighbor_group = self.groups[neighbor_group_id]
                if neighbor_group.color == color:
                    friendly_neighboring_group_ids.add(neighbor_group_id)
                else:
                    opponent_neighboring_group_ids.add(neighbor_group_id)
            else:
                empty_neighbors.add(n)

        new_group = self._merge_from_played(
            color, c, empty_neighbors, friendly_neighboring_group_ids)

        # new_group becomes stale as _update_liberties and
        # _handle_captures are called; must refetch with self.groups[new_group.id]
        for group_id in opponent_neighboring_group_ids:
            neighbor_group = self.groups[group_id]
            if len(neighbor_group.liberties) == 1:
                captured = self._capture_group(group_id)
                captured_stones.update(captured)
            else:
                self._update_liberties(group_id, remove={c})

        self._handle_captures(captured_stones)

        # suicide is illegal
        if len(self.groups[new_group.id].liberties) == 0:
            raise IllegalMove("Move at {} would commit suicide!\n".format(c))

        return captured_stones

    def _merge_from_played(self, color, played, libs, other_group_ids):
        stones = {played}
        liberties = set(libs)
        for group_id in other_group_ids:
            other = self.groups.pop(group_id)
            stones.update(other.stones)
            liberties.update(other.liberties)

        if other_group_ids:
            liberties.remove(played)
        assert stones.isdisjoint(liberties)
        self.max_group_id += 1
        result = Group(
            self.max_group_id,
            frozenset(stones),
            frozenset(liberties),
            color)
        self.groups[result.id] = result

        for s in result.stones:
            self.group_index[s] = result.id
            self.liberty_cache[s] = len(result.liberties)

        return result

    def _capture_group(self, group_id):
        dead_group = self.groups.pop(group_id)
        for s in dead_group.stones:
            self.group_index[s] = MISSING_GROUP_ID
            self.liberty_cache[s] = 0
        return dead_group.stones

    def _update_liberties(self, group_id, add=set(), remove=set()):
        group = self.groups[group_id]
        new_libs = (group.liberties | add) - remove
        self.groups[group_id] = Group(
            group_id, group.stones, new_libs, group.color)

        new_lib_count = len(new_libs)
        for s in self.groups[group_id].stones:
            self.liberty_cache[s] = new_lib_count

    def _handle_captures(self, captured_stones):
        for s in captured_stones:
            for n in NEIGHBORS[s]:
                group_id = self.group_index[n]
                if group_id != MISSING_GROUP_ID:
                    self._update_liberties(group_id, add={s})


class Region(namedtuple('Region', ['id', 'stones', 'liberties', 'chains', 'color'])):
    """
    stones: a frozenset of Coordinates belonging to this region
    liberties: subset of area that are empty
    chains: enclosing chains
    color: empty if all empty, otherwise color of the opponent
    """

    def __eq__(self, other):
        return self.stones == other.stones and self.liberties == other.liberties and \
               self.chains == other.chains and self.color == other.color


class BensonAnalyzer:
    """ Benson's algorithm to determine pass-alive chains

    Let X be the set of all Black chains and R be the set of all Black-enclosed regions of X.
    Then Benson's algorithm requires iteratively applying the following two steps until neither
    is able to remove any more chains or regions:

    Remove from X all Black chains with less than two vital Black-enclosed regions in R, where a Black-enclosed region
    is **vital** to a Black chain in X if *all* its empty intersections are also liberties of the chain.
    Remove from R all Black-enclosed regions with a surrounding stone in a chain not in X.
    The final set X is the set of all unconditionally alive Black chains.

    - is it easy to incrementally update its status?
    """
    def __init__(self, board: np.ndarray, color_bound):
        self.region_index = -np.ones([N, N], dtype=np.int32)  # type: np.ndarray
        self.regions = dict()  # type: Dict[int, Region]
        self.max_region_id = 0

        self.lib_tracker = LibertyTracker.from_board(board)
        self._find_enclosed_regions(board, color_bound)
        self.color_bound = color_bound

    def _find_enclosed_regions(self, board, color_bound):
        """
        Black-enclosed regions: start from empty spots, find the max region (include white) that's surrounded by black
        or wall. This is similar to a chain, just that it's the maximal region of empty+white
        """
        board = np.copy(board)
        lib_tracker = self.lib_tracker

        curr_region_id = 0
        for color in (EMPTY,):
            while color in board:
                curr_region_id += 1
                found_color = np.where(board == color)
                coord = found_color[0][0], found_color[1][0]
                region, reached = find_maximal_region_with_no(board, coord, color_bound)
                liberties = frozenset(r for r in region if board[r] == EMPTY)
                # reached -> set of bordering chains
                chains = frozenset(lib_tracker.group_index[s] for s in reached)
                assert all(lib_tracker.groups[i].color == color_bound for i in chains)
                # region color indicates presence of enemy stone
                region_color = -color_bound if len(liberties) < len(region) else color

                new_region = Region(curr_region_id, frozenset(region), liberties, chains, region_color)
                self.regions[curr_region_id] = new_region
                for s in region:
                    self.region_index[s] = curr_region_id
                place_stones(board, FILL, region)

        self.max_region_id = curr_region_id

    @staticmethod
    def from_board(board: np.ndarray, color_bound) -> 'BensonAnalyzer':
        return BensonAnalyzer(board, color_bound)

    def eliminate(self) -> Tuple[Set[int], Iterable[Region]]:
        """ find pass-alive chains for color, using Benson's algorithm.

        Note we use regions returned for scoring purposes. They need to be what Benson's original algo specifies.
        regions are mainly black-enclosed regions, but could be a big neighboring white region (side-by-side)
        """
        chains_current = set(idx for idx, chain in self.lib_tracker.groups.items() if chain.color == self.color_bound)
        regions_current = [r for r in self.regions.values()]

        for i in range(100):
            # print(f'Benson iter {i}: %d chains, %d regions' % (len(chains_current), len(regions_current)))

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
                return chains_pruned, []
            if len(chains_pruned) == len(chains_current) and len(regions_pruned) == len(regions_current):
                return chains_pruned, regions_pruned

            chains_current, regions_current = chains_pruned, regions_pruned

    def remove_non_vital_regions(self):
        """ remove regions that are not vital to any safe chain.
        This just establish eye-space better
        """
        # regions_final = [r for r in regions_pruned if r.id in vital_regions]
        # regions_nonvital = [r for r in regions_pruned if r.id not in vital_regions]
        # if regions_nonvital:
        #     print('final regions: removed %d of %d' % (len(regions_nonvital), len(regions_pruned)))
        #     print('\t#opp stones: %s' % [len(r.stones) - len(r.liberties) for r in regions_nonvital])


class BensonScoreDetail(namedtuple('BensonScoreDetail', ['score', 'final', 'black_area', 'white_area'])):
    """ This counts unconditional live areas correctly; for everything else, it resorts to Tromp scoring.

    Note: when final=True, we are sure of the winner, but score might not be exact, since the game may not
    have finished (but won't affect final winner). score sign should be correct.

    score has komi baked in
    """
    pass


class Position:
    def __init__(self, board=None, n=0, komi=5.5, caps=(0, 0),
                 lib_tracker=None, ko=None, recent=tuple(),
                 board_deltas=None, to_play=BLACK, zobrist_hash=None):
        """
        board: a numpy array
        n: an int representing moves played so far
        komi: a float, representing points given to the second player.
        caps: a (int, int) tuple of captures for B, W.
        lib_tracker: a LibertyTracker object
        ko: a Move
        recent: a tuple of PlayerMoves, such that recent[-1] is the last move.
        board_deltas: a np.array of shape (n, go.N, go.N) representing changes
            made to the board at each move (played move and captures).
            Should satisfy next_pos.board - next_pos.board_deltas[0] == pos.board
        to_play: BLACK or WHITE
        """
        assert type(recent) is tuple
        self.board = board if board is not None else np.copy(EMPTY_BOARD)
        # With a full history, self.n == len(self.recent) == num moves played
        self.n = n
        self.komi = komi
        self.caps = caps
        self.lib_tracker = lib_tracker or LibertyTracker.from_board(self.board)
        self.ko = ko
        self.recent = recent  # type: Sequence[PlayerMove]
        self.board_deltas = board_deltas if board_deltas is not None else np.zeros([
                                                                                   0, N, N], dtype=np.int8)
        self.to_play = to_play

        if zobrist_hash is None:
            self.zobrist_hash = zobrist_hasher.EMPTY_BOARD_HASH if board is None else zobrist_hasher.board_hash(board)
        else:
            self.zobrist_hash = zobrist_hash

    def __deepcopy__(self, memodict={}):
        new_board = np.copy(self.board)
        new_lib_tracker = copy.deepcopy(self.lib_tracker)
        return Position(new_board, self.n, self.komi, self.caps, new_lib_tracker, self.ko, self.recent,
                        self.board_deltas, self.to_play, self.zobrist_hash)

    def __str__(self, colors=True):
        if colors:
            pretty_print_map = {
                WHITE: '\x1b[0;31;47mO',
                EMPTY: '\x1b[0;31;43m.',
                BLACK: '\x1b[0;31;40mX',
                FILL: '#',
                KO: '*',
            }
        else:
            pretty_print_map = {
                WHITE: 'O',
                EMPTY: '.',
                BLACK: 'X',
                FILL: '#',
                KO: '*',
            }
        board = np.copy(self.board)
        captures = self.caps
        if self.ko is not None:
            place_stones(board, KO, [self.ko])
        raw_board_contents = []
        for i in range(N):
            row = [' ']
            for j in range(N):
                appended = '<' if (self.recent and (i, j) ==
                                   self.recent[-1].move) else ' '
                row.append(pretty_print_map[board[i, j]] + appended)
                if colors:
                    row.append('\x1b[0m')

            raw_board_contents.append(''.join(row))

        row_labels = ['%2d' % i for i in range(N, 0, -1)]
        annotated_board_contents = [''.join(r) for r in zip(
            row_labels, raw_board_contents, row_labels)]
        header_footer_rows = [
            '   ' + ' '.join('ABCDEFGHJKLMNOPQRST'[:N]) + '   ']
        annotated_board = '\n'.join(itertools.chain(
            header_footer_rows, annotated_board_contents, header_footer_rows))
        details = "\nMove: {}. Captures X: {} O: {}\n".format(
            self.n, *captures)
        return annotated_board + details

    def is_move_suicidal(self, move):
        potential_libs = set()
        for n in NEIGHBORS[move]:
            neighbor_group_id = self.lib_tracker.group_index[n]
            if neighbor_group_id == MISSING_GROUP_ID:
                # at least one liberty after playing here, so not a suicide
                return False
            neighbor_group = self.lib_tracker.groups[neighbor_group_id]
            if neighbor_group.color == self.to_play:
                potential_libs |= neighbor_group.liberties
            elif len(neighbor_group.liberties) == 1:
                # would capture an opponent group if they only had one lib.
                return False
        # it's possible to suicide by connecting several friendly groups
        # each of which had one liberty.
        potential_libs -= set([move])
        return not potential_libs

    def is_move_legal(self, move):
        'Checks that a move is on an empty space, not on ko, and not suicide'
        if move is None:
            return True
        if self.board[move] != EMPTY:
            return False
        if move == self.ko:
            return False
        if self.is_move_suicidal(move):
            return False

        return True

    def all_legal_moves(self):
        'Returns a np.array of size go.N**2 + 1, with 1 = legal, 0 = illegal'
        # by default, every move is legal
        legal_moves = np.ones([N, N], dtype=np.int8)
        # ...unless there is already a stone there
        legal_moves[self.board != EMPTY] = 0
        # calculate which spots have 4 stones next to them
        # padding is because the edge always counts as a lost liberty.
        adjacent = np.ones([N + 2, N + 2], dtype=np.int8)
        adjacent[1:-1, 1:-1] = np.abs(self.board)
        num_adjacent_stones = (adjacent[:-2, 1:-1] + adjacent[1:-1, :-2] +
                               adjacent[2:, 1:-1] + adjacent[1:-1, 2:])
        # Surrounded spots are those that are empty and have 4 adjacent stones.
        surrounded_spots = np.multiply(
            (self.board == EMPTY),
            (num_adjacent_stones == 4))
        # Such spots are possibly illegal, unless they are capturing something.
        # Iterate over and manually check each spot.
        for coord in np.transpose(np.nonzero(surrounded_spots)):
            if self.is_move_suicidal(tuple(coord)):
                legal_moves[tuple(coord)] = 0

        # ...and retaking ko is always illegal
        if self.ko is not None:
            legal_moves[self.ko] = 0

        # and pass is always legal
        return np.concatenate([legal_moves.ravel(), [1]])

    def pass_move(self, mutate=False) -> 'Position':
        pos = self if mutate else copy.deepcopy(self)
        pos.n += 1
        pos.recent += (PlayerMove(pos.to_play, None),)
        pos.board_deltas = np.concatenate((
            np.zeros([1, N, N], dtype=np.int8),
            pos.board_deltas[:6]))
        pos.to_play *= -1
        pos.ko = None
        return pos

    def flip_playerturn(self, mutate=False):
        pos = self if mutate else copy.deepcopy(self)
        pos.ko = None
        pos.to_play *= -1
        return pos

    def get_liberties(self):
        return self.lib_tracker.liberty_cache

    def play_move(self, c, color=None, mutate=False) -> 'Position':
        # Obeys CGOS Rules of Play. In short:
        # No suicides
        # Chinese/area scoring
        # Positional superko (this is very crudely approximate at the moment.)
        if color is None:
            color = self.to_play

        if c is None:
            pos = self.pass_move(mutate=mutate)
            return pos

        pos = self if mutate else copy.deepcopy(self)

        if not self.is_move_legal(c):
            raise IllegalMove("{} move at {} is illegal: \n{}".format(
                "Black" if self.to_play == BLACK else "White",
                coords.to_gtp(c), self))

        potential_ko = is_koish(self.board, c)

        place_stones(pos.board, color, [c])
        captured_stones = pos.lib_tracker.add_stone(color, c)
        place_stones(pos.board, EMPTY, captured_stones)

        opp_color = color * -1

        new_board_delta = np.zeros([N, N], dtype=np.int8)
        new_board_delta[c] = color
        place_stones(new_board_delta, color, captured_stones)

        if len(captured_stones) == 1 and potential_ko == opp_color:
            new_ko = list(captured_stones)[0]
        else:
            new_ko = None

        if pos.to_play == BLACK:
            new_caps = (pos.caps[0] + len(captured_stones), pos.caps[1])
        else:
            new_caps = (pos.caps[0], pos.caps[1] + len(captured_stones))

        pos.n += 1
        pos.caps = new_caps
        pos.ko = new_ko
        pos.recent += (PlayerMove(color, c),)
        pos.zobrist_hash = zobrist_hasher.hash_after_move(self, c, captured_stones)

        # keep a rolling history of last 7 deltas - that's all we'll need to
        # extract the last 8 board states.
        pos.board_deltas = np.concatenate((
            new_board_delta.reshape(1, N, N),
            pos.board_deltas[:6]))
        pos.to_play *= -1
        return pos

    def is_game_over(self):
        return (len(self.recent) >= 2 and
                self.recent[-1].move is None and
                self.recent[-2].move is None)

    def score_tromp(self) -> float:
        """Return score from B perspective. If W is winning, score is negative.
        score = 0 could happen if komi is integer
        """
        working_board = np.copy(self.board)
        return self._score_board(working_board)

    def _score_board(self, working_board):
        while EMPTY in working_board:
            unassigned_spaces = np.where(working_board == EMPTY)
            c = unassigned_spaces[0][0], unassigned_spaces[1][0]
            territory, borders = find_reached(working_board, c)
            border_colors = set(working_board[b] for b in borders)
            X_border = BLACK in border_colors
            O_border = WHITE in border_colors
            if X_border and not O_border:
                territory_color = BLACK
            elif O_border and not X_border:
                territory_color = WHITE
            else:
                territory_color = UNKNOWN  # dame, or seki
            place_stones(working_board, territory_color, territory)

        return np.count_nonzero(working_board == BLACK) - np.count_nonzero(working_board == WHITE) - self.komi

    def score(self) -> float:
        """ score has komi baked in """
        if self.n < NUM_MOVES_BEFORE_BENSON:
            # Pass-alive typically happens later in game. Use Tromp which is faster
            return self.score_tromp()
        score_detail = self._benson_analysis()
        return score_detail.score

    def score_benson(self) -> BensonScoreDetail:
        return self._benson_analysis()

    def _benson_analysis(self) -> BensonScoreDetail:
        """ based on Benson's algo: this method will remove dead stones in pass-alive area
        It also indicates whether winner is final: either all area is settled, or that winner's margin is large enough
        """
        working_board = np.copy(self.board)

        # first, mark pass-alive area
        area_passalive = [0, 0]    # black live, white live
        num_removed = [0, 0]       # white dead in black area, black dead in white area
        for icolor, color in enumerate((BLACK, WHITE)):
            analyzer = BensonAnalyzer(self.board, color)
            chain_ids, regions = analyzer.eliminate()
            num_dead_stones = 0
            for region in regions:  # region is black-enclosed, but could be completely owned by white
                num_opp_stones = len(region.stones) - len(region.liberties)
                if num_opp_stones >= 4 and len(region.liberties) >= 2:
                    # heuristics: consider 3 white stones not survivable in a black pass-alive area
                    # Either white region (with 2 potential eyes), or white could survive, consider it unsettled
                    continue
                num_dead_stones += num_opp_stones
                place_stones(working_board, color, region.stones)
                area_passalive[icolor] += len(region.stones)
            num_removed[icolor] = num_dead_stones

            for chain_id in chain_ids:
                chain = analyzer.lib_tracker.groups[chain_id]
                assert chain.color == color
                area_passalive[icolor] += len(chain.stones)

        # see if we know the winner regardless of unsettled area
        num_unsettled = N * N - sum(area_passalive)
        advantage = area_passalive[0] - area_passalive[1] - self.komi
        game_over = abs(advantage) > num_unsettled or num_unsettled == 0

        # everything else, use Tromp scoring
        score = self._score_board(working_board)
        # this should always hold: when game_over, score should always have the right sign as winner
        # if game_over:
        #     assert np.sign(advantage) == np.sign(score)
        # if game_over:
        #     print('benson scoring: advantage=%.1f, unsettled area=%d, game over' % (advantage, num_unsettled))
        # if sum(num_removed) > 0:
        #     print(f'score_benson: removed %s dead stones from pass-alive area -> %.1f' % (num_removed, score))
        return BensonScoreDetail(score, game_over, area_passalive[0], area_passalive[1])

    def result(self):
        score = self.score()
        if score > 0:
            return 1
        elif score < 0:
            return -1
        else:
            return 0

    def result_string(self):
        score = self.score()
        if score > 0:
            return 'B+' + '%.1f' % score
        elif score < 0:
            return 'W+' + '%.1f' % abs(score)
        else:
            return 'DRAW'
