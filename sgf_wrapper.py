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
Code to extract a series of positions + their next moves from an SGF.

Most of the complexity here is dealing with two features of SGF:
- Stones can be added via "play move" or "add move", the latter being used
  to configure L+D puzzles, but also for initial handicap placement.
- Plays don't necessarily alternate colors; they can be repeated B or W moves
  This feature is used to handle free handicap placement.
"""
from typing import Tuple, Optional, Iterable

import numpy as np
import itertools

import coords
import go
from go import Position, PositionWithContext
import utils
import sgf
from absl import logging


SGF_TEMPLATE = '''(;GM[1]FF[4]CA[UTF-8]AP[Minigo_sgfgenerator]RU[{ruleset}]
SZ[{boardsize}]KM[{komi}]PW[{white_name}]PB[{black_name}]RE[{result}]
{game_moves})'''

PROGRAM_IDENTIFIER = "Minigo"


def translate_sgf_move(player_move, comment):
    if player_move.color not in (go.BLACK, go.WHITE):
        raise ValueError("Can't translate color %s to sgf" % player_move.color)
    c = coords.to_sgf(player_move.move)
    color = 'B' if player_move.color == go.BLACK else 'W'
    if comment is not None:
        comment = comment.replace(']', r'\]')
        comment_node = "C[{}]".format(comment)
    else:
        comment_node = ""
    return ";{color}[{coords}]{comment_node}".format(
        color=color, coords=c, comment_node=comment_node)


def make_sgf(
    move_history,
    result_string,
    ruleset="Chinese",
    komi=7.5,
    white_name=PROGRAM_IDENTIFIER,
    black_name=PROGRAM_IDENTIFIER,
    comments=[]
):
    """Turn a game into SGF.

    Doesn't handle handicap games or positions with incomplete history.

    Args:
        move_history: iterable of PlayerMoves
        result_string: "B+R", "W+0.5", etc.
        comments: iterable of string/None. Will be zipped with move_history.
    """
    boardsize = go.N
    game_moves = ''.join(translate_sgf_move(*z)
                         for z in itertools.zip_longest(move_history, comments))
    result = result_string
    return SGF_TEMPLATE.format(**locals())


def sgf_prop(value_list):
    'Converts raw sgf library output to sensible value'
    if value_list is None:
        return None
    if len(value_list) == 1:
        return value_list[0]
    else:
        return value_list


def sgf_prop_get(props, key, default):
    return sgf_prop(props.get(key, default))


def handle_node(pos, node):
    'A node can either add B+W stones, play as B, or play as W.'
    props = node.properties
    black_stones_added = [coords.from_sgf(
        c) for c in props.get('AB', [])]
    white_stones_added = [coords.from_sgf(
        c) for c in props.get('AW', [])]
    if black_stones_added or white_stones_added:
        return add_stones(pos, black_stones_added, white_stones_added)
    # If B/W props are not present, then there is no move. But if it is present and equal to the empty string, then the move was a pass.
    elif 'B' in props:
        black_move = coords.from_sgf(props.get('B', [''])[0])
        return pos.play_move(black_move, color=go.BLACK)
    elif 'W' in props:
        white_move = coords.from_sgf(props.get('W', [''])[0])
        return pos.play_move(white_move, color=go.WHITE)
    else:
        return pos


def add_stones(pos, black_stones_added, white_stones_added):
    working_board = np.copy(pos.board)
    go.place_stones(working_board, go.BLACK, black_stones_added)
    go.place_stones(working_board, go.WHITE, white_stones_added)
    new_position = Position(board=working_board, n=pos.n, komi=pos.komi,
                            caps=pos.caps, ko=pos.ko, recent=pos.recent, to_play=pos.to_play)
    return new_position


def get_next_move(node):
    props = node.next.properties
    if 'W' in props:
        return coords.from_sgf(props['W'][0])
    elif 'B' in props:
        return coords.from_sgf(props['B'][0])
    else:
        raise KeyError('node has no B/W property')


def maybe_correct_next(pos, next_node):
    if (('B' in next_node.properties and not pos.to_play == go.BLACK) or
            ('W' in next_node.properties and not pos.to_play == go.WHITE)):
        pos.flip_playerturn(mutate=True)


def get_sgf_root_node(sgf_contents):
    collection = sgf.parse(sgf_contents)
    game = collection.children[0]
    return game.root


def replay_sgf(sgf_contents):
    """Wrapper for sgf files, returning go.PositionWithContext instances.

    It does NOT return the very final position, as there is no follow up.
    To get the final position, call pwc.position.play_move(pwc.next_move)
    on the last PositionWithContext returned.

    Example usage:
    with open(filename) as f:
        for position_w_context in replay_sgf(f.read()):
            print(position_w_context.position)
    """
    root_node = get_sgf_root_node(sgf_contents)
    props = root_node.properties
    assert int(sgf_prop(props.get('GM', ['1']))) == 1, "Not a Go SGF!"

    komi = 0
    if props.get('KM') is not None:
        komi = float(sgf_prop(props.get('KM')))
    result = utils.parse_game_result(sgf_prop(props.get('RE', '')))

    pos = Position(komi=komi)
    current_node = root_node
    while pos is not None and current_node.next is not None:
        if len(current_node.next.properties) == 0:
            # old sgf may have an empty node at the end
            assert current_node.next.next is None
            break
        pos = handle_node(pos, current_node)
        maybe_correct_next(pos, current_node.next)
        next_move = get_next_move(current_node)
        yield PositionWithContext(pos, next_move, result)
        current_node = current_node.next


def replay_sgf_file(sgf_file: str):
    with open(sgf_file) as f:
        s = f.read()
        for pwc in replay_sgf(s):
            yield pwc


class SGFReader(object):
    UNKNOWN_MARGIN = 1000
    MISSING_MARGIN = 2000

    def __init__(self, sgf_contents, name='unknown'):
        self.root_node = get_sgf_root_node(sgf_contents)
        self.props = self.root_node.properties
        self.name = name

        assert int(sgf_prop(self.props.get('GM', ['1']))) == 1, "Not a Go SGF!"

    @staticmethod
    def from_string(s) -> 'SGFReader':
        return SGFReader(s)

    @staticmethod
    def from_file_compatible(fname: str) -> 'SGFReader':
        with open(fname) as f:
            s = f.read()
            return SGFReader.from_string_compatible(s)

    @staticmethod
    def from_string_compatible(s: str, name='unknown') -> 'SGFReader':
        """
        NNGS uses older sgf format. Just need simple conversion
        """
        s = s.replace('CoPyright[', 'C[', 1)
        # this happens in only one game in Pro
        s = s.replace('MULTIGOGM[', 'C[', 1)
        # quite some --- at EOF
        s = s.replace('---', '', -1)
        return SGFReader(s, name)

    def komi(self) -> float:
        km_prop = sgf_prop(self.props.get('KM'))
        if km_prop is None:
            return 0
        komi = float(km_prop)
        return komi

    def board_size(self) -> int:
        size_str = sgf_prop(self.props.get('SZ'))
        return int(size_str)

    def black_name(self) -> str:
        return sgf_prop(self.props.get('PB'))

    def white_name(self) -> str:
        return sgf_prop(self.props.get('PW'))

    def not_handicap(self) -> bool:
        """ HA """
        handicap = sgf_prop(self.props.get('HA'))
        if handicap is None or handicap == 0:
            return True
        return False

    def num_nodes(self):
        """ rough estimates of #moves """
        current_node = self.root_node
        i = 0
        while current_node.next is not None:
            current_node = current_node.next
            i += 1
        return i

    def last_pos(self, up_to=-1, ignore_final_pass=False) -> go.Position:
        """ last pos, or a specific move # (up_to), based on iter_pwcs()
        ignore_final_pass: when a game ended with single or two passes, return the position before that
        """
        komi = self.komi()
        pos = Position(komi=komi)
        last_pos_before_pass = pos
        current_node = self.root_node
        i = 0
        while pos is not None and current_node is not None:
            if 0 <= up_to and pos.n == up_to:
                return pos
            try:
                pos = handle_node(pos, current_node)
                if current_node.next is not None:
                    maybe_correct_next(pos, current_node.next)
                if len(pos.recent) >= 1 and pos.recent[-1].move is not None:
                    last_pos_before_pass = pos
            except:
                logging.exception(f'{self.name} failed iter thru game: step {i}')
                break
            current_node = current_node.next
            i += 1

        if ignore_final_pass:
            return last_pos_before_pass
        else:
            return pos

    def iter_pwcs(self) -> Iterable[PositionWithContext]:
        """ based on replay_sgf: result is black margin now """
        komi = self.komi()
        result = self.black_margin_adj(adjust_komi=True)
        if result is None:
            result = self.MISSING_MARGIN

        pos = Position(komi=komi)
        current_node = self.root_node
        i = 0
        while pos is not None and current_node.next is not None:
            if len(current_node.next.properties) == 0:
                # old sgf may have an empty node at the end
                assert current_node.next.next is None
                break
            try:
                pos = handle_node(pos, current_node)
                maybe_correct_next(pos, current_node.next)
                next_move = get_next_move(current_node)
                yield PositionWithContext(pos, next_move, result)
            except:
                logging.exception(f'{self.name} failed iter thru game: step {i}')
                break
            current_node = current_node.next
            i += 1

    @staticmethod
    def _parse_result_str(s: str) -> Tuple[int, float]:
        """ B+R B+2.5 B+T   DRAW """
        s = s.upper()
        if s == 'DRAW':
            return 0, 0

        winner = s[0]
        result_sign = 0
        if winner == 'B':
            result_sign = 1
        elif winner == 'W':
            result_sign = -1

        assert s[1] == '+'
        detail = s[2:]
        if 'A' <= detail[0] <= 'Z':
            # most likely Resign or Time
            return result_sign, SGFReader.UNKNOWN_MARGIN
        return result_sign, float(detail)

    def result_str(self) -> Optional[str]:
        result_str = sgf_prop(self.props.get('RE'))
        return result_str

    def result(self) -> int:
        """ minigo game result: 0 mean no RE """
        result_str = sgf_prop(self.props.get('RE'))
        if result_str is None:
            return 0
        result_sign, margin = self._parse_result_str(result_str)
        return result_sign

    def black_margin_adj(self, adjust_komi=False) -> Optional[float]:
        """ winning margin for black. When adjusted for komi, it's the margin when komi=0

        - no RE record: None
        - R|T: +/-UNKNOWN_MARGIN
          Resign can be either large or small, I'd guess large in general.
        -
        """
        result_str = sgf_prop(self.props.get('RE'))
        if result_str is None:
            return None

        result_sign, margin = self._parse_result_str(result_str)
        black_margin = result_sign * margin

        # adjust for komi
        if adjust_komi and margin != self.UNKNOWN_MARGIN:
            black_margin += self.komi()

        return black_margin
