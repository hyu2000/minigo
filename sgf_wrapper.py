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
import re
from typing import Tuple, Optional, Iterable, List

import numpy as np
import itertools

import coords
import go
from go import Position, PositionWithContext, PlayerMove
import utils
import sgf
from absl import logging


SGF_TEMPLATE = '''(;GM[1]FF[4]CA[UTF-8]AP[Minigo_sgfgenerator]RU[{ruleset}]
SZ[{boardsize}]{add_blacks}{add_whites}KM[{komi}]PW[{white_name}]PB[{black_name}]RE[{result}]C[{game_comment}]
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
    move_history: Iterable,
    result_string,
    ruleset="Chinese",
    komi=7.5,
    black_setup_stones=None,
    white_setup_stones=None,
    white_name=PROGRAM_IDENTIFIER,
    black_name=PROGRAM_IDENTIFIER,
    game_comment=' ',
    comments=[]
):
    """Turn a game into SGF.

    Doesn't handle handicap games or positions with incomplete history.

    Args:
        move_history: iterable of PlayerMoves
        result_string: "B+R", "W+0.5", etc.
        game_comment: comment for the entire game
        comments: iterable of string/None. Will be zipped with move_history.
    """
    assert len(game_comment) > 0   # wgo won't show *any* comment if it's empty. ' ' will do
    boardsize = go.N
    add_blacks = ''
    if black_setup_stones:
        add_blacks = 'AB' + ''.join(f'[{coords.to_sgf(x)}]' for x in black_setup_stones)
    add_whites = ''
    if white_setup_stones:
        add_whites = 'AW' + ''.join(f'[{coords.to_sgf(x)}]' for x in white_setup_stones)
    game_moves = ''.join(translate_sgf_move(*z)
                         for z in itertools.zip_longest(move_history, comments))
    result = result_string
    return SGF_TEMPLATE.format(**locals())


def make_sgf_from_gtp_moves(gtp_moves: List[str], result: int,
                            ruleset="Chinese",
                            komi=7.5,
                            white_name='Unknown',
                            black_name='Unknown',
                            game_comment=' ',
                            comments=[]
                            ) -> str:
    """ convenience method: convert gtp moves to a sgf """
    player_moves = [PlayerMove(go.BLACK if i % 2 == 0 else go.WHITE, coords.from_gtp(gtp_move))
                    for i, gtp_move in enumerate(gtp_moves)]
    result_string = 'B+R' if result > 0 else ('W+R' if result < 0 else 'B+T')
    return make_sgf(player_moves, result_string, ruleset, komi, white_name, black_name,
                    game_comment=game_comment, comments=comments)


def make_sgf_from_move_str(
    sgf_move_str: str,
    result_string,
    ruleset="Chinese",
    komi=7.5,
    white_name=PROGRAM_IDENTIFIER,
    black_name=PROGRAM_IDENTIFIER,
    game_comment=' ',
    comments=[]
):
    """ convenience method: add headers to sgf_move_str """
    boardsize = go.N
    if sgf_move_str[0] == ';':
        game_moves = sgf_move_str
    else:
        game_moves = f';{sgf_move_str}'
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


def handle_node(pos, node) -> go.Position:
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
    return extract_move_and_comments(props)


def extract_move_and_comments(props: dict):
    comments = props.get('C')
    if 'W' in props:
        return coords.from_sgf(props['W'][0]), comments
    elif 'B' in props:
        return coords.from_sgf(props['B'][0]), comments
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
        next_move, _ = get_next_move(current_node)
        yield PositionWithContext(pos, next_move, result)
        current_node = current_node.next


class VariationTraverser:
    """ traverse all paths in an sgf """
    def __init__(self, path_handler=None):
        self.handle_path = path_handler or self.default_path_handler

    class PathCounter:
        """ a simple path handler class """
        def __init__(self):
            self.num_paths = 0
            self.num_correct_paths = 0

        def path_handler(self, history, comments):
            self.num_paths += 1
            is_correct = 'correct' in comments[0].lower()
            if is_correct:
                self.num_correct_paths += 1

    @staticmethod
    def default_path_handler(history: tuple, comments: List[str]):
        """ this gets called when a variation ends """
        is_correct = 'correct' in comments[0].lower()
        path = ' '.join(coords.to_gtp(x) for x in history)
        print(f'reached leaf: path: {path} {is_correct}')

    def visit_node(self, node, history: tuple):
        """ recursive """
        next_move, comments = extract_move_and_comments(node.properties)
        history = history + (next_move,)
        if node.next is None:
            self.handle_path(history, comments)
            return

        # depth-first
        # first visit the main path. Note node.variations is empty when there is only a main path
        self.visit_node(node.next, history)
        # then visit the other variations. Note when there are multiple variations, main path is listed as one of them
        for var in node.variations[1:]:
            self.visit_node(var, history)

    def traverse(self, sgf_contents):
        collection = sgf.parse(sgf_contents)
        assert len(collection.children) == 1
        gtree0 = collection.children[0]

        for i, gtree in enumerate(gtree0.children):
            node = gtree.root
            self.visit_node(node, ())


def replay_sgf_file(sgf_fname: str):
    with open(sgf_fname) as f:
        s = f.read()
        for pwc in replay_sgf(s):
            yield pwc


def add_init_stones_file(sgf_fname, black_coords, white_coords, new_fname):
    with open(sgf_fname, 'r') as f:
        sgf_str = f.read()
        new_sgf_str = add_init_stones(sgf_str, black_coords, white_coords)
    with open(new_fname, 'w') as f:
        f.write(new_sgf_str)


def add_init_stones(sgf_str, black_coords: List, white_coords: List) -> str:
    """ quick hack to add more setup stones to sgf str
    This assumes AB|AW tag already exists
    """
    # we try to limit where replace happens: typically there is a comment at the root node,
    # there might be other tags before C
    pattern = re.compile(r'(A[BW].+?)C\[')
    # match = re.search(pattern, sgf_str)
    # if not match:
    #     logging.warning('no setup stones found!')
    #     return sgf_str

    ab_str = ''.join('[%s]' % coords.to_sgf(x) for x in black_coords)
    aw_str = ''.join('[%s]' % coords.to_sgf(x) for x in white_coords)

    def process_match(match):
        s = match.group(0)
        # add new stones to the beginning
        nstr = s.replace('AB', f'AB{ab_str}')
        nstr = nstr.replace('AW', f'AW{aw_str}')
        return nstr

    new_sgf = re.sub(pattern, process_match, sgf_str, count=1)
    return new_sgf


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
            return SGFReader.from_string_compatible(s, name=fname)

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

    def black_init_stones(self) -> List:
        ss = sgf_prop(self.props.get('AB'))
        if not ss:
            return []
        return [coords.from_sgf(s) for s in ss]

    def white_init_stones(self) -> List:
        ss = sgf_prop(self.props.get('AW'))
        if not ss:
            return []
        return [coords.from_sgf(s) for s in ss]

    def root_comments(self) -> List[str]:
        """ a node can have multiple comments """
        return self.props.get('C')

    def player_to_start(self) -> int:
        player = sgf_prop(self.props.get('PL'))
        if not player:
            return go.BLACK
        letter = player[0].upper()
        if letter == 'B':
            return go.BLACK
        elif letter == 'W':
            return go.WHITE
        else:
            return 0

    def not_handicap(self) -> bool:
        """ HA """
        handicap = sgf_prop(self.props.get('HA'))
        if handicap is None or int(handicap) == 0:
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

    def first_pos(self) -> go.Position:
        """ for puzzles w/ HB|HW setup
        """
        # self.last_pos(up_to=0) doesn't work, why?

        for pwc in self.iter_pwcs():
            break
        pos = pwc.position
        pos.to_play = self.player_to_start()
        return pos

    def iter_comments(self) -> Iterable[Tuple[str, List[str]]]:
        """ this provides access to node comments

        Note comments can have multiple sections:
        ;W[ff]C[comment1][comment2]  -> 'F4', ['comment1', 'comment2']

        See test_sgf_wrapper for examples
        """
        current_node = self.root_node
        while current_node.next is not None:
            move, comments = get_next_move(current_node)
            yield coords.to_gtp(move), comments
            current_node = current_node.next

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
                next_move, _ = get_next_move(current_node)
                yield PositionWithContext(pos, next_move, result)
            except GeneratorExit:
                break
            except:
                logging.exception(f'{self.name} failed iter thru game: step {i}')
                break
            current_node = current_node.next
            i += 1

    @staticmethod
    def _parse_result_str(s: str) -> Tuple[int, float]:
        """ B+R B+2.5 B+T   DRAW|0 """
        s = s.upper()
        if s == 'DRAW' or s == '0' or s == 'VOID':
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
