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
import glob
import os.path
import re
import unittest
import go
from absl import logging

from sgf_wrapper import (replay_sgf, translate_sgf_move, make_sgf, make_sgf_from_gtp_moves, SGFReader,
                         add_init_stones, VariationTraverser)

import coords
from tests import test_utils

logging.set_verbosity(logging.INFO)

JAPANESE_HANDICAP_SGF = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Japanese]SZ[9]HA[2]RE[Void]KM[5.50]PW[test_white]PB[test_black]AB[gc][cg];W[ee];B[dg])"

CHINESE_HANDICAP_SGF = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[9]HA[2]RE[Void]KM[5.50]PW[test_white]PB[test_black]RE[B+39.50];B[gc];B[cg];W[ee];B[gg];W[eg];B[ge];W[ce];B[ec];W[cc];B[dd];W[de];B[cd];W[bd];B[bc];W[bb];B[be];W[ac];B[bf];W[dh];B[ch];W[ci];B[bi];W[di];B[ah];W[gh];B[hh];W[fh];B[hg];W[gi];B[fg];W[dg];B[ei];W[cf];B[ef];W[ff];B[fe];W[bg];B[bh];W[af];B[ag];W[ae];B[ad];W[ae];B[ed];W[db];B[df];W[eb];B[fb];W[ea];B[fa])"

NO_HANDICAP_SGF = "(;CA[UTF-8]SZ[9]PB[Murakawa Daisuke]PW[Iyama Yuta]KM[6.5]HA[0]RE[W+1.5]GM[1];B[fd];W[cf];B[eg];W[dd];B[dc];W[cc];B[de];W[cd];B[ed];W[he];B[ce];W[be];B[df];W[bf];B[hd];W[ge];B[gd];W[gg];B[db];W[cb];B[cg];W[bg];B[gh];W[fh];B[hh];W[fg];B[eh];W[ei];B[di];W[fi];B[hg];W[dh];B[ch];W[ci];B[bh];W[ff];B[fe];W[hf];B[id];W[bi];B[ah];W[ef];B[dg];W[ee];B[di];W[ig];B[ai];W[ih];B[fb];W[hi];B[ag];W[ab];B[bd];W[bc];B[ae];W[ad];B[af];W[bd];B[ca];W[ba];B[da];W[ie])"


class TestSgfGeneration(test_utils.MinigoUnitTest):
    def test_translate_sgf_move(self):
        self.assertEqual(
            ";B[db]",
            translate_sgf_move(go.PlayerMove(go.BLACK, (1, 3)), None))
        self.assertEqual(
            ";W[aa]",
            translate_sgf_move(go.PlayerMove(go.WHITE, (0, 0)), None))
        self.assertEqual(
            ";W[]",
            translate_sgf_move(go.PlayerMove(go.WHITE, None), None))
        self.assertEqual(
            ";B[db]C[comment]",
            translate_sgf_move(go.PlayerMove(go.BLACK, (1, 3)), "comment"))

    def test_make_sgf(self):
        all_pwcs = list(replay_sgf(NO_HANDICAP_SGF))
        second_last_position, last_move, _ = all_pwcs[-1]
        last_position = second_last_position.play_move(last_move)

        back_to_sgf = make_sgf(
            last_position.recent,
            last_position.score(),
            komi=last_position.komi,
        )
        reconstructed_positions = list(replay_sgf(back_to_sgf))
        second_last_position2, last_move2, _ = reconstructed_positions[-1]
        last_position2 = second_last_position2.play_move(last_move2)

        self.assertEqualPositions(last_position, last_position2)

    def test_make_sgf_from_gtp_moves(self):
        gtp_moves = 'G6 C4 D4 D5 C3 E4 D3 F6 F7 G7 G8 H7 E6 F5 E8 D6 H6 E3 C5 B5 B4 C6 G4 B2 B6 C4 E2 F2 C5 A5 A4 C4 F3 D2 C5 G5 H5 C4 C2 E1 C5 H4 H3 C4 E7 C5 E5 C8 C7 D7 F4 E2 D8 B7 G2 C1 C9 B9 G1 B3 D9 B8 F1 A3 F6 D4 H8 A1 G9 pass J9 pass J6 pass G5 pass H2 A9 J8 pass G3 pass F5 pass H9 pass J7 pass H1 pass J3 B4 J4 A6 H7 C3 F8 pass J2 pass B1'
        gtp_moves = 'C4 G6 F6 F5 G7 E6 F7 D4 D3 C3 E4 D5 E3 E5 B4 C2 D2 B3 C6 C7 B7 D7 H6 F4 F3 G3 G2 H2 F2 H5 E8 B1 A2 B6 B5 A6 A7 C5 A5 A3 D1 B8 C1 B6 A8 C8 A6 G5 A4 H7 H8 J6 J8 E7 F8 E9 F9 D9 H9 J7 B2 H4 H1 D6 G1 J2 F1 B6 B9 C9 A9 C3 J1 E1 G8 B3 A3 D8 E2 H3 C2 B3 C3 pass B3 G4 A1 J5 pass J3 pass B1'

        sgf_str = make_sgf_from_gtp_moves(gtp_moves.split(), 1)
        print(sgf_str)
        with open('/Users/hyu/Downloads/make_sgf_from_gtp.4visits.sgf', 'w') as f:
            f.write(sgf_str)

    def test_make_sgf_with_setup_stones(self):
        """ does AB|AW allow empty cases: AB[]AW[]: Yes """
        pass

    def test_parse_LnD(self):
        """ understand how sgf node works: variation """
        # too hard to parse by hand
        sgf_str = """
        (;FF[4]CA[UTF-8]AP[puzzle2sgf:0.1]GM[1]GN[総合問題１０級]SZ[9]AB[he][ge][gd][gc][fb][fa][ec]AW[ib][gb][ga][hc][hd]PL[B]C[黒先白死：Black to kill

        コウやセキは失敗：Kou and Seki are failures
        ](
            ;B[ha];W[hb] (
            ;B[id]C[CORRECT];W[ic] (;B[ia]C[CORRECT]) (;B[ie];W[ia]C[CORRECT];B[ha]C[CORRECT]) 
            ) (
            ;B[ic];W[id]C[WRONG]
            ) (
            ;B[ie];W[id]C[WRONG]
            )
        ) (
            ;B[id];W[ha]C[WRONG] (;B[ie];W[ic]C[WRONG]) (;B[ic];W[ie]C[WRONG](;B[id];W[ic]C[WRONG])(;B[ic];W[id]C[WRONG])))(;B[hb];W[ha]C[WRONG])(;B[ic];W[id]C[WRONG])(;B[ia];W[ha]C[WRONG])(;B[ie];W[id]C[WRONG])
        )        
        """
        for pwc in replay_sgf(sgf_str):
            print(pwc)

    def test_modify_LnD_puzzle_dev(self):
        """ change init board setup, but keep game tree """
        # sgf.py not super clear
        sgf_str = """
        (;FF[4]CA[UTF-8]AP[puzzle2sgf:0.1]GM[1]GN[総合問題１０級]SZ[9]AB[he][ge][gd][gc][fb][fa][ec]AW[ib][gb][ga][hc][hd]PL[B]C[黒先白死：Black to kill

        コウやセキは失敗：Kou and Seki are failures
        ](;B[ha];W[hb](;B[id]C[CORRECT];W[ic](;B[ia]C[CORRECT])(;B[ie];W[ia]C[CORRECT];B[ha]C[CORRECT]))(;B[ic];W[id]C[WRONG])(;B[ie];W[id]C[WRONG]))(;B[id];W[ha]C[WRONG](;B[ie];W[ic]C[WRONG])(;B[ic];W[ie]C[WRONG](;B[id];W[ic]C[WRONG])(;B[ic];W[id]C[WRONG])))(;B[hb];W[ha]C[WRONG])(;B[ic];W[id]C[WRONG])(;B[ia];W[ha]C[WRONG])(;B[ie];W[id]C[WRONG]))        
        """
        pattern = r'(A[BW].+?)C\['  # assume C immediately follow AB|AW
        match = re.search(pattern, sgf_str)
        print(match)
        substr = match.group(1)
        # verify

        add_blacks = 'C7 E5'
        add_whites = 'B6 C5 D4'
        ab_str = ''.join('[%s]' % coords.to_sgf(coords.from_gtp(x)) for x in add_blacks.split())
        aw_str = ''.join('[%s]' % coords.to_sgf(coords.from_gtp(x)) for x in add_whites.split())
        # add new stones to the beginning
        nstr = substr.replace('AB', f'AB{ab_str}')
        nstr = nstr.replace('AW', f'AW{aw_str}')
        print(nstr)

        # another way
        def process_match(match):
            s = match.group(0)
            nstr = s.replace('AB', f'AB{ab_str}')
            nstr = nstr.replace('AW', f'AW{aw_str}')
            return nstr
        new_sgf = re.sub(pattern, process_match, sgf_str)
        print(new_sgf)
        assert len(new_sgf) == len(sgf_str) + len(ab_str) + len(aw_str)

    def test_modify_LnD_puzzle(self):
        """ change init board setup, but keep game tree """
        # sgf.py not super clear
        sgf_str = """
        (;FF[4]CA[UTF-8]AP[puzzle2sgf:0.1]GM[1]GN[総合問題１０級]SZ[9]AB[he][ge][gd][gc][fb][fa][ec]AW[ib][gb][ga][hc][hd]PL[B]C[黒先白死：Black to kill

        コウやセキは失敗：Kou and Seki are failures
        ](;B[ha];W[hb](;B[id]C[CORRECT];W[ic](;B[ia]C[CORRECT])(;B[ie];W[ia]C[CORRECT];B[ha]C[CORRECT]))(;B[ic];W[id]C[WRONG])(;B[ie];W[id]C[WRONG]))(;B[id];W[ha]C[WRONG](;B[ie];W[ic]C[WRONG])(;B[ic];W[ie]C[WRONG](;B[id];W[ic]C[WRONG])(;B[ic];W[id]C[WRONG])))(;B[hb];W[ha]C[WRONG])(;B[ic];W[id]C[WRONG])(;B[ia];W[ha]C[WRONG])(;B[ie];W[id]C[WRONG]))        
        """
        add_blacks = 'C7 E5'
        add_whites = 'B6 C5 D4'
        black_coords = (coords.from_gtp(x) for x in add_blacks.split())
        white_coords = (coords.from_gtp(x) for x in add_whites.split())
        new_sgf = add_init_stones(sgf_str, black_coords, white_coords)
        with open('/Users/hyu/Downloads/test_modify_sgf.sgf', 'w') as f:
            f.write(new_sgf)


class TestSgfWrapper(test_utils.MinigoUnitTest):
    def test_sgf_props(self):
        sgf_replayer = replay_sgf(CHINESE_HANDICAP_SGF)
        initial = next(sgf_replayer)
        self.assertEqual(go.BLACK, initial.result)
        self.assertEqual(5.5, initial.position.komi)

    def test_japanese_handicap_handling(self):
        intermediate_board = test_utils.load_board('''
            .........
            .........
            ......X..
            .........
            ....O....
            .........
            ..X......
            .........
            .........
        ''')
        intermediate_position = go.Position(
            intermediate_board,
            n=1,
            komi=5.5,
            caps=(0, 0),
            recent=(go.PlayerMove(go.WHITE, coords.from_gtp('E5')),),
            to_play=go.BLACK,
        )
        final_board = test_utils.load_board('''
            .........
            .........
            ......X..
            .........
            ....O....
            .........
            ..XX.....
            .........
            .........
        ''')
        final_position = go.Position(
            final_board,
            n=2,
            komi=5.5,
            caps=(0, 0),
            recent=(go.PlayerMove(go.WHITE, coords.from_gtp('E5')),
                    go.PlayerMove(go.BLACK, coords.from_gtp('D3')),),
            to_play=go.WHITE,
        )
        positions_w_context = list(replay_sgf(JAPANESE_HANDICAP_SGF))
        self.assertEqualPositions(
            intermediate_position, positions_w_context[1].position)
        final_replayed_position = positions_w_context[-1].position.play_move(
            positions_w_context[-1].next_move)
        self.assertEqualPositions(final_position, final_replayed_position)

    def test_chinese_handicap_handling(self):
        intermediate_board = test_utils.load_board('''
            .........
            .........
            ......X..
            .........
            .........
            .........
            .........
            .........
            .........
        ''')
        intermediate_position = go.Position(
            intermediate_board,
            n=1,
            komi=5.5,
            caps=(0, 0),
            recent=(go.PlayerMove(go.BLACK, coords.from_gtp('G7')),),
            to_play=go.BLACK,
        )
        final_board = test_utils.load_board('''
            ....OX...
            .O.OOX...
            O.O.X.X..
            .OXXX....
            OX...XX..
            .X.XXO...
            X.XOOXXX.
            XXXO.OOX.
            .XOOX.O..
        ''')
        final_position = go.Position(
            final_board,
            n=50,
            komi=5.5,
            caps=(7, 2),
            ko=None,
            recent=(go.PlayerMove(go.WHITE, coords.from_gtp('E9')),
                    go.PlayerMove(go.BLACK, coords.from_gtp('F9')),),
            to_play=go.WHITE
        )
        positions_w_context = list(replay_sgf(CHINESE_HANDICAP_SGF))
        self.assertEqualPositions(
            intermediate_position, positions_w_context[1].position)
        self.assertEqual(
            positions_w_context[1].next_move, coords.from_gtp('C3'))
        final_replayed_position = positions_w_context[-1].position.play_move(
            positions_w_context[-1].next_move)
        self.assertEqualPositions(final_position, final_replayed_position)

    @staticmethod
    def node_comment(node):
        comments = node.properties.get('C')
        if comments is None:
            return None
        return comments[0]

    def test_visit_sgf(self):
        """ exploratory: sgf data structure
        """
        import sgf
        # How to Play Go +/Life\ and\ Death\ 2.2
        sgf_contents = """
(;FF[4]CA[UTF-8]AP[puzzle2sgf:0.1]GM[1]GN[Life and Death 2.2]SZ[9]
  AB[fi][fh][gg][hf][if][fg][ge]AW[gh][gi][hh][hg][ig]PL[B]C[In this position, White also has three spaces to make eyes in, only this time, it's in the corner. Can Black find a way to kill White?]
    (;B[ih]C[V1] (;W[ii]C[WRONG]) (;W[aa];B[ab];C[very wrong]))
    (;B[hi]C[V2];W[ii]C[WRONG])
    (;B[ii]C[V3,CORRECT])
)
"""

        collection = sgf.parse(sgf_contents)
        assert len(collection.children) == 1
        gtree = collection.children[0]
        assert len(gtree.children) == 3   # GameTree
        root = gtree.root
        assert self.node_comment(root).startswith('In this pos')
        assert len(root.variations) == 3  # Nodes, probably not useful
        for i, gtree in enumerate(gtree.children):
            node = gtree.root
            print(f'{i} %s #vars=%d' % (self.node_comment(node), len(node.variations)))
            while node is not None:
                print(f'\t%s #vars=%d' % (self.node_comment(node), len(node.variations)))
                node = node.next

    def test_traverse_sgf(self):
        """ traverse a puzzle sgf, visit all paths
        """
        # How to Play Go +/Life\ and\ Death\ 2.2
        sgf_contents = """
        (;FF[4]CA[UTF-8]AP[puzzle2sgf:0.1]GM[1]GN[Life and Death 2.2]SZ[9]
          AB[fi][fh][gg][hf][if][fg][ge]AW[gh][gi][hh][hg][ig]PL[B]C[In this position, White also has three spaces to make eyes in, only this time, it's in the corner. Can Black find a way to kill White?]
            (;B[ih]C[V1] (;W[ii]C[WRONG1]) (;W[aa];B[ab]C[very wrong]))
            (;B[hi]C[V2];W[ii]C[WRONG2])
            (;B[ii]C[V3,CORRECT])
        )
        """
        cnter = VariationTraverser.PathCounter()
        traverser = VariationTraverser(cnter.path_handler)
        traverser.traverse(sgf_contents)
        # traverser.traverse_sgf('/Users/hyu/PycharmProjects/dlgo/puzzles9x9/How to Play Go +/Life and Death 2.1.sgf')
        print(f'correct/total = {cnter.num_correct_paths} / {cnter.num_paths}')

    def test_puzzle_set_stats(self):
        """ count num correct solutions in a puzzle """
        glob_pattern = f'/Users/hyu/PycharmProjects/dlgo/puzzles9x9/How to Play Go +/Life and Death*.sgf'

        cnter = VariationTraverser.PathCounter()
        traverser = VariationTraverser(cnter.path_handler)
        for sgf_fname in glob.glob(glob_pattern):
            basename = os.path.basename(sgf_fname)
            cnter.clear()
            traverser.traverse_sgf(sgf_fname)
            print(f'{basename}  correct/total = {cnter.num_correct_paths} / {cnter.num_paths}')


class TestReader(test_utils.MinigoUnitTest):
    """ """
    def test_good_sgf(self):
        sgf = '/Users/hyu/Downloads/Minigo/1.sgf'
        reader = SGFReader.from_file_compatible(sgf)
        assert reader.board_size() == 9
        assert reader.not_handicap()
        self.assertEqual(-1, reader.result())
        self.assertEqual(6.5, reader.komi())
        self.assertEqual(-reader.UNKNOWN_MARGIN, reader.black_margin_adj())
        self.assertEqual(-reader.UNKNOWN_MARGIN, reader.black_margin_adj(adjust_komi=True))

        pwcs = [x for x in reader.iter_pwcs()]
        logging.info('Found %d moves(pwc)', len(pwcs))

    def test_NNGS(self):
        # 'CoPyright' and illegal move
        # sgf = '/Users/hyu/PycharmProjects/dlgo/9x9/games/tmp/jrd-manyfaces-07-20-17'
        # 'CoPyright'; all legal moves; empty node at the end (common in NNGS)
        sgf = '/Users/hyu/PycharmProjects/dlgo/9x9/games/tmp/jrd-tromp-07-17-29'

        reader = SGFReader.from_file_compatible(sgf)
        assert reader.board_size() == 9
        assert reader.not_handicap()
        self.assertEqual(1, reader.result())
        self.assertEqual(5.5, reader.komi())
        self.assertEqual(9.5, reader.black_margin_adj())
        self.assertEqual(15, reader.black_margin_adj(adjust_komi=True))

        pwcs = [x for x in reader.iter_pwcs()]
        logging.info('Found %d moves(pwc)', len(pwcs))

    def test_NNGS_EOF(self):
        """ after the content, ignore extra '---' at EOF """
        sgf = '/Users/hyu/PycharmProjects/dlgo/9x9/games/tmp/alfalfa-angie-26-14-20'
        reader = SGFReader.from_file_compatible(sgf)
        assert reader.board_size() == 9
        assert reader.not_handicap()
        pwcs = [x for x in reader.iter_pwcs()]
        logging.info('Found %d moves(pwc)', len(pwcs))

    def test_skip_final_pass(self):
        """ """
        sgf = '/Users/hyu/PycharmProjects/dlgo/9x9/games/Pro/9x9/computer/OZ2.sgf'
        reader = SGFReader.from_file_compatible(sgf)
        last_pos = reader.last_pos()
        last_pos_ignore_pass = reader.last_pos(ignore_final_pass=True)
        assert last_pos_ignore_pass.n + 2 == last_pos.n

    def test_comments(self):
        """ """
        sgf_str = """
(;
EV[Mini-Go]
PB[Hoshikawa Takumi]
BR[1p]
PW[Yuki Satoshi]
WR[9p]
KM[6.5]
RE[W+R]
SZ[9]
GC[not broadcast]

;B[ff]C[0.5 215 path=4][comment2];W[dd]C[comment1];B[cc];W[df];B[be];W[dc];B[cb];W[gd];B[he];W[eg]
;B[fg];W[ge];B[hf];W[fh];B[gh];W[eh];B[hd];W[gc];B[gf];W[bf]
;B[ce];W[cf];B[de];W[ee];B[ef];W[db];B[cd];W[hc];B[ed];W[fe]
;B[eb];W[ec];B[da];W[fc];B[ch];W[bh])        
"""
        reader = SGFReader.from_string(sgf_str)
        for i, (move, comments) in enumerate(reader.iter_comments()):
            if i == 0:
                assert move == 'F4'
                assert len(comments) == 2 and comments[1] == 'comment2'
            elif i == 1:
                assert move == 'D6'
                assert len(comments) == 1 and comments[0] == 'comment1'
            else:
                assert comments is None
            print(i, move, comments)
        assert i == 35

    def test_read_LnD_sgf(self):
        """ see how we handle LND puzzle sgfs: AB|AW and variations
        https://homepages.cwi.nl/~aeb/go/misc/sgfnotes.html
        """
        # https://online-go.com/puzzle/67916
        sgf_str = """
(;FF[4]CA[UTF-8]AP[puzzle2sgf:0.1]GM[1]GN[総合問題１０級]SZ[9]AB[he][ge][gd][gc][fb][fa][ec]AW[ib][gb][ga][hc][hd]PL[B]C[黒先白死：Black to kill

コウやセキは失敗：Kou and Seki are failures
](;B[ha];W[hb](;B[id]C[CORRECT];W[ic](;B[ia]C[CORRECT])(;B[ie];W[ia]C[CORRECT];B[ha]C[CORRECT]))(;B[ic];W[id]C[WRONG])(;B[ie];W[id]C[WRONG]))(;B[id];W[ha]C[WRONG](;B[ie];W[ic]C[WRONG])(;B[ic];W[ie]C[WRONG](;B[id];W[ic]C[WRONG])(;B[ic];W[id]C[WRONG])))(;B[hb];W[ha]C[WRONG])(;B[ic];W[id]C[WRONG])(;B[ia];W[ha]C[WRONG])(;B[ie];W[id]C[WRONG]))        
"""
        reader = SGFReader.from_string(sgf_str)

        # AB|AW
        black_setup_stones = reader.black_init_stones()
        white_setup_stones = reader.white_init_stones()
        print('AB:', [coords.to_gtp(p) for p in black_setup_stones])
        print('AW:', [coords.to_gtp(p) for p in white_setup_stones])
        assert reader.player_to_start() == go.BLACK

        pos = reader.first_pos()
        print(pos.board)

        # SGFReader go thru main-line only
        # (;B[ha];W[hb] (;B[id]C[CORRECT];W[ic] (;B[ia]C[CORRECT]) (;B[ie];W[ia]C[CORRECT];B[ha]C[CORRECT]) ) \
        # (;B[ic];W[id]C[WRONG])(;B[ie];W[id]C[WRONG]))(;B[id];W[ha]C[WRONG](;B[ie];W[ic]C[WRONG])(;B[ic];W[ie]C[WRONG](;B[id];W[ic]C[WRONG])(;B[ic];W[id]C[WRONG])))(;B[hb];W[ha]C[WRONG])(;B[ic];W[id]C[WRONG])(;B[ia];W[ha]C[WRONG])(;B[ie];W[id]C[WRONG]))
        print('main line:')
        for i, (move, comments) in enumerate(reader.iter_comments()):
            print(i, move, comments)

    def test_read_LnD_empty_AB(self):
        """ see how we handle LND puzzle with empty AB|AW
        """
        # https://online-go.com/puzzle/67916
        sgf_str = """
(;FF[4]CA[UTF-8]AP[puzzle2sgf:0.1]GM[1]GN[総合問題１０級]SZ[9]AB[]AW[]PL[B]C[黒先白死：Black to kill

コウやセキは失敗：Kou and Seki are failures
](;B[ha];W[hb](;B[id]C[CORRECT];W[ic](;B[ia]C[CORRECT])(;B[ie];W[ia]C[CORRECT];B[ha]C[CORRECT]))(;B[ic];W[id]C[WRONG])(;B[ie];W[id]C[WRONG]))(;B[id];W[ha]C[WRONG](;B[ie];W[ic]C[WRONG])(;B[ic];W[ie]C[WRONG](;B[id];W[ic]C[WRONG])(;B[ic];W[id]C[WRONG])))(;B[hb];W[ha]C[WRONG])(;B[ic];W[id]C[WRONG])(;B[ia];W[ha]C[WRONG])(;B[ie];W[id]C[WRONG]))        
"""
        reader = SGFReader.from_string(sgf_str)

        # AB|AW
        black_setup_stones = reader.black_init_stones()
        white_setup_stones = reader.white_init_stones()
        print('AB:', [coords.to_gtp(p) for p in black_setup_stones])
        print('AW:', [coords.to_gtp(p) for p in white_setup_stones])
        assert len(black_setup_stones) == 0 == len(white_setup_stones)
        assert reader.player_to_start() == go.BLACK

        # SGFReader go thru main-line only
        # (;B[ha];W[hb] (;B[id]C[CORRECT];W[ic] (;B[ia]C[CORRECT]) (;B[ie];W[ia]C[CORRECT];B[ha]C[CORRECT]) ) \
        # (;B[ic];W[id]C[WRONG])(;B[ie];W[id]C[WRONG]))(;B[id];W[ha]C[WRONG](;B[ie];W[ic]C[WRONG])(;B[ic];W[ie]C[WRONG](;B[id];W[ic]C[WRONG])(;B[ic];W[id]C[WRONG])))(;B[hb];W[ha]C[WRONG])(;B[ic];W[id]C[WRONG])(;B[ia];W[ha]C[WRONG])(;B[ie];W[id]C[WRONG]))
        print('main line:')
        for i, (move, comments) in enumerate(reader.iter_comments()):
            print(i, move, comments)


if __name__ == '__main__':
    unittest.main()
