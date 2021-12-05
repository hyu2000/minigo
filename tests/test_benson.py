import numpy as np
import coords
import go
from go import LibertyTracker, BensorAnalyzer
from sgf_wrapper import SGFReader
from tests import test_utils
import myconf
from tests.test_go import coords_from_gtp_set


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

    def test_benson_basic(self):
        board = test_utils.load_board('''
            .XX......
            X..X.....
            XXXX..O..
        ''' + self.EMPTY_ROW * 6)
        for color in (go.BLACK,):
            tracker = BensorAnalyzer.from_board(board, color)
            assert len(tracker.regions) == 3
            for rid, region in tracker.regions.items():
                print(rid, region.color, len(region.stones), len(region.liberties), region.chains)

            chain_ids, _ = tracker.eliminate()
            assert len(chain_ids) == 2
            print('pass-alive chains: ', chain_ids)

    def test_not_pass_alive(self):
        """ https://senseis.xmp.net/?BensonsAlgorithm  central case
        two black chains, two eyes: one vital to both, one (two spaces) vital to one only
        With either black or white(!) in one of the two spaces, it becomes vital
        """
        board = test_utils.load_board('''
            .OXXX.XO.
            .OX..XXO.
            .OXXXOOO.
            .OOOOO...
        ''' + self.EMPTY_ROW * 5)

        tracker = BensorAnalyzer.from_board(board, go.BLACK)
        assert len(tracker.regions) == 3
        for rid, region in tracker.regions.items():
            print(rid, region.color, len(region.stones), len(region.liberties), region.chains)

        chain_ids, _ = tracker.eliminate()
        assert len(chain_ids) == 0

        # however, if we put any stone in row 1, col 3, black is pass-alive
        for color in (go.BLACK, go.WHITE):
            board[1][3] = color
            tracker = BensorAnalyzer.from_board(board, go.BLACK)
            assert len(tracker.regions) == 3
            chain_ids = tracker.eliminate()
            assert len(chain_ids) == 2

    def test_pass_alive2(self):
        """ https://senseis.xmp.net/?BensonsAlgorithm  bottom case
        """
        board = test_utils.load_board('''
            OXX.X.X..
            OX.XOXXXX
            OXXOOOOOO
            OOOO.....
        ''' + self.EMPTY_ROW * 5)

        tracker = BensorAnalyzer.from_board(board, go.BLACK)
        assert len(tracker.regions) == 5
        for rid, region in tracker.regions.items():
            print(rid, region.color, len(region.stones), len(region.liberties), region.chains)

        chain_ids, _ = tracker.eliminate()
        assert len(chain_ids) == 4

    def test_benson_real1(self):
        """
        """
        fname = '1-61704860349.sgf'
        fname = '1-61717098200.sgf'     # black all pass-alive, two white chains: one alive, one not
        fname = '1-61717672696.sgf'     # big regions, nothing pass-alive
        fname = '2-61736655674.sgf'     # dead stone within pass-alive region can be safely removed
        fname = '2-61758327600.sgf'     # here dead stone removal really helps score
        fpath = f'{myconf.EXP_HOME}/selfplay17.300/sgf/full/{fname}'
        fpath = f'{myconf.EXP_HOME}/selfplay/sgf/full/2015-01-07T01:56:00.051Z_ho6o2gojvb9g-8369170389.sgf'
        reader = SGFReader.from_file_compatible(fpath)
        pos = reader.last_pos(ignore_final_pass=True)
        board = pos.board

        for color in (go.BLACK, go.WHITE):
            print(f'\nRunning for {color}')
            tracker = BensorAnalyzer.from_board(board, color)
            chain_ids, regions = tracker.eliminate()
            assert len(chain_ids) >= 0
            for chain_idx in chain_ids:
                group = tracker.lib_tracker.groups[chain_idx]
                stone0 = next(iter(group.stones))
                print(f'chain {chain_idx}: {group.color}  %d stones, %d liberties: %s' % (
                    len(group.stones), len(group.liberties), coords.to_gtp(stone0)))
            for region in regions:
                num_opp_stones = len(region.stones) - len(region.liberties)
                print(f'region {region.id}: {region.color}  size=%d, %d opp stones' % (
                    len(region.stones), num_opp_stones))


