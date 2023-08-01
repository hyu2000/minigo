import os.path

import numpy as np
import coords
import go
from go import LibertyTracker, BensonAnalyzer
from sgf_wrapper import SGFReader
from tar_dataset import GameStore
from tests import test_utils
import myconf
from tests.test_go import coords_from_gtp_set


class TestGoScoring(test_utils.MinigoUnitTest):
    EMPTY_ROW = '.' * go.N + '\n'

    def test_dev(self):
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
        print('To fully understand, see Region definition')
        print(board)
        for color in (go.BLACK,):
            tracker = BensonAnalyzer.from_board(board, color)
            assert len(tracker.regions) == 3
            for rid, region in tracker.regions.items():
                print(f'region {rid}: color={region.color} size={len(region.stones)}, liberties={len(region.liberties)}'
                      f'  enclosing chains: {region.chains}')

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

        tracker = BensonAnalyzer.from_board(board, go.BLACK)
        # the big region outside is also considered black-enclosed!
        assert len(tracker.regions) == 3
        for rid, region in tracker.regions.items():
            print(rid, region.color, len(region.stones), len(region.liberties), region.chains)

        chain_ids, _ = tracker.eliminate()
        assert len(chain_ids) == 0

        # however, if we put any stone in row 1, col 3, black is pass-alive
        for color in (go.BLACK, go.WHITE):
            board[1][3] = color
            tracker = BensonAnalyzer.from_board(board, go.BLACK)
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

        tracker = BensonAnalyzer.from_board(board, go.BLACK)
        assert len(tracker.regions) == 5
        for rid, region in tracker.regions.items():
            print(rid, region.color, len(region.stones), len(region.liberties), region.chains)

        chain_ids, _ = tracker.eliminate()
        assert len(chain_ids) == 4

    def test_benson_score(self):
        """ in a pass-alive enclosed area, opponent could still survive
        """
        board = test_utils.load_board('''
            .O...X.X.
            O.O..XXXX
            .O...X...
            XXXXXX...
        ''' + self.EMPTY_ROW * 5)

        tracker = BensonAnalyzer.from_board(board, go.BLACK)
        assert len(tracker.regions) == 4
        for rid, region in tracker.regions.items():
            num_opp_stones = len(region.stones) - len(region.liberties)
            print(f'region {region.id}: {region.color}  size=%d, %d opp stones' % (
                len(region.stones), num_opp_stones))

        chain_ids, regions = tracker.eliminate()
        assert len(chain_ids) == 1
        print('final regions: %d' % len(regions))

        pos = go.Position(board, komi=0)
        score_tromp = pos.score_tromp()
        benson_detail = pos.score_benson()
        print('Score: Tromp=%.1f, Benson=%.1f, final=%s' % (score_tromp, benson_detail.score, benson_detail.final))
        assert score_tromp == benson_detail.score

        # but if we remove any white stone (#whites < 4), it's not considered survivable
        board[0][1] = go.EMPTY
        pos = go.Position(board, komi=0)
        score_tromp = pos.score_tromp()
        benson_detail = pos.score_benson()
        print('Score: Tromp=%.1f, Benson=%.1f, final=%s' % (score_tromp, benson_detail.score, benson_detail.final))
        assert benson_detail.score == 81
        assert score_tromp < 81

    def test_benson_score_seki(self):
        """ in a seki case, white 3 stones are live, but black is not pass-alive, so this won't affect Benson score """
        board = test_utils.load_board('''
            .OOO.XO..
            XXXXXXO..
            OOOOOOO..
            .........
        ''' + self.EMPTY_ROW * 5)
        tracker = BensonAnalyzer.from_board(board, go.BLACK)
        chain_ids, regions = tracker.eliminate()
        assert len(chain_ids) == 0

        pos = go.Position(board, komi=0)
        score_tromp = pos.score_tromp()
        benson_detail = pos.score_benson()
        print('Score: Tromp=%.1f, Benson=%.1f' % (score_tromp, benson_detail.score))
        assert score_tromp == benson_detail.score

    def test_benson_real_sgf(self):
        """
        """
        fname = '1-61704860349.sgf'
        fname = '1-61717098200.sgf'     # black all pass-alive, two white chains: one alive, one with only 1 eye
        fname = '1-61717672696.sgf'     # big regions, nothing pass-alive
        fname = '2-61736655674.sgf'     # dead stone within pass-alive region can be safely removed
        fname = '2-61758327600.sgf'     # here dead stone removal really helps score (still W+)
        fpath = f'{myconf.EXP_HOME}/selfplay17.300/sgf/full/{fname}'
        fpath = f'{myconf.EXP_HOME}/endgame29/sgf/full/2015-01-07T01:56:00.051Z_ho6o2gojvb9g-8369170389.sgf'  # Tromp -7 Benson: 4
        # at move 74, game is already set
        fpath = f'{myconf.EXP_HOME}/endgame31_1_benson.labelled.600/sgf/full/2015-01-07T02:04:51.978Z_oqi051ul4urq-40841835255.sgf'
        fpath = f'{myconf.EXP_HOME}/selfplay/sgf/full/1-79782750761.sgf'
        reader = SGFReader.from_file_compatible(fpath)
        pos = reader.last_pos(ignore_final_pass=True, up_to=84)
        board = pos.board

        for color in (go.BLACK, go.WHITE):
            print(f'\nRunning for {color}')
            tracker = BensonAnalyzer.from_board(board, color)
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
        score_tromp = pos.score_tromp()
        benson_detail = pos.score_benson()
        print('Score: Tromp=%.1f, Benson=%s' % (score_tromp, benson_detail))

    def test_benson_top50(self):
        """ just curious what UL chains looks like for Top50
        """
        game_id = '2015-05-15T06:04:40.426Z_kn8727sbhgld'   # no UL chains for W
        game_id = '2015-03-30T01:00:38.957Z_x2rk400zy8fk'   # seki in top-left
        # 'endgame31/2015-08-16T13:24:20.572Z_4ya0d0havhsw-60519671507' move #89: top-right white block is killed,
        # but not counted as black area. In move #88 it's counted as black due to lack of eye space
        store = GameStore()
        ds_top = store.ds_top
        reader = ds_top.get_game(f'go9/{game_id}.sgf')
        pos = reader.last_pos(ignore_final_pass=True)
        board = pos.board

        for color in (go.BLACK, go.WHITE):
            print(f'\nRunning for {color}')
            tracker = BensonAnalyzer.from_board(board, color)
            chain_ids, regions = tracker.eliminate()
            assert len(chain_ids) >= 0
            for chain_idx in chain_ids:
                group = tracker.lib_tracker.groups[chain_idx]
                stone0 = next(iter(group.stones))
                print(f'chain {chain_idx}: {group.color}  %d stones, %d liberties: %s' % (
                    len(group.stones), len(group.liberties), coords.to_gtp(stone0)))

        score_tromp = pos.score_tromp()
        benson_detail = pos.score_benson()
        print('Score: Tromp=%.1f, Benson=%s' % (score_tromp, benson_detail))


class TestMaskedScoring(test_utils.MinigoUnitTest):
    EMPTY_ROW = '.' * go.N + '\n'

    def case10kyu_1(self):
        """ https://online-go.com/puzzle/67916
        """
        return test_utils.load_board('''
            .....XO..
            .....XO.O
            ....X.XO.
            ......XO.
            ......XX.
        ''' + self.EMPTY_ROW * 4)

    @staticmethod
    def create_mask_topright(nrows, ncols) -> np.array:
        mask = np.zeros((9, 9), dtype=np.int8)
        for i in range(nrows):
            for j in range(1, ncols+1):
                mask[i, -j] = 1
        return mask

    def test_masked_score(self):
        board = self.case10kyu_1()
        mask = self.create_mask_topright(5, 5)

        pos = go.Position(board, komi=0)
        score_tromp = pos.score_tromp()
        score_masked = pos.score_tromp(mask=mask)
        print('Score: Tromp=%.1f, score within 5x5 mask=%.1f' % (score_tromp, score_masked))

        mask = self.create_mask_topright(3, 3)
        # 3x3 corner: 1 black, 1 dame, white area=7
        score_masked = pos.score_tromp(mask=mask)
        print('Score: Tromp=%.1f, score within 3x3 mask=%.1f' % (score_tromp, score_masked))
        assert score_masked == -6

    def notest_finalize_corner(self):
        """
        # puzzles are all life-and-death. so if successful, it should be pass-alive
        It's easy to find pass-alive chains; but not so much can be said of the "enclosed" regions
        We could just test for the presence of *any* safe chains in the corner, not worry about regions
        """
        board = self.case10kyu_1()
        board[3, -1] = -1
        board[1, -2] = -1
        # pos = go.Position(board, komi=0)
        tracker = BensonAnalyzer.from_board(board, go.WHITE)
        print(len(tracker.regions), 'regions')
        for rid, region in tracker.regions.items():
            num_opp_stones = len(region.stones) - len(region.liberties)
            print(f'region {region.id}: {region.color}  size=%d, %d opp stones' % (
                len(region.stones), num_opp_stones))

    def test_masked_score_from_sgf(self):
        """
        """
        fname = '/Users/hyu/Downloads/go-puzzle9/Amigo no igo - 詰碁2023 - Life and Death/総合問題４級.sgf'
        assert os.path.isfile(fname)
        reader = SGFReader.from_file_compatible(fname)
        assert len(reader.black_init_stones()) == 9 and len(reader.white_init_stones()) == 6

        for pwc in reader.iter_pwcs():
            break
        pos = pwc.position  # type: go.Position
        pos.komi = 0

        mask4x6 = self.create_mask_topright(4, 6)
        score_tromp = pos.score_tromp()
        score_masked = pos.score_tromp(mask=mask4x6)
        print(score_tromp, score_masked)
        # no enclosed region
        assert score_tromp == 3
        assert score_tromp == score_masked + 1

        mask3x5 = self.create_mask_topright(3, 5)
        assert pos.score_tromp(mask=mask3x5) == -5

        assert pos.to_play == go.BLACK
        pos_enclosed = pos.play_move(coords.from_gtp('E9')).play_move(coords.from_gtp('F9')).\
            play_move(coords.from_gtp('D9')).play_move(coords.from_gtp('J7')).play_move(coords.from_gtp('J6'))
        assert pos_enclosed.score_tromp(mask=mask3x5) == -11
        # still white majority, but margin is thin!
        assert pos_enclosed.score_tromp(mask=mask4x6) == -2

        # play correct solution, where a big white chain is captured
        for i, pwc in enumerate(reader.iter_pwcs()):
            print(i, pwc.position.n, coords.to_gtp(pwc.next_move))
        final_pos = pwc.position.play_move(pwc.next_move)
        assert final_pos.score_tromp(mask=mask3x5) == 9 - 5
        assert final_pos.score_tromp(mask=mask4x6) == 12
