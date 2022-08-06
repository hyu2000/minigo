import itertools
from collections import Counter

from tests import test_utils
from tar_dataset import TarDataSet, KataG170DataSet
import myconf


class TestTarDataSet(test_utils.MinigoUnitTest):

    def test_all_games(self):
        data_dir = f'{myconf.DATA_DIR}'

        tgz_fname = f'{data_dir}/Pro/9x9.tgz'
        ds1 = TarDataSet(tgz_fname, handle_old_sgf=True)
        tgz_fname = f'{data_dir}/Top50/go9.tgz'
        ds2 = TarDataSet(tgz_fname)
        tgz_fname = f'{data_dir}/NNGS/NNGS.9x9.tgz'
        ds3 = TarDataSet(tgz_fname, check_sgf_suffix=False, handle_old_sgf=True)

        return itertools.chain(ds1.game_iter(), ds2.game_iter(), ds3.game_iter())

    def test_pro(self):
        tgz_fname = f'{myconf.DATA_DIR}/Pro/9x9.tgz'
        ds = TarDataSet(tgz_fname, handle_old_sgf=True)
        ds.run()
        # ds.basic_filter('9x9/ProPairgo/pg2006-d-4.sgf')

    def test_top50(self):
        tgz_fname = f'{myconf.DATA_DIR}/Top50/go9.tgz'
        ds = TarDataSet(tgz_fname)
        ds.run()

    def test_NNGS_sgf(self):
        """ CoPyright[ -> C[   comment
            end-of-file: ---
        """
        data_dir = myconf.DATA_DIR
        sgf_fname = f'{data_dir}/NNGS/1995/07/alfalfa-angie-26-14-20'
        sgf_fname = f'{data_dir}/tmp/hm-seh-9610022150'
        with open(sgf_fname, 'rb') as f:
            sgf_content = f.read().decode('latin-1')
        from sgf_wrapper import SGFReader
        reader = SGFReader.from_string_compatible(sgf_content)
        print(reader.result(), reader.board_size())

    def test_NNGS(self):
        data_dir = myconf.DATA_DIR
        tgz_fname = f'{data_dir}/NNGS/NNGS.9x9.tgz'
        ds = TarDataSet(tgz_fname, check_sgf_suffix=False, handle_old_sgf=True)
        ds.run()

    def test_kata_g170(self):
        """ g170 9x9 game stats:
        komi mostly 5.5 to 7, some 7.5, and a long tail

total 22350 games: 0 handicap
result_counter [(-1, 10753), (1, 10209), (0, 1388)]
komi [(6.5, 4594), (7.0, 3874), (6.0, 3710), (5.5, 2191), (7.5, 2029), (5.0, 999), (8.0, 707), (4.5, 449), (3.5, 289), (4.0, 272), (8.5, 253), (3.0, 229), (2.5, 186), (9.0, 168), (2.0, 134), (1.5, 107), (9.5, 107), (10.0, 102), (1.0, 101), (10.5, 93), (0.5, 92), (0.0, 88), (11.5, 76), (11.0, 73), (-1.0, 64), (12.0, 63), (-0.5, 63), (12.5, 60), (13.5, 58), (-2.0, 53), (13.0, 49), (14.5, 49), (14.0, 46), (-1.5, 43), (-2.5, 39), (15.0, 37), (16.0, 35), (-3.5, 34), (15.5, 32), (-4.5, 28), (-5.0, 27), (-4.0, 26), (16.5, 26), (-3.0, 24), (17.0, 22), (18.5, 22), (17.5, 21), (-5.5, 19), (-6.5, 18), (19.0, 17), (-6.0, 15), (20.0, 14), (-7.5, 14), (-7.0, 13), (21.0, 13), (-8.0, 12), (18.0, 10), (-9.5, 10), (-9.0, 9), (19.5, 9), (29.0, 8), (24.5, 8), (-12.0, 8), (-10.5, 8), (-11.0, 7), (-8.5, 7), (-13.0, 7), (27.0, 7), (26.0, 6), (25.5, 6), (22.0, 6), (23.0, 6), (32.0, 6), (36.0, 6), (37.0, 6), (-13.5, 6), (30.5, 6), (34.0, 5), (21.5, 5), (-16.0, 5), (-17.5, 5), (25.0, 5), (-24.5, 5), (34.5, 5), (-15.5, 5), (-12.5, 4), (-17.0, 4), (27.5, 4), (-10.0, 4), (20.5, 4), (-19.0, 4), (31.5, 4), (26.5, 4), (-11.5, 4), (28.0, 4), (-21.0, 3), (22.5, 3), (-27.0, 3), (38.0, 3), (-18.5, 3), (35.5, 3), (-16.5, 3), (-28.0, 3), (41.5, 3), (-20.5, 3), (23.5, 3), (-25.5, 3), (40.0, 2), (-21.5, 2), (44.5, 2), (-32.0, 2), (-19.5, 2), (-18.0, 2), (-23.5, 2), (-31.5, 2), (-23.0, 2), (35.0, 2), (-34.0, 2), (24.0, 2), (-25.0, 2), (-26.0, 2), (-14.5, 2), (-42.5, 2), (-14.0, 2), (31.0, 2), (80.5, 2), (37.5, 2), (32.5, 2), (28.5, 2), (43.0, 2), (-26.5, 2), (-32.5, 1), (75.0, 1), (74.0, 1), (72.5, 1), (-30.0, 1), (-41.5, 1), (47.0, 1), (45.0, 1), (42.5, 1), (-58.5, 1), (44.0, 1), (-45.0, 1), (-37.0, 1), (33.5, 1), (-62.0, 1), (-70.5, 1), (-15.0, 1), (-33.0, 1), (33.0, 1), (77.0, 1), (53.5, 1), (38.5, 1), (-43.5, 1), (36.5, 1), (-46.0, 1), (-66.5, 1), (-52.5, 1), (42.0, 1), (-48.5, 1), (-27.5, 1), (-22.0, 1), (47.5, 1), (-28.5, 1), (41.0, 1), (-34.5, 1), (-74.0, 1), (-51.0, 1), (-46.5, 1), (68.5, 1), (-29.5, 1), (50.5, 1), (-57.0, 1), (56.0, 1), (-30.5, 1), (51.0, 1), (-22.5, 1), (79.5, 1), (-44.5, 1), (-36.0, 1), (-53.0, 1), (29.5, 1), (-37.5, 1), (45.5, 1), (-24.0, 1), (63.0, 1)]
        """
        data_dir = '/Users/hyu/go/g170archive/sgfs-9x9'
        ds = KataG170DataSet(data_dir, 'zipfiles.lst')
        # sanity check: #games, RE stats
        result_counter = Counter()
        komi_counter = Counter()
        num_games, num_handicap = 0, 0
        for game_id, reader in ds.game_iter(start=0, stop=2):
            assert reader.board_size() == 9
            if not reader.not_handicap():
                num_handicap += 1
            result_counter[reader.result()] += 1
            komi_counter[reader.komi()] += 1
            num_games += 1
            # if num_games > 100:
            #     break

        print(f'total {num_games} games: {num_handicap} handicap')
        print('result_counter', result_counter.most_common())
        print('komi', komi_counter.most_common())
