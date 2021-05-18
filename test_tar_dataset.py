import itertools

from tests import test_utils
from tar_dataset import TarDataSet
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
