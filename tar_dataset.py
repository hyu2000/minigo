"""
====== Pro
Pro games from Mini-Go broadcasts by Yomiuri TV
413
Pro players

9x9.tgz 463 games

09:37:01 ERROR failed read 9x9/computer/OZ2.sgf
most rejection due to RE:T
09:37:01 INFO found 463 sgf, ok 375

[('R', 278), ('w', 246), ('b', 217)]
[(5.5, 226), (6.5, 221), (7.0, 8), (7.5, 6), (0.0, 2)]
#moves: .01 .25 .50 .75 .99
[22.61 37.   45.   54.   80.39]

====== Top50
Top 50 players from app Go Quest  6735  ELO rating above 2000 (equals about 1 Kyu)

most rejection due to no RE
09:46:52 INFO found 8606 sgf, ok 2187

no BR|WR, but maybe this is Elo rating?  PW[idontca1 (2481)]

[('na', 6419), ('w', 1164), ('b', 1023)]
[(7.0, 8606)]
#moves:
[20. 38. 47. 57. 84.]

====== NNGS data
No Name Go Server (NNGS nngs.cosmic.org)  1705 games   5 Kyu or better

all.9x9.txt  40810 games
NNGS.9x9.tgz

most rejection due to HA
13:04:16 INFO found 40810 sgf, ok 30208
sgf_wrapper: 40810 sgf, ok 30984

a few games contains illegal moves:
e.g. NNGS/1995/07/jrd-manyfaces-07-20-17

has BR|WR

[('w', 18215), ('b', 12769), ('R', 12580)]
[(0.5, 21329), (5.5, 9326), (4.5, 43), (2.5, 33), (3.5, 31), (1.5, 29), (-3.5, 29), (6.5, 26), (-4.5, 25), (-0.5, 16), (-2.5, 14), (-1.5, 12), (-7.5, 10), (0.0, 9), (-5.5, 8), (-8.5, 7), (-10.5, 5), (9.5, 5), (5.0, 3), (-11.5, 3), (1.0, 3), (-6.5, 2), (-9.5, 2), (-5.0, 2), (10.5, 2), (1071644672.0, 1), (-14.5, 1), (-12.5, 1), (-1073741824.0, 1), (-2.0, 1), (-4.0, 1), (8.5, 1), (9.0, 1), (2.0, 1), (-10.0, 1)]
#moves:
[20.   40.   51.   61.   92.75]


====== Kifu prelim (not used yet)
go9-large: lots of games, no RE
gokif2: no RE

"""
import itertools
import random
import tarfile
from typing import Iterable

from absl import logging

import myconf
from sgf_wrapper import SGFReader


class TarDataSet(object):
    """ this understands how to *properly* extract sgf game in a tgz file (to fix old format issues)

    """
    def __init__(self, tgz_fname: str, check_sgf_suffix: bool = True, handle_old_sgf: bool = False):
        self.tgz_fname = tgz_fname
        self._check_sgf_suffix = check_sgf_suffix
        self._handle_old_sgf = handle_old_sgf

        self._open_tgz()

    def _open_tgz(self):
        self._tarfile = tarfile.open(self.tgz_fname, 'r:gz')  # type: tarfile.TarFile

    def __getstate__(self):
        # tarfile not pickleable
        d = self.__dict__.copy()
        del d['_tarfile']
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        self._open_tgz()

    def getnames(self):
        """ only used for OGS legacy use case """
        return self._tarfile.getnames()

    def num_games(self) -> int:
        game_list = self._tarfile.getnames()
        return len(game_list)

    def get_game(self, game_id: str) -> SGFReader:
        sgf_content = self._tarfile.extractfile(game_id).read()
        sgf_content = sgf_content.decode('latin-1')
        if self._handle_old_sgf:
            return SGFReader.from_string_compatible(sgf_content, name=game_id)
        else:
            return SGFReader.from_string(sgf_content)

    def game_iter(self, shuffle=False):
        """ generator -> game_id, SGFReader """
        name_list = self._tarfile.getnames()
        if shuffle:
            random.shuffle(name_list)
        total, total_used = 0, 0
        for game_id in name_list:
            total += 1
            if self._check_sgf_suffix and not game_id.endswith('.sgf'):
                continue

            try:
                reader = self.get_game(game_id)
            except:
                logging.exception('failed read %s', game_id)
                continue

            yield game_id, reader
            total_used += 1
        logging.info('%s: found %d sgf, yielded %d', self.tgz_fname, total, total_used)

    @staticmethod
    def basic_filter(reader: SGFReader, game_id: str) -> bool:
        if reader.board_size() != 9:
            logging.info('%s board size = %d', game_id, reader.board_size())
            return False
        if not reader.not_handicap():
            # logging.info('%s has handicap', game_id)
            return False
        # logging.info('rank: white %s, black %s', reader.player_rank('w'), reader.player_rank('b'))
        black_margin = reader.black_margin_adj()
        komi = reader.komi()
        if abs(komi) > 20:
            logging.warning('wild komi %s: %s, black_margin=%s', game_id, komi, black_margin)
            return False
        num_nodes = reader.num_nodes()
        if num_nodes < 15:
            if black_margin is not None and abs(black_margin) != 1000:
                logging.warning('short game %s: %d nodes, black_margin=%s', game_id, num_nodes, black_margin)
            return False

        if black_margin is None:
            # logging.info('%s result: %s', game_id, reader.result())
            return True

        return True

    def run(self):
        total_ok = 0
        for game_id, reader in self.game_iter():
            if self.basic_filter(reader, game_id):
                total_ok += 1
        logging.info('sgf after filtering: %d', total_ok)

    def run2(self):
        """ for debugging NNGS.9x9.tgz """
        total = 0
        for tarinfo in self._tarfile:
            total += 1
            if tarinfo.isreg():
                pass
            elif tarinfo.isdir():
                pass
            else:
                print(tarinfo.name, "is", tarinfo.size, "bytes in size and is ", end="")

        logging.info('total: %s items', total)


class GameStore(object):
    """ easy access to TarDataSet, game_iter, move_iter """

    def __init__(self, data_dir=myconf.DATA_DIR):
        tgz_fname = f'{data_dir}/pro.tgz'
        self.ds_pro = TarDataSet(tgz_fname, handle_old_sgf=True)
        tgz_fname = f'{data_dir}/top50.tgz'
        self.ds_top = TarDataSet(tgz_fname)
        tgz_fname = f'{data_dir}/nngs.tgz'
        self.ds_nngs = TarDataSet(tgz_fname, check_sgf_suffix=False, handle_old_sgf=True)

        self.all_dss = [self.ds_pro, self.ds_top, self.ds_nngs]

    def game_iter(self, dss: Iterable[TarDataSet], filter_game=False, shuffle=False):
        it = itertools.chain.from_iterable(ds.game_iter(shuffle=shuffle) for ds in dss)
        if filter_game:
            it = filter(lambda x: TarDataSet.basic_filter(x[1], x[0]), it)
        return it