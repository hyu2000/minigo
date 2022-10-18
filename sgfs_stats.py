""" analyze a batch of selfplay / eval games for diversity, resign thresh, and other insights

- print sorted list of game moves
- should be easy to embed for live stats & actions

keep all game tree in memory? so we know where exploration happens.
Or just zhash counts in the first cut?
"""
import glob
import logging
import os.path
from collections import defaultdict, Counter
from itertools import islice
from typing import List

import numpy as np
import pandas as pd

import coords
import go
import myconf
from sgf_wrapper import SGFReader
from utils import format_game_summary

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


class StatsItf:
    def add_game(self, reader: SGFReader):
        pass

    def report(self):
        pass


class QAnalysis(StatsItf):
    """ see how well extreme Q values correlate with game outcome
    """

    def __init__(self, model_id: str = ''):
        self.model_id = model_id
        self.q_curves = dict()

    @staticmethod
    def _extract_q(comment: str) -> float:
        """
        Q=-0.99 raw=-0.99 N=435 path=-
        Note this could be the 2nd line
        """
        lines = comment.split('\n')
        line = lines[0]
        if not line.startswith('Q='):
            line = lines[1]
        flds = line.split()
        assert flds[0][:2] == 'Q='
        return float(flds[0][2:])

    def add_game(self, reader: SGFReader):
        winner_sign = reader.result()
        game_name_short = os.path.basename(reader.name)

        q_vals = []
        for move_idx, (move, comments) in enumerate(reader.iter_comments()):
            assert len(comments) == 1
            q_val = self._extract_q(comments[0])
            q_vals.append(q_val)
        # last point is actual outcome
        q_vals.append(winner_sign)
        q_series = pd.Series(q_vals)

        self.q_curves[game_name_short] = q_series


def test_q_extraction():
    stat = QAnalysis('model8_4.mlpackage')
    sgf_fname = f'{myconf.EXP_HOME}/selfplay/sgf/full/0-32080527168.sgf'
    for sgf_fname in glob.glob(f'{myconf.EXP_HOME}/selfplay/sgf/full/*.sgf'):
        reader = SGFReader.from_file_compatible(sgf_fname)
        stat.add_game(reader)


class WinnerStats(StatsItf):
    """ this can count games among multiple players """
    def __init__(self):
        self.num_games = defaultdict(lambda: defaultdict(int))
        self.num_wins = defaultdict(lambda: defaultdict(int))
        self.players = set()

    def add_game(self, reader: SGFReader):
        black_id = reader.black_name()
        white_id = reader.white_name()

        self.num_games[black_id][white_id] += 1
        self.players.add(black_id)
        self.players.add(white_id)

        result = reader.result()
        if result > 0:
            self.num_wins[black_id][white_id] += 1
        elif result < 0:
            pass
            # self.num_wins[white_id][black_id] += 1
        else:  # no RE, or tie
            logging.warning(f'game {reader.name} has no winner: {reader.result_str()}')

    def report(self):
        players = sorted(self.players)

        df_num_games = pd.DataFrame(self.num_games, index=players, columns=players).T.fillna(0).astype(int)
        df_blackwins = pd.DataFrame(self.num_wins,  index=players, columns=players).T.fillna(0).astype(int)
        df = df_blackwins.astype(str) + '/' + df_num_games.astype(str)
        print(df)
        return df


# todo integrate this into WinnerStats, or a separate review class?
def game_outcome_review(sgfs_dir):
    """ KataEngine reviews final game outcome, to see if games are properly scored/finished
    """
    from katago.analysis_engine import KataDualNetwork, KataModels
    dnn_kata = KataDualNetwork(KataModels.G170_B6C96)
    num_disagree = 0

    game_counts = defaultdict(lambda: defaultdict(int))
    disagree_cnt = defaultdict(lambda: defaultdict(int))
    models = set()
    for sgf_fname in os.listdir(f'{sgfs_dir}'):
        if not sgf_fname.endswith('.sgf'):
            continue
        reader = SGFReader.from_file_compatible(f'{sgfs_dir}/{sgf_fname}')
        black_id = reader.black_name()
        white_id = reader.white_name()
        result_sign = reader.result()
        assert result_sign != 0

        pos = reader.last_pos()
        pi, v_expert = dnn_kata.run(pos)
        sign_expert = np.sign(v_expert)
        if result_sign != sign_expert:
            result_str = reader.result_str()
            num_disagree += 1
            logging.info(f'{sgf_fname} {result_str} kata_v={v_expert}')

        models.update([black_id, white_id])
        game_counts[black_id][white_id] += 1
        disagree_cnt[black_id][white_id] += result_sign != sign_expert

    models = sorted(models)
    df_counts_raw = pd.DataFrame(game_counts, index=models, columns=models)
    df_counts = df_counts_raw.T.fillna(0).astype(int)
    df_disagree = pd.DataFrame(disagree_cnt, index=models, columns=models).T.fillna(0).astype(int)
    df_disagree.index.name = 'black_id'
    df = df_disagree.astype(str) + '/' + df_counts.astype(str)
    logging.info(f'Total disagreement: {num_disagree} / %d', df_counts.sum().sum())
    print(df.replace('0/0', '-'))


class DiversityStats(StatsItf):
    """ #unique states at certain moves """
    def __init__(self, move_indices: List[int] = None):
        """ move indices are 1-based, i.e. the index of the first move is 1 """
        self._moves_of_interest = move_indices
        if not move_indices:
            self._moves_of_interest = list(range(1, 8)) + list(range(10, 80, 10))

        self.zhash_by_move = {move_idx: set() for move_idx in self._moves_of_interest}
        self.num_games = 0

    def add_game(self, reader: SGFReader):
        zhashes = [pwc.position.zobrist_hash for pwc in reader.iter_pwcs()]
        for move_idx, s in self.zhash_by_move.items():
            zhash = zhashes[move_idx] if move_idx < len(zhashes) else zhashes[-1]
            s.add(zhash)
        self.num_games += 1

    def report(self):
        cnt_by_move = {move_idx: len(s) for move_idx, s in self.zhash_by_move.items()}
        ts = pd.Series(cnt_by_move, name='count')
        ts.index.name = 'move'
        df = pd.DataFrame({'count': ts, 'freq': ts / self.num_games})
        print(df)
        return df


class GameSequenceReport(StatsItf):
    """ report on move sequences, sorted
    """
    def __init__(self, first_n: int = 12, last_n: int = 2):
        self.first_n = first_n
        self.last_n = last_n

        self.game_summary = []  # type: List

    def add_game(self, reader: SGFReader):
        all_moves = [coords.to_gtp(pwc.next_move) for pwc in reader.iter_pwcs()]
        line = format_game_summary(all_moves, reader.result_str(), self.first_n, self.last_n, sgf_fname=reader.name)
        self.game_summary.append(line)

    def report(self):
        self.game_summary.sort()
        print('\n'.join(self.game_summary))


class SgfProcessor:
    """ first version
    - winner stats
    - diversity stats
    """
    def __init__(self, stats: List[StatsItf]):
        self._stats = stats

    def add_game(self, sgf_fname: str, reader: SGFReader):
        for stat in self._stats:
            stat.add_game(reader)

    def process(self, sgf_glob_pattern):
        sgf_fnames = glob.glob(sgf_glob_pattern)
        for sgf_fname in sgf_fnames:
            if not sgf_fname.endswith('.sgf'):
                continue
            reader = SGFReader.from_file_compatible(f'{sgf_fname}')
            self.add_game(sgf_fname, reader)


def run_tournament_report(sgf_pattern):
    wstats = WinnerStats()
    dstats = DiversityStats()
    game_report = GameSequenceReport()
    processor = SgfProcessor([wstats, dstats, game_report])
    logging.info(f'Scanning {sgf_pattern}')
    processor.process(sgf_pattern)
    logging.info(f'{sgf_pattern}: found %d games', dstats.num_games)
    wstats.report()
    logging.info('\tUnique states by move:')
    dstats.report()
    logging.info('\tgames sorted:')
    game_report.report()


def test_basic_report():
    # sgf_pattern = f'{myconf.EXP_HOME}/eval_bots-model8/model8_4#200/model8_4#200-vs-model7_4#200-*.sgf'
    sgf_pattern = f'{myconf.EXP_HOME}/eval_gating/model8_4/400-vs-elo5k#200/black/*.sgf'
    # sgf_pattern = f'{myconf.EXP_HOME}/selfplay/sgf/full/*sgf'
    run_tournament_report(sgf_pattern)
