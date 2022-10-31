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
from typing import List, Set

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
    """ see how well extreme Q values correlate with game outcome.
    find a good strategy for trade-off accuracy vs savings, also how much savings
    """

    def __init__(self, resign_thresh: float = 0.9, run_length: int = 1):
        self._thresh = resign_thresh
        self._run_length = run_length
        # self.q_curves = dict()
        self.num_games = 0
        self.num_games_resigned = 0
        self.num_errors = 0
        self.sum_moves = 0
        self.sum_saved = 0
        self._error_games = []

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

    @staticmethod
    def count_consecutive_overage(ts: pd.Series, thresh: float) -> pd.Series:
        """ count consecutive #times q has exceeded thresh
        When q-value flips between 1 to -1, we reset
        """
        consecutive_count = pd.Series(0, index=ts.index)

        prev_sign = 5  # just not 1, -1, 0
        for i in ts.index:
            qval = ts[i]
            cur_sign = np.sign(qval)
            if abs(qval) >= thresh:
                if cur_sign == prev_sign:
                    consecutive_count[i] += consecutive_count[i - 1] + 1
                else:
                    consecutive_count[i] = 1
            else:
                consecutive_count[i] = 0

            prev_sign = cur_sign

        return consecutive_count

    @staticmethod
    def extract_q_curve(reader: SGFReader) -> pd.Series:
        q_vals = []
        for move_idx, (move, comments) in enumerate(reader.iter_comments()):
            assert len(comments) == 1
            q_val = QAnalysis._extract_q(comments[0])
            q_vals.append(q_val)
        ts = pd.Series(q_vals)
        return ts

    def add_game(self, reader: SGFReader):
        """  """
        winner_sign = reader.result()
        game_name_short = os.path.basename(reader.name)

        ts = self.extract_q_curve(reader)
        ts_cc = self.count_consecutive_overage(ts, self._thresh)

        ts_resign = ts_cc[ts_cc >= self._run_length]
        if len(ts_resign) > 0:
            resign_idx = ts_resign.index[0]
            sign_at_resign = np.sign(ts[resign_idx])
            self.num_games_resigned += 1
        else:  # no resign
            resign_idx = len(ts) - 1
            sign_at_resign = winner_sign

        is_error = sign_at_resign != winner_sign
        self.num_games += 1
        self.num_errors += is_error
        self.sum_moves += len(ts)
        self.sum_saved += len(ts) - resign_idx - 1
        if is_error:
            logging.info(f'{game_name_short} %d moves {reader.result_str()}: resigned at {resign_idx} %.1f',
                         len(ts), ts[resign_idx])

    def report(self):
        logging.info(f'Total {self.num_games} games, thresh={self._thresh}, run_len={self._run_length}: '
                     f'{self.num_games_resigned} resigned = %.1f%%, {self.num_errors} errors = %.1f%%, saved {self.sum_saved} = %.1f%%',
                     self.num_games_resigned / self.num_games * 100,
                     self.num_errors / self.num_games * 100, self.sum_saved / self.sum_moves * 100)


def test_count_consecutive_overage():
    ts = pd.Series([0, 0.3, 0.9, 0.8, 0.9, 0.9, 1.0])
    ts = pd.Series([0, 0.3, 0.9, 0.8, 0, -0.9, -1.0])
    ts = pd.Series([0, 0.3, 0.9, -0.8, 0.9, -1.0, 0.9, 0.9])
    ts_cc = QAnalysis.count_consecutive_overage(ts, 0.9)
    df = pd.DataFrame({'ts': ts, 'count': ts_cc})
    print()
    print(df)


def test_resign_stats():
    """
selfplay6:  restrict on move# should help this
'11:25:23        INFO Total 3376 games, thresh=0.9, run_len=2: 2903 resigned = 86.0%, 294 errors = 8.7%, saved 94525 = 35.6%'

selfplay7:
'11:04:52        INFO Total 3324 games, thresh=0.9, run_len=1: 35 errors = 1.1%, saved 47070 = 18.1%'
'11:23:18        INFO Total 3324 games, thresh=0.9, run_len=2: 1986 resigned = 59.7%, 26 errors = 0.8%, saved 40118 = 15.4%'
'11:06:48        INFO Total 3324 games, thresh=0.95, run_len=1: 15 errors = 0.5%, saved 33322 = 12.8%'
'11:09:37        INFO Total 3324 games, thresh=0.95, run_len=2: 11 errors = 0.3%, saved 27732 = 10.7%'
    """
    qstat = QAnalysis(0.9, 2)
    sgf_fname = f'{myconf.EXP_HOME}/selfplay/sgf/full/0-32080527168.sgf'
    sgf_pattern = f'{myconf.EXP_HOME}/selfplay7/sgf/full/*.sgf'
    processor = SgfProcessor([qstat])
    processor.process(sgf_pattern)
    qstat.report()


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
    """ #unique states at certain moves, as well as freq of each zhash """
    def __init__(self, move_indices: List[int] = None):
        """ move indices are 1-based, i.e. the index of the first move is 1 """
        self._moves_of_interest = move_indices
        if not move_indices:
            self._moves_of_interest = list(range(1, 8)) + list(range(10, 80, 10))

        self.zhash_by_move = {move_idx: Counter() for move_idx in self._moves_of_interest}
        self.num_games = 0

    def add_game(self, reader: SGFReader):
        zhashes = [pwc.position.zobrist_hash for pwc in reader.iter_pwcs()]
        for move_idx, cnter in self.zhash_by_move.items():
            # if a game ends early, use the last state for higher move indices
            zhash = zhashes[move_idx] if move_idx < len(zhashes) else zhashes[-1]
            cnter[zhash] += 1
        self.num_games += 1

    def report(self) -> pd.DataFrame:
        cnt_by_move = {move_idx: len(cnter) for move_idx, cnter in self.zhash_by_move.items()}
        ts = pd.Series(cnt_by_move, name='count')
        ts.index.name = 'move'
        df = pd.DataFrame({'count': ts, 'freq': ts / self.num_games})
        print(df.T)
        return df

    def zhash_count_for_move(self, move_idx: int) -> Counter:
        return self.zhash_by_move[move_idx]

    def zhash_set_for_move(self, move_idx: int) -> Set:
        cnter = self.zhash_by_move[move_idx]
        return set(cnter.keys())


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

    def _process_game(self, sgf_fname: str, reader: SGFReader):
        for stat in self._stats:
            stat.add_game(reader)

    def process(self, sgf_glob_pattern, max_num_games: int = 1000_000) -> int:
        logging.info(f'Processing {sgf_glob_pattern} ...')
        sgf_fnames = glob.glob(sgf_glob_pattern)
        cnt = 0
        for sgf_fname in sgf_fnames:
            if cnt >= max_num_games:
                logging.info(f'#processed games exceeded {max_num_games}, break')
                break
            if not sgf_fname.endswith('.sgf'):
                continue
            reader = SGFReader.from_file_compatible(f'{sgf_fname}')
            self._process_game(sgf_fname, reader)
            cnt += 1
        return cnt

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
    sgf_pattern = f'{myconf.EXP_HOME}/eval_bots-model11/model11_2/model10/*.sgf'
    sgf_pattern = f'{myconf.EXP_HOME}/eval_gating/model12_2-vs-elo5k/kata*.sgf'
    # sgf_pattern = f'{myconf.EXP_HOME}/selfplay/sgf/full/*sgf'
    run_tournament_report(sgf_pattern)


def test_review_common_states():
    """
    0 shared states at move#60, across all selfplays!
    """
    # review_root = '/Users/hyu/PycharmProjects/dlgo/9x9-exp2/eval_review/kata1_5k'
    review_root = '/Users/hyu/PycharmProjects/dlgo/9x9-exp2'
    MOVE_OF_INTEREST = 10
    MAX_NUM_GAMES_EACH = 2000
    # model_ids = [f'model{x}'
    #              for x in ['1_5', '2_2', '3_3', '4_4', '5_2', '6_2', '7_4', '8_4', '9_4', '10_4', '11_3', '12_2']]
    model_ids = range(1, 12)
    model_ids = model_ids[:]

    zhs_list = []  # type: List[Counter]
    for model_id in model_ids:
        sgf_pattern = f'{review_root}/selfplay{model_id}/sgf/full/*.sgf'
        dstats = DiversityStats()
        processor = SgfProcessor([dstats])
        num_games = processor.process(sgf_pattern, max_num_games=MAX_NUM_GAMES_EACH)
        print(f'\nProcessed {num_games} games for {model_id}')
        df = dstats.report()
        s60 = dstats.zhash_count_for_move(MOVE_OF_INTEREST)
        zhs_list.append(s60)

    num_models = len(model_ids)
    BAND_SIZE = num_models - 1
    unique2d = np.ones((num_models, num_models), dtype=np.int) * -1
    cnt2d = np.ones((num_models, num_models), dtype=np.int) * -1
    for m1_idx in range(num_models):
        cnter1 = zhs_list[m1_idx]
        unique2d[m1_idx, m1_idx] = len(cnter1)
        cnt2d[m1_idx, m1_idx] = sum(cnter1.values())  # not double-counted
        for m2_idx in range(m1_idx + 1, min(m1_idx + BAND_SIZE + 1, num_models)):
            cnter2 = zhs_list[m2_idx]
            states_in_common = set(cnter1.keys()).intersection(cnter2.keys())
            unique2d[m1_idx, m2_idx] = len(states_in_common)
            cnt2d[m1_idx, m2_idx] = sum(cnter1[k] + cnter2[k] for k in states_in_common)
    dfu = pd.DataFrame(unique2d, index=model_ids, columns=model_ids)
    print(f'#shared states at move {MOVE_OF_INTEREST}:')
    print(dfu)
    dfc = pd.DataFrame(cnt2d, index=model_ids, columns=model_ids)
    print(f'#shared state counts at move {MOVE_OF_INTEREST}:')
    print(dfu.astype(str) + '/' + dfc.astype(str))


def test_review_all_selfplay():
    """ review state dist change across generations, see how much is shared, and thus revised
    - quantify the amount of natural exploration not due to new model: even this depends on the model
    - how much inconsistencies in vnet targets for the same state due to uncertain game outcomes
    """
