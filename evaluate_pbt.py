""" evaluate round-robin fashion among a set of models

                lr=0.1_vw=1     lr=0.1_vw=1.5   lr=0.2_vw=1     lr=0.2_vw=1.5
lr=0.1_vw=1     -               b2/3, w1/3      b1/3,w1/3       b1/2(*),w2/3
                                0.5             0.33
lr=0.1_vw=1.5                   -               ...             ...

lr=0.2_vw=1                                     -               ...

lr=0.2_vw=1.5                                                   -


- each pair needs to play the same number of games as white/black
- it's ok if more games are played in certain pairs. We only care about win-rate in the end.
- should construct the table from eval-sgfs, so that we can tolerate failures, reruns, parallel runs
"""
import os
from collections import defaultdict
import subprocess, shlex
from typing import Tuple

import numpy as np
import pandas as pd
from sgf_wrapper import SGFReader
import myconf


MIN_NUM_GAMES_SIDED = 3


class RawGameCount:
    """ wrap around raw sided game counts df, provides convenience access """
    def __init__(self, df: pd.DataFrame):
        assert df.shape[0] == df.shape[1]
        self.df = df

    def count(self, black, white):
        try:
            return self.df.loc[black, white]
        except KeyError:
            return 0

    def eye(self):
        return pd.DataFrame(np.eye(len(self.df), len(self.df)),
                            index=self.df.index, columns=self.df.columns)


def scan_results(sgfs_dir: str) -> Tuple[RawGameCount, pd.DataFrame]:
    """ scan sgfs, build the raw sided stats dfs """
    game_counts = defaultdict(lambda: defaultdict(int))
    black_wins  = defaultdict(lambda: defaultdict(int))
    models = set()
    for sgf_fname in os.listdir(f'{sgfs_dir}'):
        if not sgf_fname.endswith('.sgf'):
            continue
        reader = SGFReader.from_file_compatible(f'{sgfs_dir}/{sgf_fname}')
        black_id = reader.black_name()
        white_id = reader.white_name()
        result_sign = reader.result()
        assert result_sign != 0

        models.update([black_id, white_id])
        game_counts[black_id][white_id] += 1
        black_wins[black_id][white_id] += result_sign == 1

    # construct df: sided first
    models = sorted(models)
    df_counts_raw = pd.DataFrame(game_counts, index=models, columns=models)
    df_counts = df_counts_raw.T.fillna(0).astype(int)
    df_blackwins = pd.DataFrame(black_wins, index=models, columns=models).T.fillna(0).astype(int)
    return RawGameCount(df_counts), df_blackwins


def verify_and_fold(raw_game_counts: RawGameCount, df_blackwins: pd.DataFrame) -> pd.DataFrame:
    """ verify black/white parity, and that a minimum number of games has been played
    """
    df_counts = raw_game_counts.df
    df_eye = raw_game_counts.eye() * MIN_NUM_GAMES_SIDED
    # assert (df_counts + df_eye >= MIN_NUM_GAMES_SIDED).all().all()
    # make sure df_counts == df_counts.T for black/white parity, as well as min #games is played
    pd.testing.assert_frame_equal(df_counts, df_counts.T)
    total_num_games = df_counts.sum().sum()
    print(f'Found {total_num_games} games')

    # now merge sided stats into a single stat: upper-triangle only, ignore the lower half (although it has a natural meaning)
    df_counts2 = df_counts + df_counts.T
    # totalwin: #wins playing black + #wins as white
    df_totalwins = df_blackwins + (df_counts - df_blackwins).T
    df_wrate2 = (df_totalwins / df_counts2).fillna('-')
    dfw = pd.concat([df_totalwins, df_counts2, df_wrate2], axis=1, keys=['bwin', 'total', 'wrate'])
    dfw = dfw.swaplevel(axis=1).sort_index(axis=1)
    return dfw


def model_fname(i: int, lr, vw) -> str:
    return f'model{i}.lr={lr}_vw={vw}.h5'


def start_games(black_id, white_id, num_games: int) -> subprocess.Popen:
    """
    python evaluate.py
    """
    cmdline = """/Users/hyu/anaconda/envs/tf/bin/python /Users/hyu/PycharmProjects/dlgo/minigo/evaluate.py
                --softpick_move_cutoff=6
                --parallel_readouts=16
                --num_readouts=400
                --resign_threshold=-1
                --num_eval_games=%d
                %s %s
    """ % (num_games, black_id, white_id)
    args = shlex.split(cmdline)
    p = subprocess.Popen(args)
    # p.wait()
    return p


def run_evals(model1, model2, num_games):
    p1 = start_games(model1, model2, num_games)
    print(f'started 1')
    p2 = start_games(model2, model1, num_games)
    print(f'started 2')
    p1.wait()
    p2.wait()
    print('done')


def main():
    raw_game_count, _ = scan_results(f'{myconf.EXP_HOME}/eval_bots/sgfs')

    num_target_games = 3
    iter_id = 7
    lrs = [0.004, 0.005, 0.006, 0.008]
    vws = [0.7, 0.8, 1, 1.2]
    lr_ref, vw_ref = 0.005, 1
    # for lr in set(lrs) - {lr_ref}:
    #     model1 = model_fname(iter_id, lr, vw_ref)
    #     modelr = model_fname(iter_id, lr_ref, vw_ref)
    for vw in set(vws) - {vw_ref}:
        model1 = model_fname(iter_id, lr_ref, vw)
        modelr = model_fname(iter_id, lr_ref, vw_ref)
        procs = []
        for black, white in [(model1, modelr), (modelr, model1)]:
            games_to_run = num_target_games - raw_game_count.count(black, white)
            if games_to_run <= 0:
                continue
            p = start_games(black, white, games_to_run)
            procs.append(p)
        for p in procs:
            p.wait()


def state_of_the_world():
    raw_game_count, df_blackwins = scan_results(f'{myconf.EXP_HOME}/eval_bots/sgfs')
    dfw = verify_and_fold(raw_game_count, df_blackwins)
    pickle_fpath = '/tmp/df.pkl'
    dfw.to_pickle(pickle_fpath)
    print(f'dfw saved to {pickle_fpath}')


if __name__ == '__main__':
    main()
    state_of_the_world()
