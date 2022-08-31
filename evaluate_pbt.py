""" evaluate round-robin fashion among a set of models

All tables should be read row-wise.

     opponent   lr=0.1_vw=1     lr=0.1_vw=1.5   lr=0.2_vw=1     lr=0.2_vw=1.5
lr=0.1_vw=1     -               b2/3, w1/3      b1/3,w1/3       b1/2(*),w2/3
                                0.5             0.33
lr=0.1_vw=1.5                   -               ...             ...

lr=0.2_vw=1                                     -               ...

lr=0.2_vw=1.5                                                   -


- each pair needs to play the same number of games as white/black
- it's ok if more games are played in certain pairs. We only care about win-rate in the end.
- should construct the table from eval-sgfs, so that we can tolerate failures, reruns, parallel runs
"""
import itertools
import os
from collections import defaultdict
import subprocess, shlex
from typing import Tuple, List
import logging
import numpy as np
import pandas as pd

import utils
from evaluate import ModelConfig
from sgf_wrapper import SGFReader
import myconf

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


MIN_NUM_GAMES_SIDED = 3


class RawGameCount:
    """ wrap around raw sided game counts df, provides convenience access """
    def __init__(self, df: pd.DataFrame):
        assert df.shape[0] == df.shape[1]
        self.df = df

    def count(self, black: str, white: str):
        """ black/white should be model_id """
        try:
            return self.df.loc[black, white]
        except KeyError:
            return 0

    def eye(self):
        return pd.DataFrame(np.eye(len(self.df), len(self.df)),
                            index=self.df.index, columns=self.df.columns)

    def format_black_wins(self, df_blackwins: pd.DataFrame) -> pd.DataFrame:
        """ format black_wins to include game count """
        df = df_blackwins.astype(str) + '/' + self.df.astype(str)
        return df.replace('0/0', '-')


def scan_results(sgfs_dir: str, order=None) -> Tuple[RawGameCount, pd.DataFrame]:
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
    models = sorted(models) if order is None else order
    df_counts_raw = pd.DataFrame(game_counts, index=models, columns=models)
    df_counts = df_counts_raw.T.fillna(0).astype(int)
    df_blackwins = pd.DataFrame(black_wins, index=models, columns=models).T.fillna(0).astype(int)
    df_blackwins.index.name = 'black_id'
    return RawGameCount(df_counts), df_blackwins


def verify_and_fold(raw_game_counts: RawGameCount, df_blackwins: pd.DataFrame) -> pd.DataFrame:
    """ verify black/white parity, and that a minimum number of games has been played

    Example: two models playing each other 16 games total, equally as black/white (we check parity)
    df_blackwin: row-id is black, column-id is white
                model1  model2
    black_id
    model1      0       8  (model1 as black won all 8 games)
    model2      4       0

    dfw: this sums up a model's total win (both as black and white). Again, it's row-major
                model1              model2
                nwin total wrate    nwin total wrate
    model1                          12   16     0.75
    model2      4    16    0.25
    """
    df_counts = raw_game_counts.df
    # df_eye = raw_game_counts.eye() * MIN_NUM_GAMES_SIDED
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
    dfw = pd.concat([df_totalwins, df_counts2, df_wrate2], axis=1, keys=['nwin', 'total', 'wrate'])
    dfw = dfw.swaplevel(axis=1)  #.sort_index(axis=1)
    return dfw


def model_fname(i: int, lr, vw) -> str:
    return f'model{i}.lr={lr}_vw={vw}.h5'


class Evaluator:
    """ encapsulate various eval related settings, so we don't need to pass them around """
    
    def __init__(self, sgfs_dir, num_games_per_side: int = 32):
        self.sgfs_dir = sgfs_dir
        self.num_games_per_side = num_games_per_side

    def start_games(self, black_id, white_id, num_games: int) -> subprocess.Popen:
        """
        python evaluate.py
        """
        cmdline = """/Users/hyu/anaconda/envs/tf2/bin/python /Users/hyu/PycharmProjects/dlgo/minigo/evaluate.py
                    --softpick_move_cutoff=6
                    --dirichlet_noise_weight=0.0125
                    --parallel_readouts=16
                    --num_readouts=50
                    --resign_threshold=-1
                    %s %s %d
        """ % (black_id, white_id, num_games)
        args = shlex.split(cmdline)
        p = subprocess.Popen(args)
        # p.wait()
        return p
    
    def run_two_sided_eval(self, model1_str, model2_str):
        """ play two models against each other for n games, then switch sides """
        utils.ensure_dir_exists(self.sgfs_dir)
        raw_game_count, _ = scan_results(self.sgfs_dir)
    
        model1, model2 = ModelConfig(model1_str), ModelConfig(model2_str)
    
        procs = []
        for black, white in [(model1, model2), (model2, model1)]:
            games_to_run = self.num_games_per_side - raw_game_count.count(black.model_id(), white.model_id())
            if games_to_run <= 0:
                continue
            elif games_to_run <= 3:
                logging.info(f'starting {black} vs {white}: {games_to_run} games')
                procs.append(self.start_games(black, white, games_to_run))
            else:
                logging.info(f'starting {black} vs {white}: {games_to_run} games, 2 processes')
                procs.append(self.start_games(black, white, games_to_run // 2))
                procs.append(self.start_games(black, white, games_to_run - games_to_run // 2))

        for p in procs:
            p.wait()
        logging.info(f'Done two-sided: {model1} vs {model2}')
    
    def run_multi_models(self, models: List[str], band_size=1):
        """ eval multiple pairs of models, based on two_sided_eval

        models: ordered by (hypothesized) strength, strongest first
        todo:
        - try to be smart in figuring out actual ranking
        """
        # all pairs:
        # itertools.permutations(models, 2)

        for band in range(1, 1 + band_size):
            print(f'\n\n=====  band {band}')
            it_a, it_b = itertools.tee(models)
            for _ in range(band):
                next(it_b, None)
            for model1, model2 in zip(it_a, it_b):
                self.run_two_sided_eval(model1, model2)
    
    def main_pbt_eval(self):
        raw_game_count, _ = scan_results(self.sgfs_dir)
    
        num_target_games = 5
        gen_idx = 14
        lrs = [0.003, 0.005, 0.006, 0.008]
        vws = [0.6, 0.8, 1, 1.2]
        lr_ref, vw_ref = 0.005, 1
        # for lr in set(lrs) - {lr_ref}:
        #     model1 = model_fname(gen_idx, lr, vw_ref)
        #     modelr = model_fname(gen_idx, lr_ref, vw_ref)
        for vw in set(vws) - {vw_ref}:
            model1 = model_fname(gen_idx, lr_ref, vw)
            modelr = model_fname(gen_idx, lr_ref, vw_ref)
            procs = []
            for black, white in [(model1, modelr), (modelr, model1)]:
                games_to_run = num_target_games - raw_game_count.count(black, white)
                if games_to_run <= 0:
                    continue
                p = self.start_games(black, white, games_to_run)
                procs.append(p)
            for p in procs:
                p.wait()
    
    def state_of_the_world(self, order=None):
        raw_game_count, df_blackwins = scan_results(self.sgfs_dir, order=order)
        print('black_wins:')
        print(raw_game_count.format_black_wins(df_blackwins))
        dfw = verify_and_fold(raw_game_count, df_blackwins)
        print(dfw.swaplevel(axis=1)['wrate'])
        pickle_fpath = '/tmp/df.pkl'
        dfw.to_pickle(pickle_fpath)
        print(f'dfw saved to {pickle_fpath}')


def main():
    sgfs_dir = f'{myconf.EXP_HOME}/eval_bots/sgfs'

    models = [f'model1_epoch16#{x}' for x in (10, 50, 100, 200, 400)]
    evaluator = Evaluator(sgfs_dir, 32)
    # model1, model2 = 'model1_epoch16.h5#50', 'model1_epoch16.h5#10'
    # evaluator.run_two_sided_eval(model1, model2)
    evaluator.run_multi_models(models[::-1], band_size=2)
    evaluator.state_of_the_world(order=models)


if __name__ == '__main__':
    main()
