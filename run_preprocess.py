""" generate TFrecord files; train/val split """
import itertools
import random
from collections import Counter, defaultdict
from functools import partial
from typing import List, Iterable

import numpy as np

import preprocessing
import myconf
from sgf_wrapper import SGFReader
from tar_dataset import TarDataSet
from absl import logging


def write_batch(fname, iters: List[Iterable]):
    if len(iters) == 0:
        return
    logging.info('Writing %d games to %s', len(iters), fname)
    preprocessing.write_tf_examples(fname, itertools.chain(*iters))


class GameStats(object):
    def __init__(self):
        self.winner_count = Counter()
        self.komi_counter = Counter()
        self.dur_counter = Counter()
        self.total_moves = 0

    def process_game(self, reader: SGFReader):
        black_margin = reader.black_margin_adj()
        if black_margin is None:
            self.winner_count['na'] += 1
        elif black_margin > 0:
            self.winner_count['b'] += 1
        else:
            self.winner_count['w'] += 1

        if black_margin is not None and abs(black_margin) == reader.UNKNOWN_MARGIN:
            self.winner_count['R|T'] += 1
        komi = reader.komi()
        self.komi_counter[komi] += 1

        self.dur_counter[reader.num_nodes()] += 1

    def show(self):
        print(self.winner_count.most_common())
        print(self.komi_counter.most_common())

        qtiles = np.quantile(list(self.dur_counter.elements()), [0.01, 0.25, 0.5, 0.75, 0.99])
        print(qtiles)


class GameStore(object):
    """ easy access to TarDataSet, game_iter, move_iter """

    def __init__(self):
        data_dir = myconf.DATA_DIR
        tgz_fname = f'{data_dir}/Pro/9x9.tgz'
        self.ds_pro = TarDataSet(tgz_fname, handle_old_sgf=True)
        tgz_fname = f'{data_dir}/top50.tgz'
        self.ds_top = TarDataSet(tgz_fname)
        tgz_fname = f'{data_dir}/nngs.tgz'
        self.ds_nngs = TarDataSet(tgz_fname, check_sgf_suffix=False, handle_old_sgf=True)

        self.all_dss = [self.ds_pro, self.ds_top, self.ds_nngs]

    def game_iter(self, dss: Iterable[TarDataSet], filter_game=False):
        it = itertools.chain.from_iterable(ds.game_iter() for ds in dss)
        if filter_game:
            it = filter(lambda x: TarDataSet.basic_filter(x[1], x[0]), it)
        return it


def summarize():
    store = GameStore()
    stats = GameStats()
    for game_id, reader in store.game_iter([store.ds_nngs], filter_game=True):
        stats.process_game(reader)
    stats.show()


def preprocess(train_val_split=0.9, games_in_batch=2000):
    store = GameStore()
    game_iter = store.game_iter([store.ds_pro, store.ds_top], filter_game=True)

    num_batch = 0
    batch_train = []
    games_val = []  # all val data goes here
    for game_id, reader in game_iter:
        game_samples = preprocessing.calc_samples_from_reader(reader)
        if random.random() < train_val_split:
            batch_train.append(game_samples)
        else:
            games_val.append(game_samples)

        if len(batch_train) < games_in_batch:
            continue

        write_batch(f'{myconf.FEATURES_DIR}/train-{num_batch}.tfexamples', batch_train)
        batch_train = []
        num_batch += 1

    write_batch(f'{myconf.FEATURES_DIR}/train-{num_batch}.tfexamples', batch_train)
    write_batch(f'{myconf.FEATURES_DIR}/val.tfexamples', games_val)


def preprocess_by_stage(sample_rate=0.1, games_in_batch=2000):
    """ separate samples by begin/mid/end of game """
    store = GameStore()
    game_iter = store.game_iter([store.ds_nngs], filter_game=True)

    samples_by_stage = defaultdict(list)
    for game_id, reader in game_iter:
        if reader.black_margin_adj() is None:
            continue
        if random.random() > sample_rate:
            continue

        game_samples = preprocessing.calc_samples_from_reader(reader)
        game_samples = list(game_samples)
        num_moves = len(game_samples)
        if num_moves < 10:
            logging.warning('game %s has only %d moves', game_id, num_moves)
            continue

        for i in range(1, 5):
            samples_by_stage[f'beg-{i}'].append(game_samples[i])
            samples_by_stage[f'end-{i}'].append(game_samples[-i])
        for i in range(-1, 2):
            samples_by_stage[f'mid'].append(game_samples[num_moves // 2 + i])

    for i in range(1, 5):
        preprocessing.write_tf_examples(f'{myconf.FEATURES_DIR}/beg-{i}.tfexamples', samples_by_stage[f'beg-{i}'])
        preprocessing.write_tf_examples(f'{myconf.FEATURES_DIR}/end-{i}.tfexamples', samples_by_stage[f'end-{i}'])
    preprocessing.write_tf_examples(f'{myconf.FEATURES_DIR}/mid.tfexamples', samples_by_stage[f'mid'])


def process_adhoc():
    """ """
    store = GameStore()
    ds = store.ds_nngs
    all_games = ds.getnames()
    key = 'alfalfa-angie-26-14-20'
    game_ids = [x for x in all_games if key in x]
    logging.info('Found %d games', len(game_ids))

    batch = []
    for game_id in game_ids:
        reader = ds.get_game(game_id)
        game_samples = preprocessing.calc_samples_from_reader(reader)
        batch.append(game_samples)
    write_batch(f'{myconf.FEATURES_DIR}/tmp.tfexamples', batch)


if __name__ == '__main__':
    random.seed(2021)
    # summarize()
    preprocess()
    # preprocess_by_stage()
