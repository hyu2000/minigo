""" generate TFrecord files from KataGo analyzed games; train/val split """
import glob
import itertools
import os
import random
from collections import Counter, defaultdict
from typing import List, Iterable
import numpy as np

import myconf
import coords
import go
import preprocessing
import sgf_wrapper
from katago.analysis_engine import AResponse, KataModels, start_engine, ARequest, KataEngine, extract_policy_value
from sgf_wrapper import SGFReader
import logging

from tar_dataset import KataG170DataSet

KATA_MODEL_ID = KataModels.MODEL_B6_4k
MAX_VISITS = 50


def process_one_game(engine: KataEngine, reader: SGFReader) -> List:
    """ query Kata for policy / value targets along an existing game """
    positions = []
    moves = []
    for pwc in reader.iter_pwcs():
        positions.append(pwc.position)
        move = [go.color_str(pwc.position.to_play)[0], coords.to_gtp(pwc.next_move)]
        moves.append(move)

    turns_to_analyze = list(range(len(moves)))
    # ignore komi in the actual game: we only care about the default 5.5
    arequest = ARequest(moves, turns_to_analyze, maxVisits=MAX_VISITS)
    try:
        responses = engine.analyze(arequest)
    except Exception as e:
        logging.error(str(e))
        return []

    samples = []
    for i, (position, move, resp1) in enumerate(zip(positions, moves, responses)):
        assert resp1.turnNumber == i

        pi, v = extract_policy_value(resp1)
        features = preprocessing.calc_feature_from_pos(position, myconf.FULL_BOARD_FOCUS)
        tf_sample = preprocessing.make_tf_example(features, pi, v)
        samples.append(tf_sample)

    return samples


def write_batch(fname, samples_batch: List):
    if len(samples_batch) == 0:
        return
    logging.info(f'Writing %d samples to %s', len(samples_batch), fname)
    preprocessing.write_tf_examples(fname, samples_batch)


def scan_for_next_batch_number(feature_dir, init: bool) -> int:
    if not os.path.exists(feature_dir):
        logging.info(f'mkdir {feature_dir}')
        os.mkdir(feature_dir)
        return 0
    flist = glob.glob(f'{feature_dir}/train*.tfrecord.zz')
    if init:
        # clean up
        if len(flist) > 0:
            logging.info('Found %d batches and init=True, should clean up', len(flist))
            # todo
        return 0
    if len(flist) == 0:
        return 0
    batches = [x.removeprefix(f'{feature_dir}/train-').removesuffix('.tfrecord.zz') for x in flist]
    # todo check it's contiguous
    return 1 + max([int(x) for x in batches])


def preprocess(init=False, samples_in_batch=1e5):
    """ due to long runtime for the entire 22k games, we try to make it easy to run in batches:

    init=True: batch starts with 0
    init=False: pick up from where we left
    """
    data_dir = '/Users/hyu/go/g170archive/sgfs-9x9'
    ds = KataG170DataSet(data_dir)
    logging.info(f'Starting preprocess max_visits={MAX_VISITS} ')

    feature_dir = f'{myconf.FEATURES_DIR}/g170'
    i_batch_train = scan_for_next_batch_number(feature_dir, init)
    if i_batch_train > 0:
        logging.info(f'#### continuing, next_batch={i_batch_train} #####')

    engine = KataEngine(KATA_MODEL_ID)
    engine.start()

    samples_train = []
    # 60th zip is probably higher than elo4k
    for i_game, (game_id, reader) in enumerate(ds.game_iter(start=0, stop=60)):
        samples = process_one_game(engine, reader)
        samples_train.extend(samples)
        # logging.info(f'{i_game}th game: %d samples \t\t{game_id}', len(samples))

        if len(samples_train) >= samples_in_batch:
            print(f'progress: {i_game}th game ...')
            fname = f'{feature_dir}/train-{i_batch_train}.tfrecord.zz'
            write_batch(fname, samples_train)

            samples_train = []
            i_batch_train += 1

    fname = f'{feature_dir}/train-{i_batch_train}.tfrecord.zz'
    write_batch(fname, samples_train)

    engine.stop()


def nottest_gen_data():
    engine = KataEngine(KATA_MODEL_ID)
    engine.start()

    sgf_fname = '/Users/hyu/go/g170archive/sgfs-9x9-try1/s174479360.1.sgf'
    reader = sgf_wrapper.SGFReader.from_file_compatible(sgf_fname)
    process_one_game(engine, reader)

    # another game
    sgf_fname = '/Users/hyu/go/g170archive/sgfs-9x9-try1/s174479360.2.sgf'
    reader = sgf_wrapper.SGFReader.from_file_compatible(sgf_fname)
    process_one_game(engine, reader)

    engine.stop()


def nottest_scan_batch():
    feature_dir = f'{myconf.FEATURES_DIR}/g170'
    i_batch = scan_for_next_batch_number(feature_dir, init=False)
    assert i_batch == 0


def count_num_samples():
    """ 11/20/22 wonder how 22k games only produced the amount of data less than 2 selfplays
    """
    data_dir = '/Users/hyu/go/g170archive/sgfs-9x9'
    ds = KataG170DataSet(data_dir)
    total_samples = 0
    for i_game, (game_id, reader) in enumerate(ds.game_iter()):
        total_samples += reader.num_nodes()
        if i_game % 100 == 1:
            print(f'processed {i_game} games: avg %.1f moves/game' % (total_samples / (i_game + 1)))
    print(f'Total {i_game} games, {total_samples} samples')


if __name__ == '__main__':
    random.seed(2021)
    preprocess()
    # count_num_samples()
