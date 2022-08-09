""" generate TFrecord files from KataGo analyzed games; train/val split """
import glob
import itertools
import os
import random
from collections import Counter, defaultdict
from typing import List, Iterable
import numpy as np

import coords
import go
import preprocessing
import myconf
import sgf_wrapper
from katago.analysis_engine import AResponse, RootInfo, MoveInfo, KataModels, start_engine, ARequest, KataEngine
from sgf_wrapper import SGFReader
from absl import logging

from tar_dataset import KataG170DataSet


def assemble_train_target(resp1: AResponse):
    """ tf training target: pi, value """
    rinfo = RootInfo.from_dict(resp1.rootInfo)
    # my vnet activation is tanh: win_rate * 2 - 1
    v_tanh = rinfo.winrate * 2 - 1

    pi = np.zeros([go.N * go.N + 1], dtype=np.float32)
    for move_info in resp1.moveInfos:
        minfo = MoveInfo.from_dict(move_info)
        midx = coords.to_flat(coords.from_gtp(minfo.move))
        pi[midx] = minfo.visits
    # kata applies symmetry to minfo.visits, which may not sum up to rinfo.visits. Normalize here
    pi = pi / pi.sum()
    return pi, v_tanh


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
    arequest = ARequest(moves, turns_to_analyze)
    try:
        responses = engine.analyze(arequest)
    except Exception as e:
        logging.error(str(e))
        return []

    samples = []
    for i, (position, move, resp1) in enumerate(zip(positions, moves, responses)):
        assert resp1.turnNumber == i

        pi, v = assemble_train_target(resp1)
        features = preprocessing.calc_feature_from_pos(position)
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

    feature_dir = f'{myconf.FEATURES_DIR}/g170'
    i_batch_train = scan_for_next_batch_number(feature_dir, init)
    if i_batch_train > 0:
        logging.info(f'#### continuing, next_batch={i_batch_train} #####')

    model = KataModels.MODEL_B6_4k
    engine = KataEngine(model)
    engine.start()

    samples_train = []
    for i_game, (game_id, reader) in enumerate(ds.game_iter(start=0, stop=5)):
        samples = process_one_game(engine, reader)
        samples_train.extend(samples)
        logging.info(f'{i_game}th game: %d samples \t\t{game_id}', len(samples))

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
    model = KataModels.MODEL_B6_4k
    engine = KataEngine(model)
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


if __name__ == '__main__':
    random.seed(2021)
    preprocess()
