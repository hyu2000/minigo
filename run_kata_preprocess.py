""" generate TFrecord files from KataGo analyzed games; train/val split """
import itertools
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
    s = f'%.2f %.2f {rinfo.visits}' % (rinfo.winrate, rinfo.scoreLead)

    lines = [s]
    lines.append('move win% lead visits (%) prior pv')
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
    responses = engine.analyze(arequest)

    samples = []
    for i, (position, move, resp1) in enumerate(zip(positions, moves, responses)):
        assert resp1.turnNumber == i

        pi, v = assemble_train_target(resp1)
        features = preprocessing.calc_feature_from_pos(position)
        tf_sample = preprocessing.make_tf_example(features, pi, v)
        samples.append(tf_sample)

    print('Got %d samples' % len(samples))
    return samples


def write_batch(fname, samples_batch: List):
    if len(samples_batch) == 0:
        return
    logging.info(f'Writing %d samples to %s', len(samples_batch), fname)
    preprocessing.write_tf_examples(fname, samples_batch)


def preprocess(train_val_split=0.9, games_in_batch=1000):
    data_dir = '/Users/hyu/go/g170archive/sgfs-9x9'
    ds = KataG170DataSet(data_dir)

    model = KataModels.MODEL_B6C96
    engine = KataEngine(model)
    engine.start()

    i_batch = 0
    samples_batch = []
    for i_game, (game_id, reader) in enumerate(ds.game_iter()):
        samples = process_one_game(engine, reader)
        samples_batch.extend(samples)

        if (i_game + 1) % games_in_batch == 0:
            fname = f'{myconf.FEATURES_DIR}/train/train-{i_batch}.tfrecord.zz'
            write_batch(fname, samples_batch)

            samples_batch = []
            i_batch += 1

    fname = f'{myconf.FEATURES_DIR}/train/train-{i_batch}.tfrecord.zz'
    write_batch(fname, samples_batch)

    engine.stop()


def nottest_gen_data():
    model = KataModels.MODEL_B6C96
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


if __name__ == '__main__':
    random.seed(2021)
    preprocess()
