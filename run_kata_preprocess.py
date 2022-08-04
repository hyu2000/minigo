""" generate TFrecord files from KataGo analyzed games; train/val split """
import itertools
import random
from collections import Counter, defaultdict
from typing import List, Iterable
import json
import attr
import numpy as np

import coords
import go
import preprocessing
import myconf
import sgf_wrapper
from katago.analysis_engine import AResponse, RootInfo, MoveInfo, KataModels, start_engine, ARequest, KataEngine
from sgf_wrapper import SGFReader
from absl import logging


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


def process_one_game(engine: KataEngine, reader: SGFReader):
    """ query Kata for policy / value targets along an existing game """
    positions = []
    moves = []
    for pwc in reader.iter_pwcs():
        positions.append(pwc.position)
        move = [go.color_str(pwc.position.to_play)[0], coords.to_gtp(pwc.next_move)]
        moves.append(move)

    # moves = moves[:5]  # test only
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


def test_gen_data():
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
