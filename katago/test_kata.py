import io
import os.path
import subprocess
import json
from typing import List, Dict

import attr
import numpy as np

import coords
import go
import sgf_wrapper
from go import PlayerMove
import myconf
from katago.analysis_engine import ARequest, AResponse, MoveInfo, RootInfo, KataModels, read_multi_responses, \
    KataEngine, start_engine, assemble_comment, assemble_comments
from preprocessing import calc_feature_from_pos, make_tf_example
from run_kata_preprocess import extract_policy_value


def test_attr():
    moves = [["B", "C2"]]
    arequest = ARequest(moves, [len(moves)])
    s = json.dumps(attr.asdict(arequest))
    print(s)

    s_resp = '{"id":"foo","isDuringSearch":false,"moveInfos":[],"rootInfo":{},"turnNumber":1}'
    d_resp = json.loads(s_resp)
    aresp = AResponse(**d_resp)
    print(aresp)


def test_attr_new():
    jdict = {'lcb': 0.804340414, 'move': 'C3', 'order': 0, 'prior': 0.847978532,
         'pv': ['C3', 'B3', 'C4', 'D2', 'B4', 'B2', 'D3', 'A4', 'E2', 'D1', 'B1', 'A3', 'D5', 'E4', 'B5'],
         'scoreLead': 5.24203617, 'scoreMean': 5.24203617, 'scoreSelfplay': 5.87824093, 'scoreStdev': 15.3016962,
         'utility': 0.554078159, 'utilityLcb': 0.697062643, 'visits': 493, 'winrate': 0.753274527}
    minfo = MoveInfo.from_dict(jdict)
    assert minfo.order == 0

    rdict = {"currentPlayer": "W", "scoreLead": 6.48754066, "scoreSelfplay": 7.47134726, "scoreStdev": 14.0771683,
             "symHash": "3BACB443031474229689F014D0C0546F", "thisHash": "75FB7349DBEF656A6AE62C08628EDA61",
             "utility": 0.651306416, "visits": 515, "winrate": 0.793640599}
    rinfo = RootInfo.from_dict(rdict)
    print(rinfo)


def test_simple():
    proc, stdin, stdout = start_engine()

    request1 = '{"id":"foo","moves":[["B","C2"]],"rules":"tromp-taylor","komi":0.5,"boardXSize":5,"boardYSize":5,"analyzeTurns":[1]}'

    stdin.write(f'{request1}\n')
    output = stdout.readline()
    print(output.rstrip())
    result = json.loads(output)
    print(len(result))
    # id, turnNumber, moveInfos[{move, order, visits, winrate, pv, prior, scoreLead}, rootInfo{visits, winrate, scoreLead}],

    remainder = proc.communicate()[0].decode('utf-8')
    print(remainder)


def test_simple_parse():
    """ now with Request / Response object """
    proc, stdin, stdout = start_engine()

    moves = [["B", "C2"]]
    arequest = ARequest(moves, [len(moves)])
    request1 = json.dumps(attr.asdict(arequest))

    stdin.write(f'{request1}\n')
    output = stdout.readline()
    # print(output.rstrip())
    jdict = json.loads(output)
    resp1 = AResponse(**jdict)
    move1 = MoveInfo.from_dict(resp1.moveInfos[0])
    rinfo = RootInfo.from_dict(resp1.rootInfo)
    print(len(resp1.moveInfos), move1)
    assert rinfo.currentPlayer == 'W'

    remainder = proc.communicate()[0].decode('utf-8')
    print(remainder)


def test_selfplay():
    """ play a game using top move suggested by Kata """
    model = KataModels.G170_B20
    proc, stdin, stdout = start_engine(model)

    moves = ['C2', 'C3']  #, 'D3', 'B3', 'B2']
    moves = [['B' if i % 2 == 0 else 'W', move] for i, move in enumerate(moves)]
    comments = ['init' for x in moves]
    arequest = ARequest(moves, [len(moves)], komi=0)
    for i in range(1, 30):
        arequest.analyzeTurns = [len(moves)]
        request1 = json.dumps(attr.asdict(arequest))

        # ask engine
        stdin.write(f'{request1}\n')
        output = stdout.readline()

        jdict = json.loads(output)
        resp1 = AResponse(**jdict)
        move1 = MoveInfo.from_dict(resp1.moveInfos[0])
        assert move1.order == 0
        rinfo = RootInfo.from_dict(resp1.rootInfo)
        next_move = [rinfo.currentPlayer, move1.move]
        comment = assemble_comment(move1.move, resp1)
        print(comment[:20], '...')
        arequest.moves.append(next_move)
        comments.append(comment)

    remainder = proc.communicate()[0].decode('utf-8')
    print('remainder:\n', remainder)

    player_moves = (PlayerMove(1 if color == 'B' else -1, coords.from_gtp(pos)) for color, pos in moves)
    # TODO RE should be valid, otherwise test_analyze cannot read it in. Use Tromp score?
    sgf_str = sgf_wrapper.make_sgf(player_moves, 'B+T', komi=arequest.komi,
                                    white_name=model,
                                    black_name=model,
                                    comments=comments)
    with open(f'/Users/hyu/Downloads/test_selfplay.sgf', 'w') as f:
        f.write(sgf_str)


def play_match(engine_black: KataEngine, engine_white: KataEngine) -> str:
    """ play a game between two Kata engines
    """
    moves = []
    moves = [['B' if i % 2 == 0 else 'W', move] for i, move in enumerate(moves)]
    comments = ['init' for x in moves]
    arequest = ARequest(moves, [len(moves)], komi=5.5)
    for i in range(100):
        arequest.analyzeTurns = [len(moves)]

        # ask engine
        player = engine_black if i % 2 == 0 else engine_white
        responses = player.analyze(arequest)
        assert len(responses) == 1
        resp1 = responses[0]

        move1 = MoveInfo.from_dict(resp1.moveInfos[0])
        assert move1.order == 0
        rinfo = RootInfo.from_dict(resp1.rootInfo)
        next_move = [rinfo.currentPlayer, move1.move]
        comment = assemble_comment(move1.move, resp1)
        print(comment[:40], '...')
        arequest.moves.append(next_move)
        comments.append(comment)

    player_moves = (PlayerMove(1 if color == 'B' else -1, coords.from_gtp(pos)) for color, pos in moves)
    # TODO RE should be valid, otherwise test_analyze cannot read it in. Use Tromp score?
    sgf_str = sgf_wrapper.make_sgf(player_moves, 'B+T', komi=arequest.komi,
                                   white_name=engine_white.model_id(),
                                   black_name=engine_black.model_id(),
                                   comments=comments)
    return sgf_str


def test_paired_match():
    engines = []
    for model_fname in [KataModels.MODEL_B6_5k, KataModels.MODEL_B6_4k]:
        engines.append(KataEngine(model_fname).start())

    sgf_str = play_match(engines[0], engines[1])
    with open(f'/Users/hyu/Downloads/test_match0.sgf', 'w') as f:
        f.write(sgf_str)
    sgf_str = play_match(engines[1], engines[0])
    with open(f'/Users/hyu/Downloads/test_match1.sgf', 'w') as f:
        f.write(sgf_str)

    for engine in engines:
        engine.stop()


def test_analyze_game():
    """ analyze & annotate existing game
    """
    sgf_fname = '/Users/hyu/Downloads/katago-sgfs/katatrain.b60c320.sgf'
    # sgf_fname = '/Users/hyu/Downloads/optimalC2.5x5.sgf'
    # sgf_fname = '/Users/hyu/Downloads/katago-sgfs/5x5/C2C3D3B3.sgf'
    sgf_fname = f'{myconf.EXP_HOME}/eval_bots-model2/sgfs-model1-vs-model2/model1_epoch5#100-vs-model2_epoch2#200-15-65075018704.sgf'
    sgf_fname = f'{myconf.EXP_HOME}/eval_bots-model2/sgfs-model1-vs-model2/model1_epoch5#200-vs-model2_epoch2#300-8-63909912641.sgf'
    model = KataModels.G170_B6C96
    reader = sgf_wrapper.SGFReader.from_file_compatible(sgf_fname)

    player_moves = [PlayerMove(pwc.position.to_play, pwc.next_move)
                    for pwc in reader.iter_pwcs()]

    proc, stdin, stdout = start_engine(model)

    # moves = moves[:5]  # test only
    turns_to_analyze = list(range(len(player_moves)))
    arequest = ARequest(ARequest.format_moves(player_moves), turns_to_analyze, komi=reader.komi())

    request1 = json.dumps(attr.asdict(arequest))
    # ask engine
    stdin.write(f'{request1}\n')
    responses = read_multi_responses(stdout, len(player_moves))
    comments = assemble_comments(arequest, responses)

    remainder = proc.communicate()[0].decode('utf-8')
    print('remainder:\n', remainder)

    sgf_str = sgf_wrapper.make_sgf(player_moves, reader.result_str(), komi=arequest.komi,
                                   white_name=reader.white_name(),
                                   black_name=reader.black_name(),
                                   game_comment=f'analyzed by: {model}',
                                   comments=comments)
    out_fname = os.path.basename(sgf_fname)
    with open(f'/Users/hyu/Downloads/annotate.{out_fname}', 'w') as f:
        f.write(sgf_str)


def test_gen_data():
    """ query Kata for policy / value targets along an existing game """
    sgf_fname = f'{myconf.EXP_HOME}/selfplay6/sgf/full/211-54526899914.sgf'
    sgf_fname = '/Users/hyu/go/g170archive/sgfs-9x9-try1/s174479360.1.sgf'
    model = KataModels.G170_B6C96
    reader = sgf_wrapper.SGFReader.from_file_compatible(sgf_fname)

    positions = []
    moves = []
    player_moves = []  # type: List[PlayerMove]
    for pwc in reader.iter_pwcs():
        positions.append(pwc.position)
        player_moves.append(PlayerMove(pwc.position.to_play, pwc.next_move))
        move = [go.color_str(pwc.position.to_play)[0], coords.to_gtp(pwc.next_move)]
        moves.append(move)

    engine = KataEngine(model).start()

    # moves = moves[:5]  # test only
    turns_to_analyze = list(range(len(moves)))
    arequest = ARequest(moves, turns_to_analyze, komi=reader.komi())

    responses = engine.analyze(arequest)

    comments = []
    for i, (position, move, resp1) in enumerate(zip(positions, moves, responses)):
        assert resp1.turnNumber == i

        pi, v = extract_policy_value(resp1)
        features = calc_feature_from_pos(position)
        tf_sample = make_tf_example(features, pi, v)

        # assemble debug info in comment
        comment = assemble_comment(move[1], resp1)
        comments.append(comment)
        # print(i, len(resp1.moveInfos))

    engine.stop()

    sgf_str = sgf_wrapper.make_sgf(player_moves, reader.result_str(), komi=arequest.komi,
                                   white_name=reader.white_name(),
                                   black_name=reader.black_name(),
                                   game_comment=f'analyzed by: {model}',
                                   comments=comments)
    with open(f'/Users/hyu/Downloads/test_annotate.sgf', 'w') as f:
        f.write(sgf_str)
