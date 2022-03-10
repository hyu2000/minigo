import io
import subprocess
import json
from typing import List, Dict

import attr

import coords
import go
import sgf_wrapper
from go import PlayerMove

ANALYSIS_CONFIG = '/Users/hyu/go/analysis_example.cfg'
MODELS_DIR = '/Users/hyu/go/models'
MODEL_B6 = '/kata1-b6c96-s175395328-d26788732.txt.elo10k.gz'
MODEL_B40 = 'kata1-b40c256-s11101799168-d2715431527.bin.gz'
MODEL_B20 = 'g170e-b20c256x2-s5303129600-d1228401921.bin.gz' \
            ''
CMDLINE_TEMPLATE = '/opt/homebrew/bin/katago analysis -config {config} -model {model}'

MIN_VISITS = 5


@attr.s
class ARequest(object):
    moves = attr.ib(type=list)
    analyzeTurns = attr.ib(type=list)
    id = attr.ib(type=str, default='foo')
    boardXSize = attr.ib(type=int, default=go.N)
    boardYSize = attr.ib(type=int, default=go.N)
    komi = attr.ib(type=float, default=5.5)

    rules = attr.ib(type=str, default='chinese')


@attr.s
class AResponse(object):
    # id, turnNumber, moveInfos[{move, order, visits, winrate, pv, prior, scoreLead}, rootInfo{visits, winrate, scoreLead}],
    id = attr.ib(type=str)
    turnNumber = attr.ib(type=int)
    moveInfos = attr.ib(type=list)
    rootInfo = attr.ib(type=dict)
    isDuringSearch = attr.ib(type=bool)


@attr.define
class RootInfo(object):
    currentPlayer: str
    visits: int
    winrate: float
    scoreLead: float

    @classmethod
    def from_dict(cls, d: dict):
        """ ignore attributes we don't care """
        return RootInfo(currentPlayer=d['currentPlayer'], visits=d['visits'], winrate=d['winrate'],
                        scoreLead=d['scoreLead'])


@attr.define
class MoveInfo(object):
    # {'lcb': 0.804340414, 'move': 'C3', 'order': 0, 'prior': 0.847978532, 'pv': ['C3', 'B3', 'C4', 'D2', 'B4', 'B2', 'D3', 'A4', 'E2', 'D1', 'B1', 'A3', 'D5', 'E4', 'B5'], 'scoreLead': 5.24203617, 'scoreMean': 5.24203617, 'scoreSelfplay': 5.87824093, 'scoreStdev': 15.3016962, 'utility': 0.554078159, 'utilityLcb': 0.697062643, 'visits': 493, 'winrate': 0.753274527}
    move: str
    order: int
    visits: int
    winrate: float
    pv: list
    prior: float
    scoreLead: float

    @classmethod
    def from_dict(cls, d: dict):
        """ ignore attributes we don't care """
        return MoveInfo(move=d['move'], order=d['order'], visits=d['visits'], winrate=d['winrate'],
                        pv=d['pv'], prior=d['prior'], scoreLead=d['scoreLead'])


def start_engine(model=MODEL_B6):
    cmdline = CMDLINE_TEMPLATE.format(config=ANALYSIS_CONFIG, model=f'{MODELS_DIR}/{model}')
    proc = subprocess.Popen(
        cmdline.split(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    stdin = io.TextIOWrapper(
        proc.stdin,
        encoding='utf-8',
        line_buffering=True,  # send data on newline
    )
    stdout = io.TextIOWrapper(
        proc.stdout,
        encoding='utf-8',
    )
    return proc, stdin, stdout


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


def count_good_moves(rinfo: RootInfo, move_infos: List[Dict]) -> int:
    """ how is order determined? not by visits, lcb, winrate, """
    # vis_counts = [move.get('visits') for move in move_infos]
    if rinfo.currentPlayer == 'B':
        good_moves = [move for move in move_infos if move.get('visits') > MIN_VISITS and move.get('winrate') > 0.5]
    else:
        good_moves = [move for move in move_infos if move.get('visits') > MIN_VISITS and move.get('winrate') < 0.5]
    return len(good_moves)


def test_selfplay():
    """ play a game using top move suggested by Kata """
    proc, stdin, stdout = start_engine()

    moves = [["B", "C4"]]
    comments = ['init']
    arequest = ARequest(moves, [len(moves)])
    for i in range(1, 20):
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
        winning_moves = count_good_moves(rinfo, resp1.moveInfos)
        next_move = [rinfo.currentPlayer, move1.move]
        comment = f"move {i}: {next_move} %.2f %.2f root: %.2f %.2f %d" % (
            move1.winrate, move1.scoreLead, rinfo.winrate, rinfo.scoreLead, winning_moves)
        print(comment)
        arequest.moves.append(next_move)
        comments.append(comment)

    remainder = proc.communicate()[0].decode('utf-8')
    print('remainder:\n', remainder)

    player_moves = (PlayerMove(1 if color == 'B' else -1, coords.from_gtp(pos)) for color, pos in moves)
    sgf_str = sgf_wrapper.make_sgf(player_moves, 'UNK', komi=arequest.komi,
                                    white_name='kata',
                                    black_name='kata',
                                    comments=comments)
    with open(f'/Users/hyu/Downloads/test_kata.sgf', 'w') as f:
        f.write(sgf_str)


def assemble_comment(actual_move, resp1: AResponse) -> str:
    rinfo = RootInfo.from_dict(resp1.rootInfo)
    good_moves = count_good_moves(rinfo, resp1.moveInfos)
    s = '%.2f %.2f path=%d' % (rinfo.winrate, rinfo.scoreLead, good_moves)

    lines = [s]
    for move_info in resp1.moveInfos[:10]:
        minfo = MoveInfo.from_dict(move_info)
        is_actual_move = minfo.move == actual_move
        marker = '*' if is_actual_move else ' '
        pv = minfo.pv if is_actual_move or minfo.order == 0 else ''
        s = f'{marker}{minfo.order} {minfo.move} %.2f %.2f %d {pv}' % (minfo.winrate, minfo.scoreLead, minfo.visits)
        lines.append(s)

    return '\n'.join(lines)


def read_multi_responses(stdout, nmoves):
    """ process multiple responses (which could arrive out-of-order), sort by turns """
    responses = []
    for i in range(nmoves):
        output = stdout.readline()
        jdict = json.loads(output)
        responses.append(AResponse(**jdict))

    return sorted(responses, key=lambda x: x.turnNumber)


def test_analyze_game():
    """ analyze & annotate existing game """
    sgf_fname = '/Users/hyu/Downloads/kata1sgfs/kata.b60c320train.sgf'
    model = MODEL_B6
    reader = sgf_wrapper.SGFReader.from_file_compatible(sgf_fname)

    moves = []
    player_moves = []  # type: List[PlayerMove]
    for pwc in reader.iter_pwcs():
        player_moves.append(PlayerMove(pwc.position.to_play, pwc.next_move))
        move = [go.color_str(pwc.position.to_play)[0], coords.to_gtp(pwc.next_move)]
        moves.append(move)

    proc, stdin, stdout = start_engine(model)

    # moves = moves[:5]  # test only
    turns_to_analyze = list(range(len(moves)))
    arequest = ARequest(moves, turns_to_analyze, komi=reader.komi())

    request1 = json.dumps(attr.asdict(arequest))
    # ask engine
    stdin.write(f'{request1}\n')
    responses = read_multi_responses(stdout, len(moves))

    comments = []
    for i, (move, resp1) in enumerate(zip(moves, responses)):
        assert resp1.turnNumber == i

        # assemble debug info in comment
        comment = assemble_comment(move[1], resp1)
        comments.append(comment)
        print(i, len(resp1.moveInfos))

    remainder = proc.communicate()[0].decode('utf-8')
    print('remainder:\n', remainder)

    comments[0] = 'analyzer: %s\n%s' % (model, comments[0])
    sgf_str = sgf_wrapper.make_sgf(player_moves, reader.result_str(), komi=arequest.komi,
                                   white_name=reader.white_name(),
                                   black_name=reader.black_name(),
                                   comments=comments)
    with open(f'/Users/hyu/Downloads/test_annotate.sgf', 'w') as f:
        f.write(sgf_str)

