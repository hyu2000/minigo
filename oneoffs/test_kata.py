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
MODEL_B6_10k = 'kata1-b6c96-s175395328-d26788732.txt.elo10k.gz'
MODEL_B6_5k  = 'kata1-b6c96-s24455424-d3879081.txt.elo5k.gz'
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


def start_engine(model=MODEL_B6_10k):
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
    model = MODEL_B6_10k
    proc, stdin, stdout = start_engine(model)

    moves = ['C2', 'C3', 'D3', 'D2']
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
        print(comment)
        arequest.moves.append(next_move)
        comments.append(comment)

    remainder = proc.communicate()[0].decode('utf-8')
    print('remainder:\n', remainder)

    player_moves = (PlayerMove(1 if color == 'B' else -1, coords.from_gtp(pos)) for color, pos in moves)
    # TODO RE should be valid, otherwise test_analyze cannot read it in. Use Tromp score?
    sgf_str = sgf_wrapper.make_sgf(player_moves, 'UNK', komi=arequest.komi,
                                    white_name=model,
                                    black_name=model,
                                    comments=comments)
    with open(f'/Users/hyu/Downloads/test_selfplay.sgf', 'w') as f:
        f.write(sgf_str)


def _format_pv(move_info: MoveInfo) -> str:
    moves = move_info.pv[1:10]
    return ' '.join(moves).replace('pass', 'x')


def assemble_comment(actual_move, resp1: AResponse) -> str:
    rinfo = RootInfo.from_dict(resp1.rootInfo)
    good_moves = count_good_moves(rinfo, resp1.moveInfos)
    s = f'%.2f %.2f {rinfo.visits} path=%d' % (rinfo.winrate, rinfo.scoreLead, good_moves)

    lines = [s]
    lines.append('move win% lead visits (%) prior pv')
    for move_info in resp1.moveInfos[:10]:
        minfo = MoveInfo.from_dict(move_info)
        is_actual_move = minfo.move == actual_move
        marker = '*' if is_actual_move else ' '
        pv = _format_pv(minfo)  #if is_actual_move or minfo.order == 0 else ''
        s = f'{marker}{minfo.order} {minfo.move} %.2f %.2f %d (%.2f) %.2f {pv}' % (
            minfo.winrate, minfo.scoreLead, minfo.visits, minfo.visits / rinfo.visits, minfo.prior)
        lines.append(s)

    return '\n'.join(lines)


def read_multi_responses(stdout, nmoves):
    """ process multiple responses (which could arrive out-of-order), sort by turns """
    responses = []
    for i in range(nmoves):
        output = stdout.readline()
        jdict = json.loads(output)
        if 'error' in jdict:
            print(f'Found error in {i}: %s', jdict)
            break
        responses.append(AResponse(**jdict))

    return sorted(responses, key=lambda x: x.turnNumber)


def test_analyze_game():
    """ analyze & annotate existing game """
    # sgf_fname = '/Users/hyu/Downloads/kata1sgfs/katatrain.b60c320.sgf'
    sgf_fname = '/Users/hyu/Downloads/optimalC2.5x5.sgf'
    model = MODEL_B20
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

    sgf_str = sgf_wrapper.make_sgf(player_moves, reader.result_str(), komi=arequest.komi,
                                   white_name=reader.white_name(),
                                   black_name=reader.black_name(),
                                   game_comment=f'analyzed by: {model}',
                                   comments=comments)
    with open(f'/Users/hyu/Downloads/test_annotate.sgf', 'w') as f:
        f.write(sgf_str)

