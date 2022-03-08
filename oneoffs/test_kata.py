import io
import subprocess
import json
import attr

import coords
import sgf_wrapper
from go import PlayerMove

ANALYSIS_CONFIG = '/Users/hyu/go/analysis_example.cfg'
MODEL = '/Users/hyu/go/models/kata1-b6c96-s175395328-d26788732.txt.elo10k.gz'
cmdline = f'/opt/homebrew/bin/katago analysis -config {ANALYSIS_CONFIG} -model {MODEL}'


@attr.s
class ARequest(object):
    moves = attr.ib(type=list)
    analyzeTurns = attr.ib(type=list)
    id = attr.ib(type=str, default='foo')
    boardXSize = attr.ib(type=int, default=5)
    boardYSize = attr.ib(type=int, default=5)
    komi = attr.ib(type=float, default=0.5)

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



def start_engine():
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


def test_selfplay():
    """ play a game using top move suggested by Kata """
    proc, stdin, stdout = start_engine()

    moves = [["B", "C2"]]
    comments = ['init']
    for i in range(1, 25):
        arequest = ARequest(moves, [len(moves)])
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
        comment = f"move {i}: {next_move} %.2f %.2f" % (move1.winrate, move1.scoreLead)
        print(comment)
        moves.append(next_move)
        comments.append(comment)

    remainder = proc.communicate()[0].decode('utf-8')
    print('remainder:\n', remainder)

    player_moves = (PlayerMove(1 if color == 'B' else -1, coords.from_gtp(pos)) for color, pos in moves)
    sgf_str = sgf_wrapper.make_sgf(player_moves, 'UNK', komi=0.5,
                                    white_name='kata',
                                    black_name='kata',
                                    comments=comments)
    with open(f'/Users/hyu/Downloads/test_kata.sgf', 'w') as f:
        f.write(sgf_str)

