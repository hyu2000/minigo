import attr
import io
import subprocess
import json
from typing import List, Dict
import go

MODELS_DIR = '/Users/hyu/go/models'
ANALYSIS_CONFIG = '/Users/hyu/go/analysis_example.cfg'
CMDLINE_TEMPLATE = '/opt/homebrew/bin/katago analysis -config {config} -model {model}'


class KataModels:
    # g170 run
    MODEL_B6C96 = 'g170-b6c96-s175395328-d26788732.bin.gz'  # a bit below my level
    MODEL_B20 = 'g170e-b20c256x2-s5303129600-d1228401921.bin.gz'
    # new run
    MODEL_B6_10k = 'kata1-b6c96-s175395328-d26788732.txt.elo10k.gz'
    MODEL_B6_5k  = 'kata1-b6c96-s24455424-d3879081.txt.elo5k.gz'
    MODEL_B40 = 'kata1-b40c256-s11101799168-d2715431527.bin.gz'


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


def start_engine(model=KataModels.MODEL_B6_10k):
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
