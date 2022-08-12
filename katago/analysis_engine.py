"""
https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md
"""
import attr
import io
import subprocess
import json
from typing import List, Dict, Tuple

import numpy as np

import coords
import go

MODELS_DIR = '/Users/hyu/go/models'
ANALYSIS_CONFIG = '/Users/hyu/PycharmProjects/dlgo/minigo/katago/analysis_example.cfg'
CMDLINE_TEMPLATE = '/opt/homebrew/bin/katago analysis -config {config} -model {model}'


class KataModels:
    # g170 run
    G170_B6C96 = f'{MODELS_DIR}/g170-b6c96-s175395328-d26788732.bin.gz'  # the only b6 model in g170 archive
    G170_B20 = f'{MODELS_DIR}/g170e-b20c256x2-s5303129600-d1228401921.bin.gz'
    # new run
    MODEL_B6_10k = f'{MODELS_DIR}/kata1-b6c96-s175395328-d26788732.txt.elo10k.gz'  # last kata1 b6c96 model. same as G170_B6C96!
    MODEL_B6_5k  = f'{MODELS_DIR}/kata1-b6c96-s24455424-d3879081.txt.elo5k.gz'
    MODEL_B6_4k  = f'{MODELS_DIR}/kata1-b6c96-s18429184-d3197121.txt.elo4k.gz'
    MODEL_B40 = f'{MODELS_DIR}/kata1-b40c256-s11101799168-d2715431527.bin.gz'

    @staticmethod
    def model_id(fname: str):
        fname = fname[len(MODELS_DIR) + 1:]
        return fname.removesuffix('.gz').removesuffix('.bin')


@attr.s
class ARequest(object):
    moves = attr.ib(type=list)
    analyzeTurns = attr.ib(type=list)

    id = attr.ib(type=str, default='foo')
    boardXSize = attr.ib(type=int, default=go.N)
    boardYSize = attr.ib(type=int, default=go.N)
    komi = attr.ib(type=float, default=5.5)

    rules = attr.ib(type=str, default='chinese')

    @staticmethod
    def from_position(pos: go.Position):
        """ request to ask for next move """
        moves = [[go.color_str(x.color)[0], coords.to_gtp(x.move)] for x in pos.recent]
        assert len(moves) == pos.n
        return ARequest(moves, [pos.n])


@attr.s
class AResponse(object):
    """ Kata's analysis of a position, w/ several candidate moves """
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


def start_engine(model_path=KataModels.MODEL_B6_10k):
    cmdline = CMDLINE_TEMPLATE.format(config=ANALYSIS_CONFIG, model=model_path)
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


def read_multi_responses(stdout, nmoves) -> List[AResponse]:
    """ process multiple responses (which could arrive out-of-order), sort by turns """
    responses = []
    for i in range(nmoves):
        output = stdout.readline()
        jdict = json.loads(output)
        if 'error' in jdict or 'warning' in jdict:
            print(f'Found error in {i}: %s' % jdict)
            break
        responses.append(AResponse(**jdict))

    return sorted(responses, key=lambda x: x.turnNumber)


class KataEngine:
    def __init__(self, model=KataModels.G170_B6C96):
        self.model_fname = model

    def start(self) -> 'KataEngine':
        self._proc, self._pipe_in, self._pipe_out = start_engine(self.model_fname)
        return self

    def analyze(self, arequest: ARequest):
        request1 = json.dumps(attr.asdict(arequest))
        # ask engine
        self._pipe_in.write(f'{request1}\n')

        nturns = len(arequest.analyzeTurns)
        responses = read_multi_responses(self._pipe_out, nturns)
        assert len(responses) == nturns, f'expecting {nturns} turns, got %d' % len(responses)

        return responses

    def stop(self):
        # verify no extra data to read
        remainder = self._proc.communicate()[0].decode('utf-8')
        assert len(remainder) == 0

    def model_id(self):
        return KataModels.model_id(self.model_fname)


def extract_policy_value(resp1: AResponse):
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


class KataDualNetwork:
    """ kata masquerade as k2net

    This is quite slow.
    Since we do MCTS on our end, #visits should be reduced in kata analysis config.
    We could even just get Kata policy, no MCTS needed on kata side.
    """
    def __init__(self, model_path):
        self.engine = KataEngine(model_path)
        self.engine.start()

        self.model_id = KataModels.model_id(model_path)

    def run(self, position: go.Position):
        arequest = ARequest.from_position(position)
        responses = self.engine.analyze(arequest)
        pi, v = extract_policy_value(responses[0])
        return pi, v

    def run_many(self, positions: List[go.Position]) -> Tuple[np.ndarray, np.ndarray]:
        num_positions = len(positions)
        pis, v = np.zeros((num_positions, go.N * go.N + 1), dtype=np.float32), np.zeros(num_positions, dtype=np.float32)
        for i, pos in enumerate(positions):
            pis[i], v[i] = self.run(pos)
        return pis, v
