"""
https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md
"""
import attr
import io
import subprocess
import json
from typing import List, Dict, Tuple, Any

import numpy as np

import coords
import go
from k2net import DualNetwork

MODELS_DIR = '/Users/hyu/go/models'
ANALYSIS_CONFIG = '/Users/hyu/PycharmProjects/dlgo/minigo/katago/analysis_example.cfg'
CMDLINE_TEMPLATE = '/opt/homebrew/bin/katago analysis -config {config} -model {model}'


class KataModels:
    # g170 run
    # b6c96 takes ~13s to review a game w/ 500 visits, b20 takes ~80s
    G170_B6C96 = f'{MODELS_DIR}/g170-b6c96-s175395328-d26788732.bin.gz'  # the only b6 model in g170 archive
    G170_B20 = f'{MODELS_DIR}/g170e-b20c256x2-s5303129600-d1228401921.bin.gz'
    # new run, a.k.a. kata1
    MODEL_B6_10k = f'{MODELS_DIR}/kata1-b6c96-s175395328-d26788732.txt.elo10k.gz'  # last kata1 b6c96 model. same as G170_B6C96!
    MODEL_B6_5k  = f'{MODELS_DIR}/kata1-b6c96-s24455424-d3879081.txt.elo5k.gz'  # beats me consistently
    MODEL_B6_4k  = f'{MODELS_DIR}/kata1-b6c96-s18429184-d3197121.txt.elo4k.gz'  # I can beat consistently @ 500(?) readouts
    MODEL_B40 = f'{MODELS_DIR}/kata1-b40c256-s11101799168-d2715431527.bin.gz'

    SHORT_ID_MAP = {
        G170_B6C96: 'g170_b6c96',
        G170_B20: 'g170_b20',
        MODEL_B6_10k: 'kata1_10k',
        MODEL_B6_5k: 'kata1_5k',
        MODEL_B6_4k: 'kata1_4k',
        MODEL_B40: 'kata1_b40'
    }
    FULL_PATH_MAP = {v: k for k, v in SHORT_ID_MAP.items()}

    @staticmethod
    def full_path(model_id: str):
        return KataModels.FULL_PATH_MAP[model_id]

    @staticmethod
    def model_id(fname: str):
        # fname = fname[len(MODELS_DIR) + 1:]
        # return fname.removesuffix('.gz').removesuffix('.bin')
        return KataModels.SHORT_ID_MAP[fname]


@attr.s
class ARequest(object):
    moves = attr.ib(type=list)
    analyzeTurns = attr.ib(type=list)
    maxVisits = attr.ib(type=int, default=500)

    id = attr.ib(type=str, default='foo')
    boardXSize = attr.ib(type=int, default=go.N)
    boardYSize = attr.ib(type=int, default=go.N)
    komi = attr.ib(type=float, default=5.5)

    rules = attr.ib(type=str, default='chinese')

    @staticmethod
    def from_position(pos: go.Position, max_visits: int = 500):
        """ request to ask for next move """
        moves = [[go.color_str(x.color)[0], coords.to_gtp(x.move)] for x in pos.recent]
        assert len(moves) == pos.n
        return ARequest(moves, [pos.n], max_visits, komi=pos.komi)

    @staticmethod
    def format_moves(player_moves: List[go.PlayerMove]) -> List[Any]:
        """ format PlayerMove to kata move spec """
        moves = [[go.color_str(x.color)[0], coords.to_gtp(x.move)] for x in player_moves]
        return moves


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
    winrate: float  # from the doc seems to be mcts
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


_MIN_VISITS_FOR_GOOD_MOVE = 5


def count_good_moves(rinfo: RootInfo, move_infos: List[Dict]) -> int:
    """ how is order determined? not by visits, lcb, winrate, """
    # vis_counts = [move.get('visits') for move in move_infos]
    if rinfo.currentPlayer == 'B':
        good_moves = [move for move in move_infos
                      if move.get('visits') > _MIN_VISITS_FOR_GOOD_MOVE and move.get('winrate') > 0.5]
    else:
        good_moves = [move for move in move_infos
                      if move.get('visits') > _MIN_VISITS_FOR_GOOD_MOVE and move.get('winrate') < 0.5]
    return len(good_moves)


def _format_pv(move_info: MoveInfo) -> str:
    moves = move_info.pv[1:10]
    return ' '.join(moves).replace('pass', 'x')


def assemble_comment(actual_move: str, resp1: AResponse) -> str:
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


def assemble_comments(arequest: ARequest, responses: List[AResponse]) -> List[str]:
    # check responses are in order
    assert all(resp1.turnNumber == i for i, resp1 in enumerate(responses))

    comments = [assemble_comment(move[1], resp1) for move, resp1 in zip(arequest.moves, responses)]
    return comments


class KataEngine:
    """ a simple wrap around KataGo analysis engine """
    def __init__(self, model=KataModels.G170_B6C96):
        self.model_fname = model

    def start(self) -> 'KataEngine':
        self._proc, self._pipe_in, self._pipe_out = start_engine(self.model_fname)
        return self

    def analyze(self, arequest: ARequest) -> List[AResponse]:
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


class KataDualNetwork(DualNetwork):
    """ kata masquerade as k2net

    This is quite slow. To improve,
    1. Since we do MCTS on our end, #visits should be reduced in kata analysis config, but we do use visit dist as pi.
        We could even just get Kata policy, no MCTS needed on kata side.
    2. run_many() could send parallel requests to Kata...
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
