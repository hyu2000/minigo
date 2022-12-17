import logging
from typing import List

import absl.app
import attr
import requests

import coords
import go
import sgf_wrapper
from eval_gating import BasicPlayerInterface, KataPlayer, GameResult
from katago.analysis_engine import ARequest, MoveInfo, RootInfo, KataEngine, KataModels


@attr.s
class A0Response:
    action = attr.ib(type=int)
    terminated = attr.ib(type=bool)
    msg = attr.ib(type=str)
    current_board = attr.ib(type=list)


def test_restapi():
    root_url = 'http://localhost:5000/'
    game_id = 111
    r = requests.get(f'{root_url}/{game_id}')
    # first move -1 means AI go first
    r = requests.post(f'{root_url}/{game_id}/move', json={'human_action': 40})
    resp = r.json()
    print(r.status_code, resp['action'], resp['terminated'], resp['msg'])
    r = requests.post(f'{root_url}/{game_id}/move', json={'human_action': 23})
    a0resp = A0Response(**r.json())
    print(r.status_code, a0resp)


class A0JaxPlayer(BasicPlayerInterface):
    """ a toy impl, only for alternative turn playing """
    def __init__(self, url='http://localhost:5000/'):
        self.root_url = url
        self._my_game_id = 112

        self.pos = None
        self._last_move = -1

    def id(self):
        return 'a0jax'

    def initialize_game(self, position):
        assert position.n == 0

        # reset game
        r = requests.get(f'{self.root_url}/{self._my_game_id}')
        assert r.status_code == 200

        self.pos = position
        self._comments = []

    def play_move(self, c):
        self._last_move = coords.to_flat(c)
        self._comments.append('')

    def suggest_moves(self, position: go.Position) -> List:
        """ this would update a0jax state """
        r = requests.post(f'{self.root_url}/{self._my_game_id}/move', json={'human_action': self._last_move})
        assert r.status_code == 200
        resp = A0Response(**r.json())
        if resp.terminated:
            print(f'a0jax says done: {resp.msg}')
            return []
        if resp.msg:
            print(resp.msg)
        return [(coords.flat_to_gtp(resp.action), 1.0)]

    def get_game_comments(self) -> List[str]:
        return self._comments


def play_vs_a0jax(argv):
    """ kata vs a0jax via its stateful rest api
    """
    engine_kata = KataEngine(KataModels.MODEL_B6_5k).start()
    player_kata = KataPlayer(engine_kata, max_readouts=400)
    player_a0jax = A0JaxPlayer()
    black, white = player_kata, player_a0jax
    sgf_fname = f'/Users/hyu/Downloads/{black.id()}-vs-{white.id()}.sgf'

    init_position = go.Position()
    for player in [black, white]:
        player.initialize_game(init_position)

    cur_pos = init_position  # type: go.Position
    max_game_length = 100
    game_result = None
    for move_idx in range(init_position.n, max_game_length):
        if move_idx % 2:
            active, inactive = white, black
        else:
            active, inactive = black, white

        moves_with_probs = active.suggest_moves(cur_pos)

        if len(moves_with_probs) == 0:  # resigned
            game_result = GameResult(-1 * cur_pos.to_play, was_resign=True)
            break

        # just pick the top move
        move = moves_with_probs[0][0]

        # advance game
        c = coords.from_gtp(move)
        active.play_move(c)
        inactive.play_move(c)
        cur_pos = cur_pos.play_move(c)

        benson_score_details = cur_pos.score_benson()
        if benson_score_details.final:  # end the game when score is final
            game_result = GameResult(benson_score_details.score, was_resign=False)
            break
        if cur_pos.is_game_over():  # pass-pass: use Benson score, same as self-play
            game_result = GameResult(benson_score_details.score, was_resign=False)
            break
        if cur_pos.n >= max_game_length:
            logging.warning('ending game without a definite winner: %s', benson_score_details.score)
            game_result = GameResult(0, was_resign=False)
            break

    logging.info('Game ended: %s', game_result.sgf_str())
    final_pos = cur_pos
    black_comments = black.get_game_comments()
    white_comments = white.get_game_comments()
    assert len(black_comments) == len(white_comments) and len(black_comments) == final_pos.n
    comments = [black_comments[i] if i % 2 == 0 else white_comments[i] for i in range(final_pos.n)]

    with open(sgf_fname, 'w') as _file:
        sgfstr = sgf_wrapper.make_sgf(final_pos.recent,
                                      game_result.sgf_str(), komi=final_pos.komi,
                                      comments=comments,
                                      black_name=black.id(), white_name=white.id())
        _file.write(sgfstr)


if __name__ == '__main__':
    absl.app.run(play_vs_a0jax)
