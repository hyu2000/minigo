""" Benson scoring seems to help endgame quite a bit, but it is slow.
Here we measure when Bensor score starts to differ from Tromp score, as measured by game stage

Data: endgame selfplay sgfs
For each game, we measure when pass-alive chains start to form, how they evolve (#chains, #stones in them),
score diff between Benson and Tromp
For summary: for step 5, 10, 15, ..., chance of pass-alive chains occurring, chance of score differing, avg score diff
"""
from collections import defaultdict

import myconf
from tar_dataset import SgfDataSet
import go


VERBOSE = 0
NUM_GAMES_TO_PROCESS = 100


class ScoreStats:
    def __init__(self):
        self._step_count = defaultdict(int)
        self._step_count_ul = defaultdict(int)
        self._step_total_score_diff = defaultdict(float)

        self._game_length = defaultdict(int)
        self._game_benson_length = defaultdict(int)
        self._game_length_shorted = defaultdict(int)

    def add_step_stat(self, n: int, benson_detail: go.BensonScoreDetail, tromp_score_diff: float):
        """ called every step of a game """
        self._step_count[n] += 1
        if benson_detail.black_area + benson_detail.white_area > 0:
            self._step_count_ul[n] += 1
            self._step_total_score_diff[n] += tromp_score_diff
        else:
            assert tromp_score_diff == 0

    def add_game_stat(self, n: int, n_benson: int):
        """ called once per game:
        n_benson == n if no early stop by Benson
        """
        self._game_length[n] += 1
        self._game_benson_length[n_benson] += 1
        self._game_length_shorted[n - n_benson] += 1

    def summary(self):
        # total number of games processed
        num_games = sum(self._game_length.values())
        assert num_games == self._step_count[0]
        print(f'\nTotal #games: {num_games}')

        # onset of pass-alive chains
        print('\n% of pass-alive:')
        for n in sorted(self._step_count_ul.keys()):
            print(f'{n}\t%.2f' % (self._step_count_ul[n] / self._step_count[n]))

        # when Benson calls a game final
        print('\ngame length shortened stats:')
        for n in sorted(self._game_length_shorted.keys()):
            print(f'{n} %d' % (self._game_length_shorted[n]))


def main():
    selfplay_dir = f'{myconf.EXP_HOME}/endgame30'
    ds = SgfDataSet(f'{selfplay_dir}/sgf/full')
    stats = ScoreStats()
    for i, (game_id, reader) in enumerate(ds.game_iter(shuffle=True)):
        print(f'Processing {game_id}')
        benson_length = 0
        game_length = 0

        for pwc in reader.iter_pwcs():
            pos = pwc.position
            if pos.n < 0:
                continue

            benson_detail = pos.score_benson()
            tromp_score = pos.score_tromp()
            stats.add_step_stat(pos.n, benson_detail, tromp_score - benson_detail.score)
            total_pass_alive_area = benson_detail.black_area + benson_detail.white_area

            game_length = pos.n
            if benson_detail.final and benson_length <= 0:
                benson_length = pos.n

            if VERBOSE >= 2 and total_pass_alive_area > 0:
                print(f'{pos.n}: UA={benson_detail.black_area} {benson_detail.white_area} %s Benson=%.1f TrompDelta=%.1f' % (
                    'final' if benson_detail.final else '', benson_detail.score, tromp_score - benson_detail.score))

        if benson_length <= 0:
            benson_length = game_length
        stats.add_game_stat(game_length, benson_length)

        if NUM_GAMES_TO_PROCESS > 0 and i >= NUM_GAMES_TO_PROCESS - 1:
            break

    stats.summary()


if __name__ == '__main__':
    main()
