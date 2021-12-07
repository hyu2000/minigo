""" Benson scoring seems to help endgame quite a bit, but it is slow.
Here we measure when Bensor score starts to differ from Tromp score, as measured by game stage

Data: endgame selfplay sgfs
For each game, we measure when pass-alive chains start to form, how they evolve (#chains, #stones in them),
score diff between Benson and Tromp
For summary: for step 5, 10, 15, ..., chance of pass-alive chains occurring, chance of score differing, avg score diff
"""
import myconf
from tar_dataset import SgfDataSet
import go


def main():
    selfplay_dir = f'{myconf.EXP_HOME}/endgame30'
    ds = SgfDataSet(f'{selfplay_dir}/sgf/full')
    for game_id, reader in ds.game_iter():
        print(f'Processing {game_id}')
        for pwc in reader.iter_pwcs():
            pos = pwc.position
            benson_detail = pos.score_benson()
            total_pass_alive_area = benson_detail.black_area + benson_detail.white_area
            if total_pass_alive_area > 0:
                tromp_score = pos.score_tromp()
                print(f'{pos.n}: UA={benson_detail.black_area} {benson_detail.white_area} %s Benson=%.1f TrompDelta=%.1f' % (
                    'final' if benson_detail.final else '', benson_detail.score, tromp_score - benson_detail.score))
        break


if __name__ == '__main__':
    main()
