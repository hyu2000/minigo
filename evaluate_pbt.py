""" evaluate round-robin fashion among a set of models

                lr=0.1_vw=1     lr=0.1_vw=1.5   lr=0.2_vw=1     lr=0.2_vw=1.5
lr=0.1_vw=1     -               b2/3, w1/3      b1/3,w1/3       b1/2(*),w2/3
                                0.5             0.33
lr=0.1_vw=1.5                   -               ...             ...

lr=0.2_vw=1                                     -               ...

lr=0.2_vw=1.5                                                   -


- each pair needs to play the same number of games as white/black
- it's ok if more games are played in certain pairs. We only care about win-rate in the end.
- should construct the table from eval-sgfs, so that we can tolerate failures, reruns, parallel runs
"""
import os
from collections import defaultdict
import pandas as pd
from sgf_wrapper import SGFReader
import myconf


MIN_NUM_GAMES_SIDED = 3


def scan_results(sgfs_dir: str) -> pd.DataFrame:
    """ scan sgfs to build the model comparison matrix """
    game_counts = defaultdict(lambda: defaultdict(int))
    black_wins  = defaultdict(lambda: defaultdict(int))
    models = set()
    for sgf_fname in os.listdir(f'{sgfs_dir}'):
        if not sgf_fname.endswith('.sgf'):
            continue
        reader = SGFReader.from_file_compatible(f'{sgfs_dir}/{sgf_fname}')
        black_id = reader.black_name()
        white_id = reader.white_name()
        result_sign = reader.result()
        assert result_sign != 0

        models.update([black_id, white_id])
        game_counts[black_id][white_id] += 1
        black_wins[black_id][white_id] += result_sign == 1

    # construct df: sided first
    models = sorted(models)
    df_counts_raw = pd.DataFrame(game_counts, index=models, columns=models)
    assert (df_counts_raw.fillna(MIN_NUM_GAMES_SIDED) >= MIN_NUM_GAMES_SIDED).all().all()
    df_counts = df_counts_raw.T.fillna(0).astype(int)
    df_blackwins = pd.DataFrame(black_wins, index=models, columns=models).T.fillna(0).astype(int)
    # make sure df_counts == df_counts.T for black/white parity, as well as min #games is played
    pd.testing.assert_frame_equal(df_counts, df_counts.T)
    total_num_games = df_counts.sum().sum()
    print(f'Found {total_num_games} games')

    # now merge sided stats into a single stat: upper-triangle only, ignore the lower half (although it has a natural meaning)
    df_counts2 = df_counts + df_counts.T
    # totalwin: #wins playing black + #wins as white
    df_totalwins = df_blackwins + (df_counts - df_blackwins).T
    df_wrate2 = (df_totalwins / df_counts2).fillna('-')
    dfw = pd.concat([df_totalwins, df_counts2, df_wrate2], axis=1, keys=['bwin', 'total', 'wrate'])
    dfw = dfw.swaplevel(axis=1).sort_index(axis=1)
    return dfw


def main():
    df = scan_results(f'{myconf.EXP_HOME}/eval_bots/sgfs')
    df.to_pickle('/tmp/df.pkl')


if __name__ == '__main__':
    main()
