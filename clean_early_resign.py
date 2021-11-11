""" one-off: my new selfplay, even with resign_thresh=-0.99, still generates lots of early resignations, all W+R

    Remove those tfrecords if sgf has <=40 moves
"""
import os
import myconf
import shutil
from sgf_wrapper import SGFReader


def main():
    sgf_path = f'{myconf.SELFPLAY_DIR}/sgf/full'
    sgf_move_to = f'{myconf.SELFPLAY_DIR}/sgf/short'

    num_short_game = num_removed = 0
    for sgf_fname in os.listdir(sgf_path):
        if not sgf_fname.endswith('.sgf'):
            continue

        reader = SGFReader.from_file_compatible(f'{sgf_path}/{sgf_fname}')
        num_nodes = reader.num_nodes()
        if num_nodes <= 40:
            num_short_game += 1
            game_id = os.path.splitext(os.path.basename(sgf_fname))[0]
            print(f'{game_id} {num_nodes} {reader.result_str()}')

            shutil.move(f'{sgf_path}/{sgf_fname}', f'{sgf_move_to}/{sgf_fname}')

            tfrecord = os.path.join(myconf.SELFPLAY_DIR, 'train', f'{game_id}.tfrecord.zz')
            if not os.path.exists(tfrecord):
                print(f'cant find tfrecord {tfrecord}')
            else:
                os.remove(tfrecord)
                num_removed += 1

    print(f'Total: found {num_short_game} short games, deleled {num_removed} tfrecords')


if __name__ == '__main__':
    main()
