""" DNN trained on expert-games tend to pass-pass early to finish the game. Quite reasonable to humans.
But Tromp scoring on final board is often incorrect due to dead stones. We try another vnet (which
plays quite ugly, but seems to have better value prediction) to see if it can score those final positions properly
"""
import os

import myconf
from sgf_wrapper import SGFReader
from k2net import DualNetwork


def eval_game(sgf_fname, dnet: DualNetwork):
    reader = SGFReader.from_file_compatible(sgf_fname)
    last_pos = reader.last_pos()
    policy, value = dnet.run(last_pos)
    _, sgf_base = os.path.split(sgf_fname)
    print(f'{sgf_base} RE: %s,  vnet: %.1f' % (reader.result_str(), value))


def main():
    sgf_dir = f'{myconf.EXP_HOME}/eval_bots/sgfs'
    model_fname = f'{myconf.MODELS_DIR}/model4_epoch3.h5'
    model = DualNetwork(model_fname)
    for sgf_fname in os.listdir(sgf_dir):
        if not sgf_fname.endswith('.sgf'):
            continue
        eval_game(f'{sgf_dir}/{sgf_fname}', model)


if __name__ == '__main__':
    main()

