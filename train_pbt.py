""" population based training """
import os.path
import sys
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from absl import logging, flags
import go
import myconf
import k2net as dual_net
from run_train import custom_BCE_loss, load_selfplay_data

N = go.N
HOME_DIR = myconf.EXP_HOME
FEATURES_DIR = myconf.FEATURES_DIR


bce = tf.keras.losses.BinaryCrossentropy()
bca = tf.keras.metrics.BinaryAccuracy()


def compile_pbt(lr: float = 5e-3, value_weight: float = 0.5):
    """
    my default: 5e-3
    # SAI: 1e-4
    # KataGo: per-sample learning rate of 6e-5, except 2e-5 for the first 5mm samples
    """
    input_shape = (N, N, dual_net.get_features_planes())
    model = dual_net.build_model(input_shape)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt,
                  loss={
                      'policy': 'categorical_crossentropy',
                      'value':  custom_BCE_loss},
                  loss_weights={
                      'policy': 0.50,
                      'value':  value_weight},
                  metrics={
                      'policy': keras.metrics.CategoricalAccuracy(name="move_acc"),
                  })
    return model


class Population:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir

    def bootstrap(self, init_model_file: str, features_dir: str):
        init_hyper_params = [(lr, vw) for lr in (4e-3, 5e-3, 6e-3, 8e-3) for vw in (0.7, 0.8, 1, 1.2)]

        ds_train = load_selfplay_data(features_dir)
        for lr, value_weight in init_hyper_params:
            model = compile_pbt(lr, value_weight)
            model.load_weights(init_model_file)
            # train
            print(f'training lr={lr} vw={value_weight} ...')
            history = model.fit(ds_train.shuffle(2000).batch(64), epochs=1)
            # save to disk
            model_path = f'{self.models_dir}/%s' % self._model_fname(0, lr, value_weight)
            model.save(model_path)

    def _model_fname(self, i, lr, vw):
        return f'model{i}.lr={lr};vw={vw}.h5'

    def list(self) -> List:
        """ scan model files on disk for the current list of hyper-params """

    def train(self, features_dir: str):
        """ train population on new data """

    def eval(self):
        """ round-robin games between all models; rank models; bottom 20% gets swapped out & mutated
        16 agents: 120 pairs * 50 games = 6k!!
        """


def main(argv):
    assert len(argv) == 3
    models_dir, features_dir = argv[1], argv[2]
    p = Population(models_dir)
    p.bootstrap(f'{models_dir}/model6.lr=0.005_vw=1.h5', f'{features_dir}')


if __name__ == '__main__':
    main(sys.argv)
