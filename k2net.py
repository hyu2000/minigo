""" modeled after dual_net w/ TF2.4 """
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

import coords
import features as features_lib
import go
import myconf


def get_features():
    return features_lib.REDUX_FEATURES  # DLGO_FEATURES


def get_features_planes():
    return features_lib.REDUX_FEATURES_PLANES  # DLGO_FEATURES_PLANES


Conv2D = keras.layers.Conv2D
conv2d_kwargs = dict(padding='same', kernel_initializer='he_normal', data_format='channels_last')


def residual_module(layer_in, n_filters, kernel_size=(3, 3)):
    """
    this aligns more w/ AGZ resnet setup
    """
    merge_input = layer_in
    # check if the number of filters needs to be increase
    if layer_in.shape[-1] != n_filters:
        x = Conv2D(n_filters, (1, 1), activation='relu', **conv2d_kwargs)(layer_in)
        merge_input = keras.layers.BatchNormalization()(x)
    # conv1
    x = Conv2D(n_filters, kernel_size, activation=None, **conv2d_kwargs)(layer_in)
    x = keras.layers.BatchNormalization()(x)
    conv1 = keras.layers.Activation('relu')(x)
    # conv2
    x = Conv2D(n_filters, kernel_size, activation=None, **conv2d_kwargs)(conv1)
    conv2 = keras.layers.BatchNormalization()(x)
    # add
    x = keras.layers.add([conv2, merge_input])
    layer_out = keras.layers.Activation('relu')(x)
    return layer_out


def build_model(input_shape):
    """
    Trainable params: 72k
    2nd round: 125k
    6-block 9x9: 396k
    """
    inputs = keras.Input(shape=input_shape)
    # add "ones" feature plain
    x = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [0, 1]], 'CONSTANT', constant_values=1)

    # block 1
    x = residual_module(x, 32, (5, 5))
    for i in range(5):
        x = residual_module(x, 64, (3, 3))

    features_common = x

    # value head
    x = Conv2D(1, (1, 1), **conv2d_kwargs)(features_common)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    # predicting win/loss now
    output_value = keras.layers.Dense(1, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.01),
                                      name='value')(x)

    # policy head
    # final conv, to get down to 1 filter (for score)
    x = Conv2D(1, (1, 1), **conv2d_kwargs)(features_common)
    move_prob = keras.layers.Flatten()(x)
    # x = tf.pad(x, [(0, 0), (0, 1)], mode='constant', constant_values=-1e6)
    pass_inputs = tf.stack([
        tf.reduce_mean(move_prob, axis=1),
        tf.reduce_max(move_prob, axis=1),
        tf.math.reduce_std(move_prob, axis=1),
        tf.squeeze(output_value, axis=1)
    ], axis=1)
    pass_prob = keras.layers.Dense(1, activation=None)(pass_inputs)
    x = tf.concat([move_prob, pass_prob], axis=1)
    output_policy = keras.layers.Activation('softmax', name='policy')(x)
    # somehow a dense layer makes it harder to train
    # output_policy = keras.layers.Dense(82, activation='softmax', name='policy')(x)

    model = keras.Model(inputs, [output_policy, output_value])

    return model


def build_model_v1(input_shape):
    """
    Trainable params: 72k
    2nd round: 125k
    6-block 9x9: 392k
    """
    inputs = keras.Input(shape=input_shape)
    # add "ones" feature plain
    x = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [0, 1]], 'CONSTANT', constant_values=1)

    # block 1
    x = residual_module(x, 32, (5, 5))
    # +2 more blocks: #params = 72k
    x = residual_module(x, 64, (3, 3))
    x = residual_module(x, 32, (3, 3))

    features_common = x

    # value head
    x = Conv2D(1, (1, 1), **conv2d_kwargs)(features_common)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    # predicting win/loss now
    output_value = keras.layers.Dense(1, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.01),
                                      name='value')(x)

    # policy head
    # final conv, to get down to 1 filter (for score)
    x = Conv2D(1, (1, 1), **conv2d_kwargs)(features_common)
    move_prob = keras.layers.Flatten()(x)
    # x = tf.pad(x, [(0, 0), (0, 1)], mode='constant', constant_values=-1e6)
    pass_inputs = tf.stack([
        tf.reduce_mean(move_prob, axis=1),
        tf.reduce_max(move_prob, axis=1),
        tf.math.reduce_std(move_prob, axis=1),
        tf.squeeze(output_value, axis=1)
    ], axis=1)
    pass_prob = keras.layers.Dense(1, activation=None)(pass_inputs)
    x = tf.concat([move_prob, pass_prob], axis=1)
    output_policy = keras.layers.Activation('softmax', name='policy')(x)
    # somehow a dense layer makes it harder to train
    # output_policy = keras.layers.Dense(82, activation='softmax', name='policy')(x)

    model = keras.Model(inputs, [output_policy, output_value])

    return model


def build_model_for_eval():
    """ load_model will try to resolve all custom objects used during training, a pain to use
    Use this, and model.load_weights() instead
    """
    # self.model = keras.models.load_model(save_file,
    #                                      custom_objects={'custom_BCE_loss': None})
    input_shape = (go.N, go.N, get_features_planes())
    model = build_model(input_shape)
    return model


class DualNetwork(object):
    def __init__(self, save_file):
        self.model_id = save_file
        model = build_model_for_eval()
        model.load_weights(save_file)
        self.model = model

    def run(self, position: go.Position):
        probs, values = self.run_many([position])
        return probs[0], values[0]

    # @tf.function(experimental_relax_shapes=True)
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, 9, 9, 11], dtype=tf.uint8),))
    def tf_run(self, input):
        """ https://www.tensorflow.org/guide/function
        """
        print('tracing...')
        return self.model(input, training=False)

    def run_many(self, positions: List[go.Position]) -> Tuple[np.ndarray, np.ndarray]:
        f = get_features()
        processed = [features_lib.extract_features(p, f) for p in positions]
        # model.predict() doc suggests to use __call__ for small batch
        # probs, values = self.model(tf.convert_to_tensor(processed), training=False)
        probs, values = self.tf_run(tf.convert_to_tensor(processed, dtype=tf.uint8))
        return probs.numpy(), values.numpy().squeeze(axis=-1)


class DummyNetwork(object):
    """ same interface as DualNetwork. Flat policy, Tromp score as value """
    def __init__(self):
        self.model_id = 'dummy'

    def run(self, position):
        probs, values = self.run_many([position])
        return probs[0], values[0]

    @staticmethod
    def zeroout_edges(probs: np.ndarray):
        prototype = probs
        for irow in (0, go.N - 1):
            prototype[irow * go.N: (irow+1) * go.N] = 0
        # disallow pass as well
        prototype[0:: go.N] = 0
        prototype[go.N - 1: go.N * go.N: go.N] = 0
        return probs / np.sum(probs)

    def run_many(self, positions: List[go.Position]) -> Tuple[np.ndarray, np.ndarray]:
        probs = np.ones(myconf.TOTAL_MOVES) / myconf.TOTAL_MOVES
        # not allow edge plays when position.n < 10
        if positions[0].n < 10:
            probs = self.zeroout_edges(probs)
        probs = np.tile(probs, (len(positions), 1))
        values = np.array([p.score() for p in positions])
        return probs, np.sign(values)


def bootstrap():
    N = go.N
    input_shape = (N, N, get_features_planes())
    model = build_model(input_shape)

    fname = '/tmp/k2net.0.h5'
    model.save(fname)


def test_filter_np():
    probs = np.ones(myconf.TOTAL_MOVES) / myconf.TOTAL_MOVES
    probs_filtered = DummyNetwork.zeroout_edges(probs)
    print(np.reshape(probs_filtered[:go.N * go.N], (go.N, go.N)))
    print(probs_filtered[-1-go.N:])


def test_load_model():
    fname = f'{myconf.EXP_HOME}/checkpoints/model7_epoch1.h5'
    model = build_model_for_eval()
    model.load_weights(fname)
    model.summary()
