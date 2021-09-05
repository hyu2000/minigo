""" we run exps in colab, so try have less code dependencies

   in main(), we only need dataset.py for data pipeline
"""
import collections
import os.path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from absl import logging, flags
import go
import myconf
import k2net as dual_net
from sgf_wrapper import SGFReader

N = go.N
HOME_DIR = myconf.EXP_HOME
FEATURES_DIR = myconf.FEATURES_DIR

from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='cpu')  # Available options are 'cpu', 'gpu', and ‘any'.


bce = tf.keras.losses.BinaryCrossentropy()
bca = tf.keras.metrics.BinaryAccuracy()


def custom_value_loss(y_true, y_pred):
    """ on top of BinaryCrossEntropy, this handles missing label
    """
    not_nan = tf.math.logical_not(tf.math.is_nan(y_true))
    # convert black_margin to win/loss
    y_true = tf.where(y_true > 0, 1, 0)
    return bce(y_true, y_pred, sample_weight=not_nan)


def custom_MSE_loss(y_true, y_pred):
    """ based on keras.losses.MeanSquaredError, but w/ special handling for
    - missing value: 0-loss
    - unknown margin: 0-loss if y_pred is in the right region. Given this is not major, just use 0
    """
    is_label_missing = tf.abs(y_true) >= 1000
    squared_loss = tf.square(y_true - y_pred)
    return tf.where(is_label_missing, 0.0, squared_loss)


class MyBinaryAccuracy(keras.metrics.Metric):
    def __init__(self, **kwargs):
        super(MyBinaryAccuracy, self).__init__(**kwargs)

        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true_org, y_pred_org, sample_weight=None):
        not_nan = tf.math.logical_not(tf.math.is_nan(y_true_org))
        y_true = tf.where(y_true_org[not_nan] > 0, 1, 0)
        y_pred_org = tf.squeeze(y_pred_org)
        y_pred = tf.where(y_pred_org[not_nan] >= 0.5, 1, 0)
        y_match = tf.cast(y_true == y_pred, tf.float32)
        # logger.info('MBCA: %s -> %s', y_true_org.shape, y_pred_org.shape)
        self.total.assign_add(tf.reduce_sum(y_match))
        self.count.assign_add(tf.cast(tf.size(y_match), tf.float32))

    def result(self):
        return self.total / self.count


def custom_value_accuracy(y_true, y_pred):
    """ ignore missing label (seems same as BCA) """
    not_nan = tf.math.logical_not(tf.math.is_nan(y_true))
    y_true = tf.where(y_true > 0, 1, 0)
    return bca(y_true, y_pred, sample_weight=not_nan)


def compile_dual():
    input_shape = (N, N, dual_net.get_features_planes())
    model = dual_net.build_model(input_shape)
    opt = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt,
                  loss={
                      'policy': 'categorical_crossentropy',
                      'value': custom_MSE_loss},
                  loss_weights={
                      'policy': 0.90,
                      'value':  0.10},
                  metrics={
                      'policy': keras.metrics.CategoricalAccuracy(name="move_acc"),
                      # 'value': custom_value_accuracy,
                      # 'value':  MyBinaryAccuracy(name="vacc")
                  })
    # model.summary()
    return model


def load_model(fname):
    logging.info('load_model %s', fname)
    model = compile_dual()
    model.load_weights(fname)
    return model


def test_save_model():
    """
    h5 is old keras full model format (rather than the TF full model format)
    This somehow doesn't have the MLC issue when loading into colab.
    To load into colab:
    model = keras.models.load_model('features/dualnet.0.h5',
                                    custom_objects={'custom_MSE_loss': custom_MSE_loss})
    """
    fname = f'{FEATURES_DIR}/dualnet.0.h5'

    model = compile_dual()
    model.summary()
    model.save(fname)
    logging.info('initial model saved to %s', fname)


def test_update_model():
    """ load existing weights, but change model config"""
    fname = f'{myconf.EXP_HOME}/checkpoints/model3_epoch_5.h5'
    model = load_model(fname)
    model.save(f'{myconf.EXP_HOME}/checkpoints/model3_epoch_5.new.h5')


def test_eval_model():
    """ on my MBP, loading TF full model works, but still needs to provide custom_objects dict """
    model = keras.models.load_model(f'{FEATURES_DIR}/../checkpoints/model0_epoch_3.h5',
                                    custom_objects={'custom_MSE_loss': custom_MSE_loss})
    fkeys = [f'beg-{i}' for i in range(1, 5)]
    fkeys.append('mid')
    fkeys.extend([f'end-{i}' for i in range(4, 0, -1)])
    for fkey in fkeys:
        ds_val = load_dataset(fkey)
        print(fkey)
        results = model.evaluate(ds_val.batch(64), return_dict=True)
        # prediction = model.predict(ds_val.batch(64))
        # print(fkey, results)


def label_top50():
    """ majority of top50 lacks RE records:    [('na', 6419), ('w', 1164), ('b', 1023)]
    let's label them!
    """
    dnn = dual_net.DualNetwork(f'{FEATURES_DIR}/../checkpoints/model2_epoch_3.h5')
    game_dir = f'{myconf.DATA_DIR}/Top50/go9'
    for sgf_fname in os.listdir(game_dir):
        if not sgf_fname.endswith('.sgf'):
            continue
        reader = SGFReader.from_file_compatible(f'{game_dir}/{sgf_fname}')
        komi = reader.komi()
        black_margin = reader.black_margin_adj(adjust_komi=True)
        pos = reader.last_pos()
        move_probs, predicted = dnn.run(pos)
        summary = ''
        if black_margin is None:
            summary = 'new'
        elif (black_margin - komi) * (predicted - komi) > 0:
            summary = 'ok'
        logging.info('%s \tkomi=%.1f raw_margin=%s predicted=%.1f %s',
                     sgf_fname, komi, black_margin, predicted, summary)


def read_tfrecord(record):
    planes = dual_net.get_features_planes()

    tfrecord_format = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'pi': tf.io.FixedLenFeature([], tf.string),
        'outcome': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_example(record, tfrecord_format)
    x = tf.io.decode_raw(parsed['x'], tf.uint8)
    x = tf.cast(x, tf.float32)

    # normally BATCH_READ_SIZE, except last batch might be smaller. -1 works for all
    batch_size = -1
    shape = [batch_size, N, N, planes]
    x = tf.reshape(x, shape)

    pi = tf.io.decode_raw(parsed['pi'], tf.float32)
    outcome = parsed['outcome']
    outcome = tf.reshape(outcome, [batch_size])

    # apply symmetries: should be able to expand #samples?
    # x, pi, outcome = apply_symmetry_dual(x, pi, outcome)
    return x, {'policy': pi, 'value': outcome}


def load_dataset(file_pattern: str):
    BATCH_READ_SIZE = 64

    # filenames = [f'{myconf.FEATURES_DIR}/pro.tfexamples']
    filenames = tf.data.Dataset.list_files(f'{FEATURES_DIR}/{file_pattern}.tfexamples')
    dataset = tf.data.TFRecordDataset(
        filenames,
        compression_type='ZLIB'
    )  # automatically interleaves reads from multiple files
    dataset = dataset.batch(BATCH_READ_SIZE).map(
        read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).unbatch()
    return dataset


def load_selfplay_data(selfplay_dir: str):
    """ many small files. need perf tune:

dataset = tf.data.TFRecordDataset(filenames_to_read,
    compression_type=None,    # or 'GZIP', 'ZLIB' if compress you data.
    buffer_size=10240,        # any buffer size you want or 0 means no buffering
    num_parallel_reads=os.cpu_count()  # or 0 means sequentially reading
    )

# Maybe you want to prefetch some data first.
dataset = dataset.prefetch(buffer_size=batch_size)

    """
    BATCH_READ_SIZE = 64

    filenames = tf.data.Dataset.list_files(f'{selfplay_dir}/*.tfrecord.zz')
    dataset = tf.data.TFRecordDataset(
        filenames,
        compression_type='ZLIB',
        num_parallel_reads=8
        # automatically interleaves reads from multiple files
    )
    dataset = dataset.batch(BATCH_READ_SIZE).prefetch(tf.data.AUTOTUNE).map(
        read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).unbatch()
    return dataset


def train():
    # model = compile_dual()
    model = load_model(f'{myconf.MODELS_DIR}/model4_epoch_1.h5')

    data_dir = myconf.SELFPLAY_DIR
    # data_dir = f'{myconf.EXP_HOME}/selfplay'
    ds_train = load_selfplay_data(f'{data_dir}/train')
    ds_val = load_selfplay_data(f'{data_dir}/val')
    callbacks = [
        keras.callbacks.ModelCheckpoint(f'{myconf.MODELS_DIR}/model_epoch_{{epoch}}.h5'),
        # keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    ]
    history = model.fit(ds_train.shuffle(1000).batch(64), validation_data=ds_val.batch(64),
                        epochs=5, callbacks=callbacks)
    print(history.history)


if __name__ == '__main__':
    train()
    # test_save_model()
    # test_eval_model()
    # label_top50()
    # test_update_model()
