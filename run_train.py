""" we run exps in colab, so try have less code dependencies

   in main(), we only need dataset.py for data pipeline
"""
import collections
import os.path
import random
import sys
from typing import List, Tuple

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


bce = tf.keras.losses.BinaryCrossentropy()
bca = tf.keras.metrics.BinaryAccuracy()


def custom_value_loss(y_true, y_pred):
    """ on top of BinaryCrossEntropy, this handles missing label
    """
    not_nan = tf.math.logical_not(tf.math.is_nan(y_true))
    # convert black_margin to win/loss
    y_true = tf.where(y_true > 0, 1, 0)
    return bce(y_true, y_pred, sample_weight=not_nan)


def custom_BCE_loss(y_true, y_pred):
    """ basically BinaryCrossEntropy, just that label is 1/-1, and y_pred with tanh activation (rather than sigmoid)
    """
    y_true = tf.where(y_true > 0, 1, 0)
    y_pred = (y_pred + 1) * 0.5
    return bce(y_true, y_pred)


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
    # SAI: 1e-4
    # KataGo: per-sample learning rate of 6e-5, except 2e-5 for the first 5mm samples
    # 1e-3 / 64 = 1.6e-5
    opt = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt,
                  loss={
                      'policy': 'categorical_crossentropy',
                      'value':  custom_BCE_loss},
                  loss_weights={
                      'policy': 0.50,
                      'value':  0.50},
                  metrics={
                      'policy': keras.metrics.CategoricalAccuracy(name="move_acc"),
                      # 'value': custom_value_accuracy,
                      # 'value':  MyBinaryAccuracy(name="vacc")
                  })
    # model.summary()
    return model


def recompile_value_loss_only(model):
    """ recompile a dual model to have only value loss, essentially ignoring policy head"""
    model.compile(optimizer=model.optimizer,
                  loss={
                      'value':  custom_BCE_loss},
                  loss_weights={
                      'value':  0.75},
                  metrics={
                      # 'policy': keras.metrics.CategoricalAccuracy(name="move_acc"),
                      # 'value': custom_value_accuracy,
                      # 'value':  MyBinaryAccuracy(name="vacc")
                  })


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
    fname = f'{myconf.EXP_HOME}/checkpoints/model10_epoch4.h5'
    model = load_model(fname)
    model.summary()
    # model.save(f'{myconf.EXP_HOME}/checkpoints/model3_epoch_5.new.h5')


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


def load_selfplay_data(selfplay_dir_pattern: str, subtype: str = ''):
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

    if subtype:
        filenames = tf.data.Dataset.list_files(f'{selfplay_dir_pattern}/*.tfrecord.{subtype}.zz')
    else:
        filenames = tf.data.Dataset.list_files(f'{selfplay_dir_pattern}/*.tfrecord.zz')
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


def test_eval_model13_on_selfplay():
    """ the opposite of eval_model_on_selfplay13: we try to assess selfplay data quality """
    model_path = f'{myconf.EXP_HOME}/models-old/model13_epoch2.h5'
    model = load_model(model_path)

    for selfplay_iter in range(0, 5):
        print(f'====== evaluating on selfplay {selfplay_iter}')
        ds_val = load_selfplay_data(f'{myconf.EXP_HOME}/selfplay{selfplay_iter}/train', 'full')
        results = model.evaluate(ds_val.batch(64), return_dict=True, verbose=2)


def test_eval_model_on_selfplay13():
    """ eval various model on policy/value target from selfplay13 """
    # model = keras.models.load_model(f'{FEATURES_DIR}/../checkpoints/model0_epoch_3.h5',
    #                                 custom_objects={'custom_MSE_loss': custom_MSE_loss})
    ds_val = load_selfplay_data(f'{myconf.EXP_HOME}/exps-on-old-models/selfplay13/train', 'full')
    for i_gen in range(5, 6):
        for epoch in range(2, 3):
            model_id = f'model{i_gen}_epoch{epoch}'
            print('='*10, model_id)
            model = load_model(f'{myconf.EXP_HOME}/checkpoints/{model_id}.h5')
            results = model.evaluate(ds_val.batch(64), return_dict=True, verbose=2)


def dataset_split(ds: tf.data.Dataset, train_val_ratio: int = 9) -> Tuple:
    """ this split on sample level, probably better on the game level?
    https://stackoverflow.com/questions/48213766/split-a-dataset-created-by-tensorflow-dataset-api-in-to-train-and-test
    https://www.tensorflow.org/api_docs/python/tf/data/Dataset#window
    """
    f_flatmap = lambda *ds: ds[0] if len(ds) == 1 else tf.data.Dataset.zip(ds)
    ds_train = ds.window(train_val_ratio, train_val_ratio + 1).flat_map(f_flatmap)
    ds_val = ds.skip(train_val_ratio).window(1, train_val_ratio + 1).flat_map(f_flatmap)
    return ds_train, ds_val


def train_bootstrap():
    """ train on g170 data """
    # model = compile_dual()
    model = load_model(f'{myconf.MODELS_DIR}/model7_6.h5')

    data_dir = myconf.SELFPLAY_DIR
    data_dir = f'{myconf.FEATURES_DIR}/enhance'
    ds_all = load_selfplay_data(f'{data_dir}')
    # ds_val = load_selfplay_data(f'{data_dir}/val')
    callbacks = [
        keras.callbacks.ModelCheckpoint(f'{myconf.MODELS_DIR}/model7_{{epoch}}.h5'),
        # keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    ]
    ds_train, ds_val = dataset_split(ds_all.shuffle(1000, seed=2023))
    history = model.fit(ds_train.batch(64), validation_data=ds_val.batch(64),
                        initial_epoch=6, epochs=4, callbacks=callbacks, verbose=2)
    print(history.history)


def train_local():
    # model = compile_dual()
    model = load_model(f'{myconf.MODELS_DIR}/model7_10.h5')
    # model = load_model(f'{myconf.MODELS_DIR}/model0_0.h5')
    data_dir = f'{myconf.EXP_HOME}/selfplay6p/enhance'
    ds_train = load_selfplay_data(f'{data_dir}/train', subtype='full')
    # ds_val   = load_selfplay_data(f'{data_dir}/val1')
    callbacks = [
        keras.callbacks.ModelCheckpoint(f'{myconf.MODELS_DIR}/model8_{{epoch}}.h5'),
        # keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    ]
    history = model.fit(ds_train.shuffle(4000).batch(64),
                        # validation_data=ds_val.batch(64),
                        epochs=4, callbacks=callbacks, verbose=2)
    return model


def _ds_size(dataset: tf.data.Dataset):
    # does it have to be difficult?
    num_samples = dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()
    return num_samples


def _ds_take_every(dataset: tf.data.Dataset, n):
    # why is random sampling so hard!
    # why can't I chain these two ops together!
    ds_train = dataset.enumerate().filter(lambda i, x: i % n == 0)
    return ds_train.map(lambda i, x: x)


def convert_tf_to_coreml(tf_fpath):
    model_dir = os.path.dirname(tf_fpath)
    basename, _ = os.path.splitext(os.path.basename(tf_fpath))
    mlmodel = dual_net.CoreMLNet.convert_tf2_to_coreml(tf_fpath)
    mlmodel.save(f'{model_dir}/{basename}.mlpackage')


def train(argv: List):
    assert len(argv) >= 4
    train_dir  = argv[1]  # can be a dir pattern, e.g. "tfrecords/enhance*/train". Needs quote.
    model_dir  = argv[2]
    start_model = argv[3]  # either start_iter "3" or with epoch "3_4"
    val_dir = argv[4] if len(argv) > 4 else None

    if '_' in start_model:
        start_iter = int(start_model.split('_')[0])
    else:
        DEFAULT_START_EPOCH = 4
        start_iter = int(start_model)
        start_model = f'{start_iter}_{DEFAULT_START_EPOCH}'

    new_iter = start_iter + 1
    print(f'train on {train_dir}: {start_model} -> {new_iter}')

    model_file = f'{model_dir}/model{start_model}.h5'
    if start_iter == 0 and not os.path.exists(model_file):
        logging.info('Using random initialization')
        model = compile_dual()
    else:
        model = load_model(model_file)
    ds_train = load_selfplay_data(train_dir, 'full')
    if val_dir:
        ds_val = load_selfplay_data(val_dir, 'full')

    callbacks = [
        keras.callbacks.ModelCheckpoint(f'{model_dir}/model{new_iter}_{{epoch}}.h5'),
        # keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    ]
    NUM_EPOCHS = 4
    if val_dir:
        # SYMMETRIES_USED = 8
        # if SYMMETRIES_USED < 8:
        #     ds_train = _ds_take_every(ds_train, 8 / SYMMETRIES_USED)
        # batches_without_symmetry = _ds_size(ds_train) // (SYMMETRIES_USED * 64)
        # NUM_PASSES_OVER_DATA = 4
        # ds_train = ds_train.repeat(NUM_PASSES_OVER_DATA)
        # history = model.fit(ds_train.shuffle(4000).batch(64),
        #                     epochs=NUM_PASSES_OVER_DATA * SYMMETRIES_USED, steps_per_epoch=batches_without_symmetry,
        #                     validation_data=ds_val.batch(64), callbacks=callbacks, verbose=2)
        history = model.fit(ds_train.shuffle(4000).batch(64), epochs=NUM_EPOCHS, steps_per_epoch=None,
                            validation_data=ds_val.batch(64), callbacks=callbacks, verbose=2)
    else:
        history = model.fit(ds_train.shuffle(4000).batch(64), epochs=NUM_EPOCHS, callbacks=callbacks, verbose=2)
    print(history.history)

    for epoch in range(1, 1+NUM_EPOCHS):
        convert_tf_to_coreml(f'{model_dir}/model{new_iter}_{epoch}.h5')


if __name__ == '__main__':
    train(sys.argv)
    # train_local()
    # train_bootstrap()
    # test_save_model()
    # test_eval_model()
    # label_top50()
    # test_update_model()
