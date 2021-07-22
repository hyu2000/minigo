"""
apply symmetries to tfrecords. Ideally we can do this on-the-fly, but would need to ensure
all np transforms are properly mapped to tensors w/ batching
"""
import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow import keras
import go
import myconf
import k2net as dual_net
import preprocessing
import utils
from symmetries import apply_symmetry_dual

N = go.N


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

    return x, {'policy': pi, 'value': outcome}


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

    filenames = tf.data.Dataset.list_files(f'{selfplay_dir}/*.tfrecord.zz')  #.take(3)
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


def sample_generator(ds):
    i = 0
    for datum in ds:
        x_tensor, y_dict = datum
        # feature needs to be uint8!
        x_org  = tf.cast(x_tensor, tf.uint8).numpy()
        pi_org = y_dict['policy'].numpy()
        v_org  = y_dict['value'].numpy()
        x_new, pi_new, outcome_new = apply_symmetry_dual(x_org, pi_org, v_org)
        for x, pi, value in zip(x_new, pi_new, outcome_new):
            yield preprocessing.make_tf_example(x, pi, value)


def test_iter_chunks():
    it = np.ones((32, 3))
    for i, group in enumerate(utils.iter_chunks(10, it)):
        print(i, len(group), sum(group))


def main():
    data_dir = f'{myconf.EXP_HOME}/selfplay'
    for tag in ['train', 'val']:
        source_data_dir = f'{data_dir}/{tag}'
        output_data_dir = f'{data_dir}/{tag}-symmetries'
        ds = load_selfplay_data(source_data_dir)

        try:
            print(f'Removing {output_data_dir}')
            shutil.rmtree(output_data_dir)
        except FileNotFoundError:
            pass
        utils.ensure_dir_exists(output_data_dir)

        for i, tf_examples in enumerate(utils.iter_chunks(10000, sample_generator(ds))):
            fname = f'{output_data_dir}/chunk-{i}.tfrecord.zz'
            print(f'{tag} chunk {i}: writing %d records' % len(tf_examples))
            preprocessing.write_tf_examples(fname, tf_examples)

        print('swap in enhanced data dir')
        shutil.move(source_data_dir, f'{source_data_dir}-org')
        shutil.move(output_data_dir, source_data_dir)


if __name__ == '__main__':
    main()
    # test_iter_chunks()