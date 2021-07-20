import numpy as np
import tensorflow as tf
from tensorflow import keras
import go
import myconf
import k2net as dual_net
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

    filenames = tf.data.Dataset.list_files(f'{selfplay_dir}/*.tfrecord.zz').take(1)  # todo
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


def main():
    data_dir = f'{myconf.EXP_HOME}/selfplay'
    ds = load_selfplay_data(f'{data_dir}/train')
    cnt = 0
    for datum in ds:
        cnt += 1
    x_tensor, y_dict = datum
    x = x_tensor.numpy()
    pi = y_dict['policy'].numpy()
    outcome = y_dict['value'].numpy()
    x_new, pi_new, outcome_new = apply_symmetry_dual(x, pi, outcome)
    print(x_new.shape, pi_new.shape, outcome_new.shape)


if __name__ == '__main__':
    main()
