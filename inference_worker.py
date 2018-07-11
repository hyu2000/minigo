# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Worker process for running remote inference.
The worker wraps the inference model in an infinte loop: input features are
fetched via RPC at the top of the loop, and inference output is written back
at the bottom (again, via RPC).
"""

import sys
import time
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.training import saver
from tensorflow.contrib.proto.python.ops import decode_proto_op
from tensorflow.contrib.proto.python.ops import encode_proto_op
import signal
import threading
import numpy as np
from absl import flags
import grpc
from proto import inference_service_pb2
from proto import inference_service_pb2_grpc
import dual_net
import features as features_lib
import go

flags.DEFINE_string("model", "", "Path to the TensorFlow model.")

flags.DEFINE_string("checkpoint_dir", "",
                    "Path to a directory containing TensorFlow model "
                    "checkpoints. The inference worker will monitor this "
                    "when a new checkpoint is found, load the model and use it "
                    "for futher inferences.")

flags.DEFINE_string("server_address", "localhost:50051",
                    "Inference server local address.")

flags.DEFINE_string("descriptor",
                    "proto/inference_service_py_pb2.pb.descriptor_set",
                    "Path to the InferenceService proto descriptor.")

flags.DEFINE_integer("parallel_tpus", 8,
                     "Number of TPU cores to run on in parallel.")

FLAGS = flags.FLAGS


# The default maximum receive RPC size is only 4MB, which isn't large enough
# for our messages.
GRPC_OPTIONS = [
    ("grpc.max_message_length", 50 *  1024 * 1024),
    ("grpc.max_receive_message_length", 50 *  1024 * 1024),
]

NUM_WORKER_THREADS = 2


class RwLock(object):
    """A simple read/write mutex.

    I'm kind of surprised Python doesn't provide one of these by default.
    """
    def __init__(self):
        self._resource_lock = threading.Semaphore()
        self._read_lock = threading.Semaphore()
        self._read_count = 0

    def acquire_write(self):
        self._resource_lock.acquire()

    def release_write(self):
        self._resource_lock.release()

    def acquire_read(self):
        with self._read_lock:
            self._read_count += 1
            if self._read_count == 1:
                self._resource_lock.acquire()

    def release_read(self):
        with self._read_lock:
            self._read_count -= 1
            if self._read_count == 0:
                self._resource_lock.release()


def const_model_inference_fn(features):
    """Builds the model graph with weights marked as constant.

    This improves TPU inference performance because it prevents the weights
    being transferred to the TPU every call to Session.run().

    Returns:
        (policy_output, value_output, logits) tuple of tensors.
    """
    def custom_getter(getter, name, *args, **kwargs):
        with tf.control_dependencies(None):
            return tf.guarantee_const(
                getter(name, *args, **kwargs), name=name+"/GuaranteeConst")
    with tf.variable_scope("", custom_getter=custom_getter):
        return dual_net.model_inference_fn(features, False)


class Worker(object):
    def __init__(self):
        # Event that gets set after a model is loaded.
        # The worker threads wait for this event before starting inference.
        self.model_available = threading.Event()

        self._get_server_config()
        self._init_sess()

    def run(self):
        self._run_threads()
        self._shutdown_sess()

    def _get_server_config(self):
        while True:
            try:
                channel = grpc.insecure_channel(FLAGS.server_address)
                self.stub = inference_service_pb2_grpc.InferenceServiceStub(
                    channel)
                config = self.stub.GetConfig(
                    inference_service_pb2.GetConfigRequest())
                break
            except grpc.RpcError:
                print("Waiting for server", flush=True)
                time.sleep(1)

        if config.board_size != go.N:
            raise RuntimeError("Board size mismatch: server=%d, worker=%d" % (
                config.board_size, go.N))

        positions_per_inference = (config.games_per_inference *
                                   config.virtual_losses)
        if positions_per_inference % FLAGS.parallel_tpus != 0:
            raise RuntimeError(
                "games_per_inference * virtual_losses must be divisible by "
                "parallel_tpus")
        self.batch_size = positions_per_inference // FLAGS.parallel_tpus

        print("parallel_tpus = %d" % FLAGS.parallel_tpus)
        print("games_per_inference = %d" % config.games_per_inference)
        print("virtual_losses = %d" % config.virtual_losses)
        print("positions_per_inference = %d" % positions_per_inference)
        print("batch_size = %d" % self.batch_size)
        sys.stdout.flush()

    def _init_sess(self):
        tpu_grpc_url = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu=[FLAGS.tpu_name]).get_master()
        self.sess = tf.Session(tpu_grpc_url)

        self.feature_placeholders = []
        with self.sess.graph.as_default():
            for i in range(FLAGS.parallel_tpus):
                features = tf.placeholder(
                    tf.float32, [None, go.N, go.N, features_lib.NEW_FEATURES_PLANES],
                    name='pos_tensor')
                self.feature_placeholders.append((features,))

            self.outputs = tf.contrib.tpu.replicate(
                const_model_inference_fn, self.feature_placeholders)

            # tpu.replicate requires a list, but sess.run requires a tuple...
            self.feature_placeholders = tuple(self.feature_placeholders)

            # TODO(tommadams): remove this
            if FLAGS.model:
                tf.train.Saver().restore(self.sess, FLAGS.model)
                self.model_available.set()

        print("initializing tpu", flush=True)
        tpu_init = tf.contrib.tpu.initialize_system()
        self.sess.run(tpu_init)
        print("tpu ready", flush=True)

    def _run_threads(self):
        """Run inference threads and optionally a thread that updates the model.

        Synchronization between the inference threads and the model update
        thread is performed using a RwLock that protections access to self.sess.
        The inference threads enter the critical section using a read lock, so
        they can both run inference concurrently. The model update thread enters
        the critical section using a write lock for exclusive access.
        """
        self.lock = RwLock()
        self.model_path = None

        threads = []
        if FLAGS.checkpoint_dir:
            threads.append(threading.Thread(target=self._checkpoint_thread))
        for i in range(NUM_WORKER_THREADS):
            threads.append(threading.Thread(target=self._worker_thread))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def _checkpoint_thread(self):
        print("starting model loader thread", flush=True)
        while True:
            freshest = saver.latest_checkpoint(FLAGS.checkpoint_dir)
            print("freshest model = %s" % freshest, flush=True)
            if not freshest or freshest == self.model_path:
                time.sleep(10)
                continue

            self.lock.acquire_write()
            print("loading %s" % freshest, flush=True)
            try:
                with self.sess.graph.as_default():
                    tf.train.Saver().restore(self.sess, freshest)
                self.model_path = freshest
                self.model_available.set()
            finally:
                self.lock.release_write()

    def _worker_thread(self):
        num_board_features = go.N * go.N * features_lib.NEW_FEATURES_PLANES

        print("waiting for model", flush=True)
        self.model_available.wait()
        while True:
            features_response = self.stub.GetFeatures(
                inference_service_pb2.GetFeaturesRequest())
            all_features = features_response.features

            features = []
            num_features = self.batch_size * num_board_features
            for i in range(FLAGS.parallel_tpus):
                begin = i * num_features
                end = begin + num_features
                x = np.frombuffer(
                    all_features, dtype=np.int8, count=num_features, offset=begin)
                x = x.reshape([self.batch_size, go.N, go.N,
                               features_lib.NEW_FEATURES_PLANES])
                features.append(x)

            try:
                self.lock.acquire_read()
                outputs = self.sess.run(self.outputs,
                                        {self.feature_placeholders: features})
                # Make a local copy of self.model_path while this worker has
                # the read lock.
                local_model_path = self.model_path
            finally:
                self.lock.release_read()

            flat_policy = []
            value = []
            for x in outputs:
                flat_policy.extend(x[0])
                value.extend(x[1])

            put_outputs_request = inference_service_pb2.PutOutputsRequest(
                 batch_id=features_response.batch_id,
                 policy=np.concatenate(flat_policy), value=value,
                 model_path=local_model_path)
            self.stub.PutOutputs(put_outputs_request)

    def _shutdown_sess(self):
        tpu_shutdown = tf.contrib.tpu.shutdown_system()
        print("shutting down TPU", flush=True)
        self.sess.run(tpu_shutdown)
        print("all done!", flush=True)


def main():
    # Tell Python to use the system's default signal handler when handling
    # SIGINT. This will stop the inference worker child process becoming
    # orphaned in some cases when the process group is interrupted (e.g. with
    # ctrl-C).
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    tf.logging.set_verbosity(tf.logging.DEBUG)
    worker = Worker()
    worker.run()


if __name__ == "__main__":
    flags.FLAGS(sys.argv, known_only=True)
    main()
