from __future__ import print_function

import argparse
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

parser = argparse.ArgumentParser()
parser.add_argument('core_counts', nargs='+', type=int)
parser.add_argument('--use-devs', action='store_true')
parser.add_argument('--use-inter', action='store_true')
parser.add_argument('--use-intra', action='store_true')
parser.add_argument('--no-const-fold', action='store_false', dest='const_fold')
args = parser.parse_args()

for n_cpus in args.core_counts:
    n_devs = n_cpus if args.use_devs else 1
    n_inter = n_cpus if args.use_inter else 1
    n_intra = n_cpus if args.use_intra else 1
    with tf.Session(config=tf.ConfigProto(
            device_count={ "CPU": n_devs },
            inter_op_parallelism_threads=n_inter,
            intra_op_parallelism_threads=n_intra,
    )) as sess:

        print('Running on %s CPU devices with %s inter- and %s intra-parallelism' % (
            n_devs, n_inter, n_intra))

        size = 8000

        ops = []
        feed = {}
        for i in range(n_cpus):
            d = "/cpu:%s" % (i % n_devs)
            print('  Assigning matmul to %s' % d)
            with tf.device(d):
                if args.const_fold:
                    A = tf.ones([size, size], name=("A%s" % i))
                    B = tf.ones([size, size], name=("B%s" % i))
                else:
                    A_name = "A%s" % i
                    B_name = "B%s" % i
                    A = tf.placeholder(tf.float32, shape=[size, size], name=A_name)
                    B = tf.placeholder(tf.float32, shape=[size, size], name=B_name)
                    feed["%s:0" % A_name] = np.random.rand(size, size)
                    feed["%s:0" % B_name] = np.random.rand(size, size)
                x = tf.matmul(A, B)
                ops.append(x)

        start_time = time.perf_counter()
        start_clock = time.clock()
        sess.run(ops, feed_dict=feed)
        stop_time = time.perf_counter()
        stop_clock = time.clock()

        print('  Duration (via time.perf_counter()): %f (%f - %f)' % (stop_time - start_time, stop_time, start_time))
        print('  Clock (via time.clock()): %f (%f - %f)' % (stop_clock - start_clock, stop_clock, start_clock))

        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        # sess.run([x, y, z], options=run_options, run_metadata=run_metadata)

        # for device in run_metadata.step_stats.dev_stats:
        #     device_name = device.device
        #     print(device.device)
        #     for node in device.node_stats:
        #         print("   ", node.node_name)

        # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        # chrome_trace = fetched_timeline.generate_chrome_trace_format()
        # with open('timeline_01.json', 'w') as f:
        #     f.write(chrome_trace)
