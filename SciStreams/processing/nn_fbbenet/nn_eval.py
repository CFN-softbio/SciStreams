# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score
import yaml

from . import model
from . import nn_input

# define a global graph
# (helps when distributing on cluster)
eval_graph = tf.Graph()

FLAGS = tf.app.flags.FLAGS

# architecture is now determined by cmd args
# tf.app.flags.DEFINE_string('architecture', 'conv2',
#                            """Network acrhitecture.""")
# tf.app.flags.DEFINE_string('eval_dir', '../xray_data/fbb_output/',
#                            """Directory where to write event logs.""")
# tf.app.flags.DEFINE_string('eval_data', 'train_eval',
#                            """Either 'test' or 'train_eval'.""")
# tf.app.flags.DEFINE_string('checkpoint_dir', '../xray_data/fbb_output/',
#                            """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
# tf.app.flags.DEFINE_integer('num_examples',
#                             nn_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL,
#                             """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")
eval_dir = '' # FLAGS.eval_dir + FLAGS.architecture + '_eval'
checkpoint_dir = '' # FLAGS.checkpoint_dir + FLAGS.architecture + '_train'
# if FLAGS.eval_data == 'test':
#   num_examples = nn_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
# else:
#   num_examples = nn_input.NUM_EXAMPLES_PER_EPOCH_FOR_VAL

#config_path = './run_config.yml'
#run_config = yaml.safe_load(open(config_path))
batch_size = FLAGS.batch_size


def eval_once(saver, prob_op, gt_op, conv_feat_op, mask=None, output=False):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    prob_op: Probability op.
    gt_op: Ground truth op.
    summary_op: Summary op.
    mask: indicate the entries used for eval.
  """
  # Using eval_graph global
  with tf.Session(graph=eval_graph) as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(run_config['num_examples'] / batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * batch_size
      result_size = [total_sample_count, model.NUM_CLASSES]
      pred_all = np.zeros(result_size)
      gt_all = np.zeros(result_size)
      feature_all = np.zeros([total_sample_count, conv_feat_op.get_shape()[1]])
      step = 0
      while step < num_iter and not coord.should_stop():
        pred, gt, conv = sess.run([prob_op, gt_op, conv_feat_op])
        pred_all[step*batch_size:(step+1)*batch_size,:] = pred
        gt_all[step*batch_size:(step+1)*batch_size,:] = gt
        if output:
          feature_all[step*batch_size:(step+1)*batch_size,:] = conv

        print('%d / %d' % (step + 1, num_iter))
        step += 1

      # Compute precision @ 1.
      if mask is not None:
        pred_all = pred_all[:, mask]
        gt_all = gt_all[:, mask]
      gt_all = gt_all.astype(np.int)
      pred_all_th = (pred_all > 0.5).astype(np.int)
      true_count = (pred_all_th == gt_all).astype(np.int).sum()
      precision = true_count / gt_all.size
      ap = average_precision_score(gt_all, pred_all, average=None)
      map = np.mean(ap)
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
      print('AP = ')
      print(ap)
      print('mAP = %f' % map)
      np.save('pred.npy', pred_all)
      np.save('gt.npy', gt_all)
      if output:
        np.save('features.npy', feature_all)

      #summary = tf.Summary()
      #summary.ParseFromString(sess.run(summary_op))
      #summary.value.add(tag='Precision @ 1', simple_value=precision)
      #summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
    return pred_all, gt_all


def evaluate(architecture='joint', output=False):
  """Eval CIFAR-10 for a number of steps."""
  #global num_examples, batch_size  # will reset if data is from file list
  mask = None  # used for inconsistent tags (real data)
  # using the global graph
  with eval_graph.as_default() as g:
    # Get images and labels for CIFAR-10.
    # eval_data = FLAGS.eval_data == 'test'
    _, coefs, images, labels = model.inputs(num_threads=1)
    # if eval_data:
    #   mask = np.array([0, 1, 2, 3, 4, 10, 11, 13, 14])

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, _, _, conv_features = model.inference(coefs, images, architecture=architecture)
    probs = tf.sigmoid(logits)

    ## Calculate predictions.
    #top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # # Build the summary operation based on the TF collection of Summaries.
    # summary_op = tf.merge_all_summaries()
    #
    # summary_writer = tf.train.SummaryWriter(eval_dir, g)

    while True:
      eval_once(saver, probs, labels, conv_features, mask, output=output)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  #alexnet.maybe_download_and_extract()
  try:
    architecture = argv[argv.index('-a') + 1]
  except Exception:
    architecture = 'joint'
  output = '-o' in argv[1:]
  print('Architecture: ' + architecture)
  global eval_dir, checkpoint_dir
  eval_dir = run_config['train_dir'] + architecture + '_eval'
  checkpoint_dir = run_config['train_dir'] + architecture + '_train'
  tf.app.flags.DEFINE_string('checkpoint_dir', checkpoint_dir,
                             """Directory where to read model checkpoints.""")

  if tf.gfile.Exists(eval_dir):
    tf.gfile.DeleteRecursively(eval_dir)
  tf.gfile.MakeDirs(eval_dir)
  evaluate(architecture=architecture, output=output)


if __name__ == '__main__':
  tf.app.run()
