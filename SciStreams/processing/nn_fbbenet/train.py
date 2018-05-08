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

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import yaml

from nn_fbbenet import model

FLAGS = tf.app.flags.FLAGS

# architecture is now determined by cmd args
# tf.app.flags.DEFINE_string('architecture', 'conv2',
#                            """Network acrhitecture.""")
# tf.app.flags.DEFINE_string('train_dir', '../xray_data/fbb_output/',
#                            """Directory where to write event logs """
#                            """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
train_dir = '' # FLAGS.train_dir + FLAGS.architecture + '_train'
config_path = './run_config.yml'
run_config = yaml.safe_load(open(config_path))
pretrain_dir = '../xray_data/5layer_dropout_train'


def train(architecture='joint', from_checkpoint=False):
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # Get images and labels for CIFAR-10.
    _, coefs, images, labels = model.distorted_inputs()
    #if architecture == 'fbb':
    #  coefs, images, labels = model.rebalanced_distorted_inputs('../xray_data/fbb_output/more_symmetry_binary')
    #else:
    #  coefs, images, labels = model.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, w_coefs, w_images, _ = model.inference(coefs, images, architecture=architecture)

    # Calculate loss.
    loss = model.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = model.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    if architecture == 'fbb':
        if from_checkpoint:
            ckpt = tf.train.get_checkpoint_state(train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step_cnt = global_step.eval(session=sess)
                print('Resuming from checkpoint: step %d' % global_step_cnt)
            else:
                print('Checkpoint not found')
                from_checkpoint = False
        if not from_checkpoint:
            print('Training from fresh start')
            if tf.gfile.Exists(train_dir):
                tf.gfile.DeleteRecursively(train_dir)
            tf.gfile.MakeDirs(train_dir)
            sess.run(init)
            global_step_cnt = 0
    elif architecture == 'cnn':
        if from_checkpoint:
            ckpt = tf.train.get_checkpoint_state(train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step_cnt = global_step.eval(session=sess)
                print('Resuming from checkpoint: step %d' % global_step_cnt)
            else:
                print('Checkpoint not found')
                from_checkpoint = False
        if not from_checkpoint:
            # print('Training (fine-tuning) from pretrained vanilla 5 layer model')
            # loader_images = tf.train.Saver(w_images)
            # if tf.gfile.Exists(train_dir):
            #     tf.gfile.DeleteRecursively(train_dir)
            # tf.gfile.MakeDirs(train_dir)
            # ckpt = tf.train.get_checkpoint_state(pretrain_dir)
            # if ckpt and ckpt.model_checkpoint_path:
            #     sess.run(init)
            #     loader_images.restore(sess, ckpt.model_checkpoint_path)
            # else:
            #     raise Exception('Pretrained model not found')
            # global_step_cnt = 0
            print('Training from fresh start')
            if tf.gfile.Exists(train_dir):
                tf.gfile.DeleteRecursively(train_dir)
            tf.gfile.MakeDirs(train_dir)
            sess.run(init)
            global_step_cnt = 0
    elif architecture == 'joint':
        if from_checkpoint:
            ckpt = tf.train.get_checkpoint_state(train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step_cnt = global_step.eval(session=sess)
                print('Resuming from checkpoint: step %d' % global_step_cnt)
            else:
                print('Checkpoint not found')
                from_checkpoint = False
        if not from_checkpoint:
            print('Training (fine-tuning) from pretrained models')
            loader_images = tf.train.Saver(w_images)
            loader_coefs = tf.train.Saver(w_coefs)
            pretrain_image_dir = run_config['train_dir'] + 'cnn_train'
            pretrain_coef_dir = run_config['train_dir'] + 'fbb_train'
            if tf.gfile.Exists(train_dir):
                tf.gfile.DeleteRecursively(train_dir)
            tf.gfile.MakeDirs(train_dir)
            ckpt_image = tf.train.get_checkpoint_state(pretrain_image_dir)
            ckpt_coef = tf.train.get_checkpoint_state(pretrain_coef_dir)
            sess.run(init)
            if ckpt_image and ckpt_image.model_checkpoint_path:
                loader_images.restore(sess, ckpt_image.model_checkpoint_path)
            else:
                raise Exception('Pretrained CNN model not found')
            if ckpt_coef and ckpt_coef.model_checkpoint_path:
                loader_coefs.restore(sess, ckpt_coef.model_checkpoint_path)
            else:
                raise Exception('Pretrained FBB model not found')
            global_step_cnt = 0
    else:
        raise ValueError('Unrecognized architecture')

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

    for step in xrange(global_step_cnt, FLAGS.max_steps):
      start_time = time.time()

      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  #alexnet.maybe_download_and_extract()
  try:
    architecture = argv[argv.index('-a') + 1]
  except Exception:
    architecture = 'joint'
  print('Architecture: ' + architecture)
  global train_dir
  train_dir = run_config['train_dir'] + architecture + '_train'

  train(architecture=architecture, from_checkpoint=not '--reset' in argv)


if __name__ == '__main__':
  tf.app.run()

