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

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 227

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10 # this is not mutually exclusive! it's actually real_label_bytes
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 45000
NUM_EXAMPLES_PER_EPOCH_FOR_VAL = 5000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2000
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN_AUG = 5000

# import math
# UINT32_MAX = 4294967295
# c_log_image = UINT32_MAX / math.log(1 + UINT32_MAX)


def read_data(filename_queue):
  """Reads and parses examples from CIFAR10 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class DataRecord(object):
    pass
  result = DataRecord()
  coef = DataRecord()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 32  # 2 for CIFAR-100
  coef_height = 40
  coef_width = 11
  coef_depth = 2
  result.height = 256
  result.width = 256
  result.depth = 1
  coef_bytes = coef_height * coef_width * coef_depth * 4
  image_bytes = result.height * result.width * result.depth

  record_bytes = label_bytes + coef_bytes + image_bytes
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # 2-format raw data reader
  uint_tensor = tf.decode_raw(value, tf.uint8)
  float_tensor = tf.decode_raw(value, tf.float32)
  result.label = tf.cast(
    tf.slice(uint_tensor, [0], [NUM_CLASSES]), tf.int32
  )
  coef = tf.reshape(tf.slice(float_tensor, [int(label_bytes / 4)],
                             [coef_height * coef_width * coef_depth]),
                    [coef_depth, coef_height, coef_width])
  coef = tf.transpose(coef, [1, 2, 0])
  normalized_coef = 255 * coef / tf.maximum(tf.reduce_max(tf.abs(coef)), 1.)
  #result.coef = normalized_coef
  #result.coef = tf.log(normalized_coef + 1) / tf.log(256.)
  result.coef = tf.multiply(tf.sign(normalized_coef), tf.log(tf.abs(normalized_coef) + 1) / np.log(256.))
  # result.coef = tf.transpose(coef, [1, 2, 0])

  #result.coef = tf.slice(tf.transpose(coef, [1, 2, 0]), [0, 0, 0], [-1, -1, -1])
  image = tf.reshape(
    tf.slice(uint_tensor, [label_bytes + coef_bytes],
             [result.height * result.width * result.depth]),
    [result.depth, result.height, result.width]
  )
  result.image = tf.transpose(image, [1, 2, 0])

  return result


def _generate_image_and_label_batch(key, coef, image, label, min_queue_examples,
                                    batch_size, shuffle, num_threads=16,
                                    input_type=None):
  """
  1D curve added.

  Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  #num_preprocess_threads = 16
  if shuffle:
    keys, coefs, images, labels = tf.train.shuffle_batch(
        [key, coef, image, label],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    keys, coefs, images, labels = tf.train.batch(
        [key, coef, image, label],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  if input_type is None:
    tf.summary.image('images', images, max_outputs=20)
  else:
    tf.summary.image('images_' + input_type, images, max_outputs=20)
  # cs = coefs.get_shape()
  # empty_dim = tf.zeros([cs[0].value, cs[1].value, cs[2].value, 1])
  # summary_coefs = tf.concat([coefs, empty_dim], 3)
  # if input_type is None:
  #   tf.summary.image('coefs', summary_coefs, max_outputs=20)
  # else:
  #   tf.summary.image('coefs_' + input_type, summary_coefs, max_outputs=20)

  return keys, coefs, images, labels


def distorted_inputs(run_config, batch_size, shuffle=True, input_type=None, num_threads=16, nodistort=False):
  """
  1D curve added.

  Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # if input_type is None:
  #   filenames = [os.path.join(data_dir, 'batch-%d.bin' % i)
  #                for i in range(9)]
  # elif input_type == 'aug':
  #   # augmented symmetry samples dataset
  #   filenames = [os.path.join(data_dir, 'batch-0.bin')]
  # else:
  #   raise ValueError('Unrecognized input type')
  filenames = run_config['record_paths']
  num_examples = run_config['num_examples']

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_data(filename_queue)
  reshaped_image = tf.cast(read_input.image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for training the network. Note the many random
  # distortions applied to the image.


  if nodistort:
    # no random distortion (crop) when running through training set for just
    # exporting fc features
    distorted_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                             width, height)
  else:
    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 1])
  #
  # # Randomly flip the image horizontally.
  # distorted_image = tf.image.random_flip_left_right(distorted_image)
  #
  # # Because these operations are not commutative, consider randomizing
  # # the order their operation.
  # distorted_image = tf.image.random_brightness(distorted_image,
  #                                              max_delta=63)
  # distorted_image = tf.image.random_contrast(distorted_image,
  #                                            lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  normalized_image = tf.image.per_image_standardization(distorted_image)

  normalized_coef = tf.image.per_image_standardization(read_input.coef)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  # if input_type is None:
  #   min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
  #                            min_fraction_of_examples_in_queue)
  # elif input_type == 'aug':
  #   min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN_AUG *
  #                            min_fraction_of_examples_in_queue)
  # else:
  #   raise ValueError
  min_queue_examples = int(num_examples * min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(read_input.key, normalized_coef,
                                         normalized_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=shuffle, input_type=input_type,
                                         num_threads=num_threads)


def inputs(run_config, batch_size, num_threads=16):
  """
  1D curve added.

  Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # if not eval_data:
  #   # validation
  #   filenames = [os.path.join(data_dir, 'batch-9.bin')]
  #                #'data_batch_%d.bin' % i)
  #                #for i in xrange(1, 6)]
  #   #num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  #   num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_VAL
  # else:
  #   # test
  #   raise Exception()
  #   filenames = [os.path.join(data_dir, 'batch-9.bin')]
  #   num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
  filenames = run_config['record_paths']
  num_examples_per_epoch = run_config['num_examples']

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_data(filename_queue)
  reshaped_image = tf.cast(read_input.image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         width, height)
  # resized_image = tf.image.resize_images(reshaped_image, height, width)

  # Subtract off the mean and divide by the variance of the pixels.
  normalized_image = tf.image.per_image_standardization(resized_image)

  normalized_coef = tf.image.per_image_standardization(read_input.coef)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(read_input.key, normalized_coef,
                                         normalized_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False, num_threads=num_threads)

