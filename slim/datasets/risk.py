"""Provides data for the MammoAI Risk dataset.

The dataset scripts used to create the dataset can be found at:
../data/*.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = '%s_shards/%s-*'

_SPLITS_TO_SIZES = {'trainneg': 118242, 'trainpos': 1603, 'val': 20262, 'test': 30961}
_NUM_CLASSES = 2

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading MNIST.

  Args:
    split_name: A train/val/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % (split_name, split_name))

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      ## COMPLETE THIS WITH MEDICAL RECORDS FIELDS!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
      'medical/basename': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/class/label': tf.FixedLenFeature(
          [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
      'medical/age': tf.FixedLenFeature([1], tf.float32, default_value=tf.zeros([1], dtype=tf.float32)),
      'medical/exposure': tf.FixedLenFeature([1], tf.float32, default_value=tf.zeros([1], dtype=tf.float32)),
      'medical/current': tf.FixedLenFeature([1], tf.float32, default_value=tf.zeros([1], dtype=tf.float32)),
      'medical/thickness': tf.FixedLenFeature([1], tf.float32, default_value=tf.zeros([1], dtype=tf.float32)),
      'medical/compression': tf.FixedLenFeature([1], tf.float32, default_value=tf.zeros([1], dtype=tf.float32)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
      'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
      'basename': slim.tfexample_decoder.Tensor('medical/basename', shape=[]),
      'age': slim.tfexample_decoder.Tensor('medical/age',shape=[]),
      'exposure': slim.tfexample_decoder.Tensor('medical/exposure',shape=[]),
      'current': slim.tfexample_decoder.Tensor('medical/current',shape=[]),
      'thickness': slim.tfexample_decoder.Tensor('medical/thickness',shape=[]),
      'compression': slim.tfexample_decoder.Tensor('medical/compression',shape=[]),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  _ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'label': 'The label id of the image, integer between 0 and 999',
  }
  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      num_classes=_NUM_CLASSES,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS)
