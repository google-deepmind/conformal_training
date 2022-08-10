# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for data."""
import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import numpy as np
import tensorflow_datasets as tfds

import data as cpdata
DATA_DIR = './data/'


class DataTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(dataset='mnist', train_examples=60000, val_examples=10000),
      dict(dataset='mnist', train_examples=60000, val_examples=0),
  ])
  def test_load_data_split_sizes(self, dataset, train_examples, val_examples):
    data = cpdata.load_data_split(
        dataset, val_examples=val_examples, data_dir=DATA_DIR)

    ds_sizes = data['sizes']
    self.assertLen(data['train'], train_examples - val_examples)
    self.assertLen(data['test'], 10000)
    self.assertEqual(ds_sizes['train'], train_examples - val_examples)
    self.assertEqual(ds_sizes['val'], val_examples)
    self.assertEqual(ds_sizes['test'], 10000)

    if val_examples > 0:
      self.assertLen(data['val'], val_examples)
    else:
      self.assertIsNone(data['val'])

  def test_load_data_split_errors(self):
    with self.assertRaises(ValueError):
      cpdata.load_data_split('mnist', val_examples=-1, data_dir=DATA_DIR)

  @parameterized.parameters([
      dict(batch_size=128),
  ])
  def test_load_batches(self, batch_size):
    val_examples = 59500
    train_examples = 60000 - val_examples
    data = cpdata.load_data_split(
        'mnist', val_examples=val_examples, data_dir=DATA_DIR)
    data['train'] = data['train'].batch(batch_size)

    b = 0
    for b, (inputs, labels) in enumerate(cpdata.load_batches(data['train'])):
      chex.assert_rank([inputs, labels], [4, 1])
      # Batch size might be smaller for the last batch!
      if b == 0:
        chex.assert_shape(inputs, (batch_size, 28, 28, 1))
        chex.assert_shape(labels, (batch_size,))
      # For MNIST, the scaling has to happen manually.
      self.assertGreaterEqual(255, np.max(inputs))
      self.assertGreaterEqual(np.max(inputs), 0)
      self.assertGreaterEqual(9, np.max(labels))
    self.assertEqual(b + 1, np.ceil(train_examples/batch_size))

  # Testing all will cause a timeout, so just testing autoaugment
  # from now on as that's the most complex augmentation.
  @parameterized.parameters([
      dict(augmentation_name='augment_flip_crop', augmentation_args=dict(
          shape=(32, 32, 3), crop=4, mode='CONSTANT', replace=121)),
      dict(augmentation_name='augment_autoaugment',
           augmentation_args=dict(shape=(32, 32, 3), replace=121)),
      dict(augmentation_name='augment_cutout',
           augmentation_args=dict(replace=121, pad=8)),
  ])
  def test_augment(self, augmentation_name, augmentation_args):
    batch_size = 100
    # Not using cpdata.load_data_split to avoid timeouts.
    ds = tfds.load(
        'cifar10', split='train[:1000]', with_info=False, data_dir=DATA_DIR)
    augmentation = getattr(cpdata, augmentation_name, None)
    self.assertIsNotNone(augmentation)
    augmentation = functools.partial(augmentation, **augmentation_args)
    ds = ds.map(augmentation).batch(batch_size)

    for inputs, _ in cpdata.load_batches(ds):
      chex.assert_shape(inputs, (batch_size, 32, 32, 3))
      self.assertGreaterEqual(255, np.max(inputs))
      self.assertGreaterEqual(np.max(inputs), 0)
      break

if __name__ == '__main__':
  absltest.main()
