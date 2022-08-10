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

"""Tests for data utilities."""
from absl.testing import absltest
from absl.testing import parameterized

import chex
import ml_collections as collections

import conformal_training.data as cpdata
import conformal_training.data_utils as cpdatautils
DATA_DIR = './data/'


class DataUtilsTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(cifar_augmentation='standard+autoaugment+cutout'),
  ])
  def test_apply_cifar_augmentation(self, cifar_augmentation):
    batch_size = 100
    data = cpdata.load_data_split(
        'cifar10', val_examples=50000 - batch_size, data_dir=DATA_DIR)
    config = collections.ConfigDict()
    config.cifar_augmentation = cifar_augmentation
    ds = cpdatautils.apply_cifar_augmentation(
        config, data['train'], data['shape'])
    ds = ds.batch(batch_size)

    inputs, _ = next(cpdata.load_batches(ds))
    chex.assert_shape(inputs, (batch_size, 32, 32, 3))


if __name__ == '__main__':
  absltest.main()
