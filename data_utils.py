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

"""Utilities for loading datasets for training."""
import functools
from typing import Dict, Any, Tuple

from absl import logging
import jax.numpy as jnp
import ml_collections as collections
import tensorflow as tf

import data as cpdata


def apply_cifar_augmentation(
    config: collections.ConfigDict, ds: tf.data.Dataset,
    shape: Tuple[int, int, int]) -> tf.data.Dataset:
  """Applies data augmentation for CIFAR dataset.

  Args:
    config: training configuration
    ds: dataset to apply augmentation to
    shape: image shape

  Returns:
    Augmented dataset.
  """
  if config.cifar_augmentation == 'standard':
    standard_fn = functools.partial(
        cpdata.augment_flip_crop,
        shape=shape, crop=4, mode='CONSTANT', replace=121)
    ds = ds.map(standard_fn)
  elif config.cifar_augmentation == 'autoaugment':
    autoaugment_fn = functools.partial(
        cpdata.augment_autoaugment, shape=shape, replace=121)
    ds = ds.map(autoaugment_fn)
  elif config.cifar_augmentation == 'standard+cutout':
    standard_fn = functools.partial(
        cpdata.augment_flip_crop,
        shape=shape, crop=4, mode='CONSTANT', replace=121)
    cutout_fn = functools.partial(cpdata.augment_cutout, replace=121, pad=8)
    ds = ds.map(standard_fn)
    ds = ds.map(cutout_fn)
  elif config.cifar_augmentation == 'standard+autoaugment':
    standard_fn = functools.partial(
        cpdata.augment_flip_crop,
        shape=shape, crop=4, mode='CONSTANT', replace=121)
    autoaugment_fn = functools.partial(
        cpdata.augment_autoaugment, shape=shape, replace=121)
    ds = ds.map(standard_fn)
    ds = ds.map(autoaugment_fn)
  elif config.cifar_augmentation == 'standard+autoaugment+cutout':
    standard_fn = functools.partial(
        cpdata.augment_flip_crop,
        shape=shape, crop=4, mode='CONSTANT', replace=121)
    autoaugment_fn = functools.partial(
        cpdata.augment_autoaugment, shape=shape, replace=121)
    cutout_fn = functools.partial(cpdata.augment_cutout, replace=121, pad=8)
    ds = ds.map(standard_fn)
    ds = ds.map(autoaugment_fn)
    ds = ds.map(cutout_fn)
  else:
    raise ValueError('Invalid augmentation for CIFAR10.')
  return ds


def get_data_stats(config: collections.ConfigDict) -> Dict[str, Any]:
  """Get data statistics for selected dataset.

  Retrieves data sizes, shapes and whitening statistics based on the
  dataset selected in config.dataset.

  Args:
    config: training configuration

  Returns:
    Dictionary containing statistics of loaded data split.
  """

  data = {}
  if config.dataset == 'wine_quality':
    data['classes'] = 2
    train_examples = int(5000*0.8) - config.val_examples
    test_examples = 5000 - config.val_examples - train_examples
    data['sizes'] = {
        'train': train_examples,
        'val': config.val_examples,
        'test': test_examples,
    }
    data['shape'] = (1, 1, 11)
    data['means'] = [
        10.532083, 0.04565686, 0.33281144, 0.99399555, 6.850714,
        35.23343, 3.187603, 6.373672, 0.49019712, 138.01242, 0.27974856
    ]
    data['stds'] = [
        1.2350279, 0.022253787, 0.119335935, 0.003012671, 0.85485053,
        17.152323, 0.15184218, 5.0720124, 0.11392499, 42.492615, 0.102494776
    ]
  elif config.dataset == 'mnist':
    data['classes'] = 10
    data['sizes'] = {
        'train': 60000 - config.val_examples,
        'val': config.val_examples,
        'test': 10000,
    }
    data['shape'] = (28, 28, 1)
    data['means'] = [0.5]
    data['stds'] = [0.5]
  elif config.dataset == 'emnist_byclass':
    # For evaluation, we want to keep the number of test examples and validation
    # examples down, because >10-20k test examles slows down evaluation
    # considerably, and we run into OOM problems.
    data['classes'] = 26 * 2
    data['sizes'] = {
        'train': 104000 - config.val_examples,  # = 52 * 2000
        'val': config.val_examples,
        'test': 10400,  # = 52 * 200
    }
    data['shape'] = (28, 28, 1)
    data['means'] = [0.5]
    data['stds'] = [0.5]
  elif config.dataset == 'fashion_mnist':
    data['classes'] = 10
    data['sizes'] = {
        'train': 60000 - config.val_examples,
        'val': config.val_examples,
        'test': 10000,
    }
    data['shape'] = (28, 28, 1)
    data['means'] = [0.5]
    data['stds'] = [0.5]
  elif config.dataset == 'cifar10':
    data['classes'] = 10
    data['sizes'] = {
        'train': 50000 - config.val_examples,
        'val': config.val_examples,
        'test': 10000,
    }
    data['shape'] = (32, 32, 3)
    data['means'] = [0.49137254902, 0.482352941176, 0.446666666667]
    data['stds'] = [0.247058823529, 0.243529411765, 0.261568627451]
  elif config.dataset == 'cifar100':
    data['classes'] = 100
    data['sizes'] = {
        'train': 50000 - config.val_examples,
        'val': config.val_examples,
        'test': 10000,
    }
    data['shape'] = (28, 28, 1)
    data['means'] = [0.491399755166, 0.4821585592989, 0.446530913373]
    data['stds'] = [0.2470322514179, 0.2434851647, 0.2615878392604]
  else:
    raise ValueError('Invalid dataset.')

  data['means'] = jnp.array(data['means'])
  data['stds'] = jnp.array(data['stds'])

  return data


def _check_batch_sizes(config: collections.ConfigDict, data: Dict[str, Any]):
  """Helper to check whether dataset sizes are divisible by batch sizes.

  Args:
    config: training configuration
    data: datasets and sizes
  """
  for key, batch_size in zip([
      'train', 'test', 'val'
  ], [
      config.batch_size, config.test_batch_size, config.test_batch_size,
  ]):
    if data['sizes'][key] % batch_size != 0:
      raise ValueError(
          'Trying to do conformal training with batch size %d '
          'but %s set size %d is not divisible by the batch size '
          '(and drop_remainder is False).' % (
              batch_size, key, data['sizes'][key],
          ))


def _batch_sets(
    config: collections.ConfigDict, data: Dict[str, Any], drop_remainder: bool):
  """Helper to take care of training set shuffling.

  Args:
    config: training configuration
    data: datasets and sizes
    drop_remainder: whether to drop the remaining examples if they
      cannot fill a full batch
  """
  # For some datasets, we need to drop any batch that is smaller than
  # the requested batch size at the end. This is because, for conformal
  # training, the batch size is fixed due to the smooth sorting component used.
  # So, to be fair, we just drop any batch at the end.

  if data['sizes']['train'] % config.batch_size != 0:
    drop_remainder = True
    logging.warning(
        'dropping last batch as %d training examples not divisible '
        'by %d batch size!', data['sizes']['train'], config.batch_size)

  # Unshuffled and clean versions for computing logits in a
  # deterministic way.
  data['train_ordered'] = data['train'].batch(
      config.batch_size, drop_remainder=drop_remainder)
  data['train_clean'] = data['train_clean'].batch(
      config.batch_size, drop_remainder=drop_remainder)

  # We allow to run cross-validation like experiments by repeating the
  # training set X times, shuffling and then taking the first
  # examples. This creates a training set of same size but
  # emulates sampling with up to config.resampling replacements.
  if config.resampling:
    if config.resampling <= 1:
      raise ValueError('Cannot resample training set once or less often.')
    data['train'] = data['train'].repeat(config.resampling)
    data['train'] = data['train'].shuffle(
        config.resampling * data['sizes']['train'], seed=config.seed)
    data['train'] = data['train'].take(data['sizes']['train'])
  else:
    data['train'] = data['train'].shuffle(
        data['sizes']['train'], seed=config.seed)

  data['train'] = data['train'].batch(
      config.batch_size, drop_remainder=drop_remainder)
  if data['val'] is not None:
    data['val'] = data['val'].batch(
        config.test_batch_size, drop_remainder=drop_remainder)
  data['test'] = data['test'].batch(
      config.test_batch_size, drop_remainder=drop_remainder)

  if not drop_remainder:
    _check_batch_sizes(config, data)


def get_data(config: collections.ConfigDict) -> Dict[str, Any]:
  """Get data for training and testing.

  Args:
    config: training configuration

  Returns:
    Dictionary containing training and test datasets, number of classes,
    and mean and std per channel for training dataset.
  """
  def map_mnist_cifar(batch):
    """Mapping for image int to float on MNIST/CIFAR."""
    return {
        'image': tf.cast(batch['image'], tf.float32) / 255.,
        'label': batch['label'],
    }
  def map_emnist_byclass_transpose_and_labels(batch):
    """Helper to map labels for EMNIST/byClass."""
    return {
        'image': tf.cast(
            tf.transpose(batch['image'], perm=[1, 0, 2]), tf.float32) / 255.,
        'label': batch['label'] - 10,
    }
  def filter_emnist_byclass(batch):
    """Helper to filter out digits in EMNIST/byClass."""
    return batch['label'] >= 10
  def map_wine_quality_expand_and_relabel(batch):
    """Helper to expand features to image size for win quality."""
    keys = [
        'alcohol',
        'chlorides',
        'citric acid',
        'density',
        'fixed acidity',
        'free sulfur dioxide',
        'pH',
        'residual sugar',
        'sulphates',
        'total sulfur dioxide',
        'volatile acidity',
    ]
    features = tf.stack(
        [tf.cast(batch['features'][k], tf.float32) for k in keys], axis=0)
    return {
        'image': tf.cast(tf.reshape(features, (1, 1, -1)), tf.float32),
        'label': 1 if batch['quality'] >= 6 else 0,
    }

  data = get_data_stats(config)
  drop_remainder = False
  if config.dataset == 'wine_quality':
    train_examples = data['sizes']['train']
    val_examples = data['sizes']['val']
    data_split = cpdata.create_data_split(
        'wine_quality/white',
        train_examples, val_examples, padding_size=5000)
    data['train'] = data_split['train'].map(map_wine_quality_expand_and_relabel)
    data['val'] = data_split['val']
    if data['val'] is not None:
      data['val'] = data['val'].map(map_wine_quality_expand_and_relabel)
    data['test'] = data_split['test'].map(map_wine_quality_expand_and_relabel)
    data['train_clean'] = data['train']
    # Adapt data split to avoid check on batch size below.
    data_split['sizes'] = data['sizes']
  elif config.dataset == 'emnist_byclass':
    # The validation example number is a fix for type checking:
    # We want data_split['val'] to be None if val_examples=0, otherwise
    # type checks below will fail.
    # So we request 1 validation examples if val_examples > 0 and 0 else.
    train_examples = data['sizes']['train']
    val_examples = data['sizes']['val']
    test_examples = data['sizes']['test']
    data_split = cpdata.load_data_split(
        'emnist/byclass', val_examples=min(config.val_examples, 1))
    # Train and validation set is created from the provided train dataset
    # by filtering, mapping and then taking train_examples + val_examples.
    data['train'] = data_split['train'].filter(filter_emnist_byclass)
    data['train'] = data['train'].map(map_emnist_byclass_transpose_and_labels)
    data['train'] = data['train'].take(train_examples + val_examples)
    data['val'] = data_split['val']
    if data['val'] is not None:
      data['val'] = data['train'].skip(train_examples)
    # Important to take after defining the validation set!
    data['train'] = data['train'].take(train_examples)
    data['test'] = data_split['test'].filter(filter_emnist_byclass)
    data['test'] = data['test'].map(map_emnist_byclass_transpose_and_labels)
    data['test'] = data['test'].take(test_examples)
    data['train_clean'] = data['train']
    # Adapt data split to avoid check on batch size below.
    data_split['sizes'] = data['sizes']
  elif config.dataset in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']:
    data_split = cpdata.load_data_split(
        config.dataset, val_examples=config.val_examples)

    # We need to apply data augmentation before the mapping as the mapping
    # divides by 255 (which was before done in load_batches), but
    # data augmentation operates on proper images, not floats.
    data['train'] = data_split['train']
    if config.dataset.find('cifar') >= 0:
      logging.info('Adding data augmentation for CIFAR.')
      data['train'] = apply_cifar_augmentation(
          config, data['train'], data_split['shape'])
    data['train'] = data['train'].map(map_mnist_cifar)

    # Dataset without data augmentation:
    data['train_clean'] = data_split['train'].map(map_mnist_cifar)
    data['val'] = data_split['val']
    if data['val'] is not None:
      data['val'] = data['val'].map(map_mnist_cifar)
    data['test'] = data_split['test'].map(map_mnist_cifar)
  else:
    raise ValueError('Invalid dataset.')

  data['sizes'] = data_split['sizes']
  data['shape'] = data_split['shape']

  # This takes care of shuffling, batching and resampling with replacement
  # if requested.
  _batch_sets(config, data, drop_remainder)

  return data
