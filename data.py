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

"""Datasets and data augmentation."""
from typing import Tuple, Dict, Iterator, Any, Optional

import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

import auto_augment as augment


def load_data_split(
    dataset: str = 'mnist',
    val_examples: int = 10000,
    data_dir: Optional[str] = './data',
) -> Dict[str, Any]:
  """Load 3-fold data split (train, val and test).

  Get a 3-split of a dataset for conformal prediction.
  We always preserve the original test set for comparable results,
  but use a part of the training set as validation set.

  This is used for datasets that come with both a train and a test split.
  For datasets with only a train split, use create_data_split instead.

  Args:
    dataset: dataset to load
    val_examples: number of validation examples to use
      (will be the last val_examples examples from training set)
    data_dir: data directory to load datasets in

  Returns:
    Three datasets corresponding to training, validation, test datasets,
    and a tuple of the corresponding dataset info.
  """
  if val_examples < 0:
    raise ValueError('Cannot load a negative number of validation examples.')

  if val_examples > 0:
    train_ds, train_info = tfds.load(
        dataset, split=f'train[:-{val_examples}]',
        data_dir=data_dir, with_info=True)
    val_ds = tfds.load(
        dataset, split=f'train[-{val_examples}:]', data_dir=data_dir)
  else:
    train_ds, train_info = tfds.load(
        dataset, split='train', data_dir=data_dir, with_info=True)
    val_ds = None
  test_ds, test_info = tfds.load(
      dataset, split='test', data_dir=data_dir, with_info=True)

  shape = tuple(train_info.features['image'].shape)
  sizes = {
      'train': train_info.splits['train'].num_examples - val_examples,
      'val': val_examples,
      'test': test_info.splits['test'].num_examples,
  }

  return {
      'train': train_ds,
      'val': val_ds,
      'test': test_ds,
      'sizes': sizes,
      'shape': shape,
  }


def create_data_split(
    dataset: str, train_examples: int, val_examples: int,
    padding_size: Optional[int] = None) -> Dict[str, Any]:
  """Create a 3-fold data split for a dataset with only a train split.

  Also see load_data_split. This function has the same functionality but for
  datasets which do not come with a train/test split by default.

  Args:
    dataset: dataset to load
    train_examples: number of training examples to use
    val_examples: number of validation examples to use
    padding_size: dataset size with padding, allows to pad the dataset by repeat
      the first few elements, can at most double the size

  Returns:
    Three datasets corresponding to training, validation, test datasets,
    and a tuple of the corresponding dataset info.
  """
  if train_examples <= 0:
    raise ValueError(
        'Cannot load a negative or zero number of training examples.')
  if val_examples < 0:
    raise ValueError('Cannot load a negative number of validation examples.')

  ds, info = tfds.load(dataset, split='train', with_info=True)
  if padding_size is not None:
    ds = ds.repeat(2).take(padding_size)

  if val_examples > 0:
    val_ds = ds.skip(train_examples).take(val_examples)
  else:
    val_ds = None
  train_ds = ds.take(train_examples)
  test_ds = ds.skip(train_examples + val_examples)

  if 'features' in info.features.keys():
    shape = tuple(info.features['features'].shape)
  elif 'image' in info.features.keys():
    shape = tuple(info.features['image'].shape)
  else:
    raise ValueError('Could not determine feature/image shape.')

  sizes = {
      'train': train_examples,
      'val': val_examples,
      'test': info.splits['train'].num_examples - val_examples - train_examples,
  }

  return {
      'train': train_ds,
      'val': val_ds,
      'test': test_ds,
      'sizes': sizes,
      'shape': shape,
  }


def load_batches(
    dataset: tf.data.Dataset) -> Iterator[Tuple[jnp.array, jnp.array]]:
  """Generator for iterating over batches.

  Yields one batch of images and labels. Assumes a dataset on which
  .batch was called to obtain proper batches.

  Args:
    dataset: the dataset to load batches from

  Yields:
    Two arrays corresponding to one batch of inputs and labels.
  """
  for batch in tfds.as_numpy(dataset):
    inputs = jnp.asarray(batch['image'])
    labels = jnp.asarray(batch['label']).astype(int)
    yield inputs, labels


def _augment_flip_crop(
    image: tf.Tensor, shape: Tuple[int, int, int],
    crop: int, mode: str, replace: int) -> tf.Tensor:
  """Apply random flip and crop augmentation.

  Args:
    image: input image
    shape: image shape needed for cropping
    crop: maximum cropping on each side
    mode: mode used for padding before cropping, see tf.pad
    replace: value to use for filling the cut out patch

  Returns:
    Augmented image.
  """
  image = tf.image.random_flip_left_right(image)
  image = tf.pad(
      image, paddings=[[crop, crop], [crop, crop], [0, 0]], mode=mode,
      constant_values=replace)
  return tf.image.random_crop(image, shape)


def _augment_autoaugment(
    image: tf.Tensor, shape: Tuple[int, int, int], replace: int) -> tf.Tensor:
  """Applies an AutoAugment policy to the input image.

  Args:
    image: input image
    shape: image shape
    replace: value to use for filling empty regions

  Returns:
    Augmented image
  """

  return augment.distort_image_with_autoaugment(
      image, augmentation_name='cifar10',
      cutout_const=replace, translate_const=shape[1])


def augment_flip_crop(
    batch: Dict[str, Any], shape: Tuple[int, int, int],
    crop: int, mode: str, replace: int) -> Dict[str, Any]:
  """CIFAR10 standard data augmentation of clips and crops.

  Args:
    batch: dictionary containing single image and label
    shape: image shape needed for cropping
    crop: maximum cropping on each side
    mode: mode used for padding before cropping, see tf.pad
    replace: value to use for filling the cut out patch

  Returns:
    Dictionary with augmented image and unchanged label
  """
  return {
      'image': _augment_flip_crop(
          batch['image'], shape=shape, crop=crop, mode=mode, replace=replace),
      'label': batch['label']
  }


def augment_autoaugment(
    batch: Dict[str, Any], shape: Tuple[int, int, int],
    replace: int) -> Dict[str, Any]:
  """CIFAR10 AutoAugment data augmentation.

  Args:
    batch: dictionary containing single image and label
    shape: image shape
    replace: value to use for filling the cut out patch

  Returns:
    Dictionary with augmented image and unchanged label
  """
  return {
      'image': _augment_autoaugment(
          batch['image'], shape=shape, replace=replace),
      'label': batch['label']
  }


def augment_cutout(
    batch: Dict[str, Any], replace: int, pad: int) -> Dict[str, Any]:
  """CIFAR10 augmentation with flip/crop, AutoAugment and Cutout.

  Args:
    batch: dictionary containing single image and label
    replace: value to use for filling the cut out patch
    pad: cutout size is 2*pad

  Returns:
    Dictionary with augmented image and unchanged label
  """
  return {
      'image': augment.cutout(batch['image'], pad_size=pad, replace=replace),
      'label': batch['label']
  }
