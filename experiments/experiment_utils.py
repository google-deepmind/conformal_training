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

"""Utilities for experiments."""
from typing import Sequence

import numpy as np


def loss_matrix_singleton_zero(
    off: float, on: float, singleton: int, classes: int) -> Sequence[float]:
  """Loss matrix to discourage overlap with a single class.

  Creates a classes x classes loss matrix where the elements
  k, singleton are set to off for all k != singleton in [0, classes-1].

  Args:
    off: off-diagonal value to set
    on: on-diagonal value to set
    singleton: class to discourage overlap with
    classes: number of classes

  Returns:
    Flattened loss matrix as tuple
  """
  loss_matrix = np.eye(classes) * on
  loss_matrix[:, singleton] = off
  np.fill_diagonal(loss_matrix, on)
  return tuple(loss_matrix.flatten())


def loss_matrix_singleton_one(
    off: float, on: float, singleton: int, classes: int) -> Sequence[float]:
  """Loss matrix to discourage overlap with all other classes.

  Creates a classes x classes loss matrix where the elements
  k, singleton are set to off for all k != singleton in [0, classes-1].

  Args:
    off: off-diagonal value to set
    on: on-diagonal value to set
    singleton: class to discourage overlap with
    classes: number of classes

  Returns:
    Flattened loss matrix as tuple
  """
  loss_matrix = np.eye(classes) * on
  loss_matrix[singleton, :] = off
  loss_matrix[singleton, singleton] = on
  return tuple(loss_matrix.flatten())


def loss_matrix_group_zero(
    off: float, on: float,
    groups: Sequence[int], classes: int) -> Sequence[float]:
  """Discourage confidence sets of group 0 to contain group 1 classes.

  Creates a loss matrix that discourages overlap between two groups of classes.
  We penalize confidence sets of group 0 to contain classes of group 1.

  Args:
    off: off-diagonal value to set
    on: on-diagonal value to set
    groups: group index for each class
    classes: number of classes

  Returns:
    Flattened loss matrix as tuple
  """
  groups = np.array(groups)
  loss_matrix = np.eye(classes) * on
  true_indices = np.where(groups == 0)[0]
  pred_indices = np.where(groups == 1)[0]
  loss_matrix[np.ix_(true_indices, pred_indices)] = off
  np.fill_diagonal(loss_matrix, on)
  return tuple(loss_matrix.flatten())


def loss_matrix_group_one(
    off: float, on: float,
    groups: Sequence[int], classes: int) -> Sequence[float]:
  """Discourage confidence sets of group 1 to contain group 0 classes.

  Opposite of loss_matrix_group_zero.

  Args:
    off: off-diagonal value to set
    on: on-diagonal value to set
    groups: group index for each class
    classes: number of classes

  Returns:
    Flattened loss matrix as tuple
  """
  groups = np.array(groups)
  loss_matrix = np.eye(classes) * on
  true_indices = np.where(groups == 1)[0]
  pred_indices = np.where(groups == 0)[0]
  loss_matrix[np.ix_(true_indices, pred_indices)] = off
  np.fill_diagonal(loss_matrix, on)
  return tuple(loss_matrix.flatten())


def loss_matrix_importance(
    weights: Sequence[float], classes: int) -> Sequence[float]:
  """Loss matrix with different weights on diagonal.

  Creates a diagonal loss matrix with the given weights on the diagonal.

  Args:
    weights: on-diagonal weights
    classes: number of classes

  Returns:
    Flattened loss matrix as tuple
  """
  loss_matrix = np.eye(classes)
  np.fill_diagonal(loss_matrix, np.array(weights))
  return tuple(loss_matrix.flatten())


def loss_matrix_confusion(
    class_a: int, class_b: int, off_a_b: float, off_b_a: float,
    on: float, classes: int) -> Sequence[float]:
  """Loss matrix to penalize confusion between two classes.

  Creates a loss matrix to discourage confusion between classes a and b using
  the off-diagonal weights off_a_b and off_b_a and the on-diagonal weight on.

  Args:
    class_a: first class
    class_b: second class
    off_a_b: penalty of including class_b in confidence sets of class_a
    off_b_a: penalty of including class_a in confidence sets of class_b
    on: on-diagonal value
    classes: number of classes

  Returns:
    Flattened loss matrix as tuple
  """
  loss_matrix = np.eye(classes) * on
  loss_matrix[class_a, class_b] = off_a_b
  loss_matrix[class_b, class_a] = off_b_a
  return tuple(loss_matrix.flatten())


def loss_matrix_confusion_triple(
    class_a: int, class_b: int, class_c: int,
    off: float, on: float, classes: int) -> Sequence[float]:
  """Loss matrix to penalize confusion between three classes.

  Loss_matrix_confusion for three pairs of classes using the same off-diagonal
  weight for all combinations.

  Args:
    class_a: first class
    class_b: second class
    class_c: third class
    off: off-diagonal penalty to use
    on: on-diagonal value
    classes: number of classes

  Returns:
    Flattened loss matrix as tuple
  """
  loss_matrix = np.eye(classes) * on
  # Example: 4, 5, 7, pairs (4, 5), (5, 4), (4, 7), (7, 4), (5, 7), (7, 5)
  loss_matrix[class_a, class_b] = off
  loss_matrix[class_a, class_c] = off
  loss_matrix[class_b, class_a] = off
  loss_matrix[class_b, class_c] = off
  loss_matrix[class_c, class_a] = off
  loss_matrix[class_c, class_b] = off
  return tuple(loss_matrix.flatten())


def loss_matrix_confusion_row(
    selected_class: int, off: float, on: float,
    classes: int) -> Sequence[float]:
  """Loss matrix to penalize confusion for one class with all others.

  Loss_matrix_confusion for a full row of the coverage confusion matrix.
  That is, we penalize the confidence sets of selected_class to include
  any other class.

  Args:
    selected_class: class or row in coverage confusion matrix
    off: off-diagonal weight to apply
    on: on-diagonal value
    classes: number of classes

  Returns:
    Flattened loss matrix as tuple
  """
  loss_matrix = np.eye(classes) * on
  loss_matrix[selected_class, :] = off
  return tuple(loss_matrix.flatten())


def size_weights_group(
    groups: Sequence[int], weights: Sequence[float]) -> Sequence[float]:
  """Helper to set up class size weights.

  Define class weights for multiple groups of classes.

  Args:
    groups: group index per class
    weights: weight for each group

  Returns:
    Size weights as tuple
  """
  groups = np.array(groups)
  weights = np.array(weights)
  unique_groups = np.unique(groups)
  if unique_groups.size != weights.size:
    raise ValueError('Invalid groups or weights.')
  size_weights = np.zeros(groups.shape)
  for group, weight in zip(unique_groups, weights):
    size_weights[groups == group] = weight
  return tuple(size_weights)


def size_weights_selected(
    selected_classes: Sequence[int],
    weight: float, classes: int) -> Sequence[float]:
  """Helper to set up class size weights.

  Obtain size weights where the weight of the selected classes is weight
  and all others are 1.

  Args:
    selected_classes: classes to set the given size weight
    weight: size weight to apply
    classes: number of classes

  Returns:
    Size weights as tuple
  """
  selected_classes = np.array(selected_classes)
  size_weights = np.ones(classes)
  size_weights[selected_classes] = weight
  return tuple(size_weights)
