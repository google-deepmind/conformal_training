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

"""Evaluation metrics for conformal prediction."""
from typing import Tuple

import jax
import jax.numpy as jnp


def _check_labels(probabilities: jnp.ndarray, labels: jnp.ndarray):
  """Helper to check shapes or probabilities/sets and labels.

  Checks shapes of probabilities of confidence sets and labels for
  evaluation.

  Args:
    probabilities: probabilities or confidence sets
    labels: corresponding ground truth labels

  Raises:
    ValueError if shapes do not match.
  """
  if probabilities.ndim != 2:
    raise ValueError('Expecting probabilities/confidence sets of '
                     'shape n_examples x n_classes.')
  if labels.ndim != 1:
    raise ValueError('Expecting labels of shape n_examples.')
  if probabilities.shape[1] == 0:
    raise ValueError('Expecting at least one class.')
  if probabilities.shape[0] != labels.shape[0]:
    raise ValueError('Number of probabilities/confidence sets does '
                     'not match number of labels.')
  if not jnp.issubdtype(labels.dtype, jnp.integer):
    raise ValueError('Expecting labels to be integers.')
  if jnp.max(labels) >= probabilities.shape[1]:
    raise ValueError(
        'labels contains more classes than probabilities/confidence sets.')


def _check_one_hot_labels(
    probabilities: jnp.ndarray, one_hot_labels: jnp.ndarray):
  """Helper to check shapes of probabilities/sets and one-hot labels.

  Args:
    probabilities: probabilities or confidence sets
    one_hot_labels: corresponding ground truth labels in one-hot format

  Raises:
    ValueError if shapes do not match.
  """
  if probabilities.ndim != 2:
    raise ValueError('Expecting probabilities/confidence sets of '
                     'shape n_examples x n_classes.')
  if one_hot_labels.ndim != 2:
    raise ValueError('Expecting labels in one-hot format of '
                     'shape n_examples x n_classes.')
  if probabilities.shape[1] == 0:
    raise ValueError('Expecting at least one class.')
  if probabilities.shape[0] != one_hot_labels.shape[0]:
    raise ValueError('Number of probabilities/confidence sets does '
                     'not match number of labels.')
  if probabilities.shape[1] != one_hot_labels.shape[1]:
    raise ValueError('Number of classes in probabilities/confidence '
                     'sets and one-hot labels do not match.')


def _check_conditional_labels(
    probabilities: jnp.ndarray,
    conditional_labels: jnp.ndarray):
  """Helper to check conditional_labels for metric computation.

  Args:
    probabilities: probabilities or confidence sets
    conditional_labels: labels tp condition on for all examples

  Raises:
    ValueError if shapes do not match
  """

  if conditional_labels.ndim != 1:
    raise ValueError('Expecting conditional_labels of shape n_examples.')
  if conditional_labels.shape[0] != probabilities.shape[0]:
    raise ValueError('Number of probabilities/confidence sets does '
                     'not match number of conditional labels.')
  if not jnp.issubdtype(conditional_labels.dtype, jnp.integer):
    raise ValueError('Expecting conditional labels to be integers.')


def compute_conditional_accuracy(
    probabilities: jnp.ndarray, labels: jnp.ndarray,
    conditional_labels: jnp.ndarray, conditional_label: int) -> float:
  """Computes conditional accuracy given softmax probabilities and labels.

  Conditional accuracy is defined as the accuracy on a subset of the examples
  as selected using the conditional label(s). For example, this allows
  to compute accuracy conditioned on class labels.

  Args:
    probabilities: predicted probabilities on test set
    labels: ground truth labels on test set
    conditional_labels: conditional labels to compute accuracy on
    conditional_label: selected conditional label to compute accuracy on

  Returns:
    Accuracy
  """
  selected = (conditional_labels == conditional_label)
  num_examples = jnp.sum(selected)
  predictions = jnp.argmax(probabilities, axis=1)
  error = selected * (predictions != labels)
  error = jnp.where(num_examples == 0, 1, jnp.sum(error)/num_examples)
  return 1 - error


def compute_conditional_accuracy_with_checks(
    probabilities: jnp.ndarray, labels: jnp.ndarray,
    conditional_labels: jnp.ndarray, conditional_label: int) -> float:
  """compute_conditional_accuracy with extra argument checks."""
  _check_labels(probabilities, labels)
  _check_conditional_labels(probabilities, conditional_labels)
  return compute_conditional_accuracy(
      probabilities, labels, conditional_labels, conditional_label)


def compute_accuracy(probabilities: jnp.ndarray, labels: jnp.ndarray) -> float:
  """Compute unconditional accuracy using compute_conditional_accuracy."""
  return compute_conditional_accuracy(
      probabilities, labels, jnp.zeros(labels.shape, int), 0)


def compute_accuracy_with_checks(
    probabilities: jnp.ndarray, labels: jnp.ndarray) -> float:
  """compute_accuracy with additional argument checks raising ValuzeError."""
  return compute_conditional_accuracy_with_checks(
      probabilities, labels, jnp.zeros(labels.shape, int), 0)


def compute_conditional_multi_coverage(
    confidence_sets: jnp.ndarray, one_hot_labels: jnp.ndarray,
    conditional_labels: jnp.ndarray, conditional_label: int) -> float:
  """Compute coverage of confidence sets, potentially for multiple labels.

  The given labels are assumed to be one-hot labels and the implementation
  supports checking coverage of multiple classes, i.e., whether one of
  the given ground truth labels is in the confidence set.

  Args:
    confidence_sets: confidence sets on test set as 0-1 array
    one_hot_labels: ground truth labels on test set in one-hot format
    conditional_labels: conditional labels to compute coverage on a subset
    conditional_label: selected conditional to compute coverage for

  Returns:
    Coverage.
  """
  selected = (conditional_labels == conditional_label)
  num_examples = jnp.sum(selected)
  coverage = selected * jnp.clip(
      jnp.sum(confidence_sets * one_hot_labels, axis=1), 0, 1)
  coverage = jnp.where(num_examples == 0, 1, jnp.sum(coverage)/num_examples)
  return coverage


def compute_conditional_multi_coverage_with_checks(
    confidence_sets: jnp.ndarray, one_hot_labels: jnp.ndarray,
    conditional_labels: jnp.ndarray, conditional_label: int) -> float:
  """compute_conditional_multi_coverage with additional argument checks."""
  _check_one_hot_labels(confidence_sets, one_hot_labels)
  _check_conditional_labels(confidence_sets, conditional_labels)
  return compute_conditional_multi_coverage(
      confidence_sets, one_hot_labels, conditional_labels, conditional_label)


def compute_coverage(
    confidence_sets: jnp.ndarray, labels: jnp.ndarray) -> float:
  """Compute unconditional coverage using compute_conditional_multi_coverage.

  Args:
    confidence_sets: confidence sets on test set as 0-1 array
    labels: ground truth labels on test set (not in one-hot format)

  Returns:
    Coverage.
  """
  one_hot_labels = jax.nn.one_hot(labels, confidence_sets.shape[1])
  return compute_conditional_multi_coverage(
      confidence_sets, one_hot_labels, jnp.zeros(labels.shape, int), 0)


def compute_coverage_with_checks(
    confidence_sets: jnp.ndarray, labels: jnp.ndarray) -> float:
  """compute_coverage with additional argument checks raising ValueError."""
  return compute_conditional_coverage_with_checks(
      confidence_sets, labels, jnp.zeros(labels.shape, int), 0)


def compute_conditional_coverage(
    confidence_sets: jnp.ndarray, labels: jnp.ndarray,
    conditional_labels: jnp.ndarray, conditional_label: int) -> float:
  """Compute conditional coverage using compute_conditional_multi_coverage.

  Args:
    confidence_sets: confidence sets on test set as 0-1 array
    labels: ground truth labels on test set (not in one-hot format)
    conditional_labels: conditional labels to compute coverage on a subset
    conditional_label: selected conditional to compute coverage for

  Returns:
    Conditional coverage.
  """
  one_hot_labels = jax.nn.one_hot(labels, confidence_sets.shape[1])
  return compute_conditional_multi_coverage(
      confidence_sets, one_hot_labels, conditional_labels, conditional_label)


def compute_conditional_coverage_with_checks(
    confidence_sets: jnp.ndarray, labels: jnp.ndarray,
    conditional_labels: jnp.ndarray, conditional_label: int) -> float:
  """compute_conditional_coverage with additional argument checks raising."""
  _check_labels(confidence_sets, labels)
  _check_conditional_labels(confidence_sets, conditional_labels)
  return compute_conditional_coverage(
      confidence_sets, labels, conditional_labels, conditional_label)


def compute_miscoverage(
    confidence_sets: jnp.ndarray, one_hot_labels: jnp.ndarray) -> float:
  """Compute mis-coverage for given one-hot labels.

  Mis-coverage is the coverage for multiple labels as given
  in one_hot_labels that should not be included in the sets.

  Args:
    confidence_sets: confidence sets on test set as 0-1 array
    one_hot_labels: ground truth labels on test set in one-hot format

  Returns:
    Mis-coverage.
  """
  return compute_conditional_multi_coverage(
      confidence_sets, one_hot_labels,
      jnp.zeros(confidence_sets.shape[0], int), 0)


def compute_miscoverage_with_checks(
    confidence_sets: jnp.ndarray, one_hot_labels: jnp.ndarray) -> float:
  """compute_miscoverage with additional argument checks."""
  _check_one_hot_labels(confidence_sets, one_hot_labels)
  return compute_miscoverage(confidence_sets, one_hot_labels)


def compute_conditional_miscoverage(
    confidence_sets: jnp.ndarray, one_hot_labels: jnp.ndarray,
    conditional_labels: jnp.ndarray, conditional_label: int) -> float:
  """Compute conditional mis-coverage for given one-hot labels.

  Args:
    confidence_sets: confidence sets on test set as 0-1 array
    one_hot_labels: ground truth labels on test set in one-hot format
    conditional_labels: conditional labels to compute coverage on a subset
    conditional_label: selected conditional to compute coverage for

  Returns:
    Mis-coverage.
  """
  return compute_conditional_multi_coverage(
      confidence_sets, one_hot_labels,
      conditional_labels, conditional_label)


def compute_conditional_miscoverage_with_checks(
    confidence_sets: jnp.ndarray, one_hot_labels: jnp.ndarray,
    conditional_labels: jnp.ndarray, conditional_label: int) -> float:
  """compute_conditional_miscoverage with additional argument checks."""
  _check_one_hot_labels(confidence_sets, one_hot_labels)
  _check_conditional_labels(confidence_sets, conditional_labels)
  return compute_conditional_miscoverage(
      confidence_sets, one_hot_labels, conditional_labels, conditional_label)


def _check_confidence_sets(confidence_sets: jnp.ndarray):
  """Helper to check shape of confidence sets.

  Args:
    confidence_sets: predicted confidence sets

  Raises:
    ValueError if shape is incorrect.
  """
  if confidence_sets.ndim != 2:
    raise ValueError(
        'Expecting confidence_sets of shape n_examples x n_classes.')
  if confidence_sets.shape[1] == 0:
    raise ValueError('Expecting at least one class.')


def compute_conditional_size(
    confidence_sets: jnp.ndarray,
    conditional_labels: jnp.ndarray,
    conditional_label: int) -> Tuple[float, int]:
  """Compute confidence set size.

  Args:
    confidence_sets: confidence sets on test set
    conditional_labels: conditional labels to compute size on
    conditional_label: selected conditional to compute size for

  Returns:
    Average size.
  """
  selected = (conditional_labels == conditional_label)
  num_examples = jnp.sum(selected)
  size = selected * jnp.sum(confidence_sets, axis=1)
  size = jnp.where(num_examples == 0, 0, jnp.sum(size)/num_examples)
  return size, num_examples


def compute_conditional_size_with_checks(
    confidence_sets: jnp.ndarray,
    conditional_labels: jnp.ndarray,
    conditional_label: int) -> Tuple[float, int]:
  """compute_conditional_size with additional argument checks."""
  _check_confidence_sets(confidence_sets)
  _check_conditional_labels(confidence_sets, conditional_labels)
  return compute_conditional_size(
      confidence_sets, conditional_labels, conditional_label)


def compute_size(confidence_sets: jnp.ndarray) -> Tuple[float, int]:
  """Compute unconditional coverage using compute_conditional_coverage."""
  return compute_conditional_size(
      confidence_sets, jnp.zeros(confidence_sets.shape[0], int), 0)


def compute_size_with_checks(confidence_sets: jnp.ndarray) -> Tuple[float, int]:
  """compute_size with additional argument checks raising ValueError."""
  return compute_conditional_size_with_checks(
      confidence_sets, jnp.zeros(confidence_sets.shape[0], int), 0)
