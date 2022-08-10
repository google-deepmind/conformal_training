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

"""Utils for evaluation in Colabs or notebooks."""
from typing import Tuple, Callable, Dict, Any, List

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import sklearn.metrics

import conformal_training.conformal_prediction as cp
import conformal_training.evaluation as cpeval
import conformal_training.open_source_utils as cpstaging


_CalibrateFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], float]
_PredictFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
load_predictions = cpstaging.load_predictions


def get_threshold_fns(
    alpha: float, jit: bool = True) -> Tuple[_CalibrateFn, _PredictFn]:
  """Prediction and calibration function for threshold conformal prediction.

  Args:
    alpha: confidence level
    jit: jit prediction and calibration function

  Returns:
    Calibration and prediction functions
  """
  def calibrate_threshold_fn(logits, labels, rng):  # pylint: disable=unused-argument
    probabilities = jax.nn.softmax(logits, axis=1)
    return cp.calibrate_threshold(
        probabilities, labels, alpha=alpha)
  def predict_threshold_fn(logits, tau, rng):  # pylint: disable=unused-argument
    probabilities = jax.nn.softmax(logits, axis=1)
    return cp.predict_threshold(
        probabilities, tau)
  if jit:
    calibrate_threshold_fn = jax.jit(calibrate_threshold_fn)
    predict_threshold_fn = jax.jit(predict_threshold_fn)
  return calibrate_threshold_fn, predict_threshold_fn


def get_raps_fns(
    alpha: float, k_reg: int, lambda_reg: float,
    jit: bool = True) -> Tuple[_CalibrateFn, _PredictFn]:
  """Prediction and calibration function for RAPS.

  Args:
    alpha: confidence level
    k_reg: k for regularization
    lambda_reg: lambda for regularization
    jit: jit prediction and calibration function

  Returns:
    Calibration and prediction functions
  """
  def calibrate_raps_fn(logits, labels, rng):
    probabilities = jax.nn.softmax(logits, axis=1)
    return cp.calibrate_raps(
        probabilities, labels, alpha=alpha,
        k_reg=k_reg, lambda_reg=lambda_reg, rng=rng)
  def predict_raps_fn(logits, tau, rng):
    probabilities = jax.nn.softmax(logits, axis=1)
    return cp.predict_raps(
        probabilities, tau, k_reg=k_reg, lambda_reg=lambda_reg, rng=rng)
  if jit:
    calibrate_raps_fn = jax.jit(calibrate_raps_fn)
    predict_raps_fn = jax.jit(predict_raps_fn)
  return calibrate_raps_fn, predict_raps_fn


def get_groups(dataset: str, key: str) -> jnp.ndarray:
  """Helper to define groups for evaluation.

  Args:
    dataset: dataset identifier
    key: type of loss to load

  Returns:
    Class groups for given dataset and key
  """
  if dataset == 'wine_quality':
    if key == 'identity':
      groups = jnp.arange(2)
    else:
      raise NotImplementedError
  elif dataset == 'mnist':
    if key == 'identity':
      groups = jnp.arange(10)
    elif key == 'singleton':
      # Hardest class.
      groups = jnp.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], int)
    elif key == 'groups':
      # Odd vs. even.
      groups = jnp.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], int)
    else:
      raise NotImplementedError
  elif dataset == 'emnist_byclass':
    if key == 'identity':
      groups = jnp.arange(52)
    elif key == 'groups':
      groups = jnp.array([
          # Upper case:
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0,
          # Lower case:
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1,
      ], int)
    else:
      raise NotImplementedError
  elif dataset == 'fashion_mnist':
    if key == 'identity':
      groups = jnp.arange(10)
    elif key == 'singleton':
      # Hardest class.
      groups = jnp.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], int)
    else:
      raise NotImplementedError
  elif dataset == 'cifar10':
    if key == 'identity':
      groups = jnp.arange(10)
    elif key == 'singleton':
      # Hardest class.
      groups = jnp.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], int)
    elif key == 'groups':
      # Human-made vs. animals.
      groups = jnp.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0], int)
    else:
      raise NotImplementedError
  elif dataset == 'cifar100':
    if key == 'identity':
      groups = jnp.arange(100)
    elif key == 'groups':
      # Human-made vs. animals or natural scenes,
      # people are considered human-made.
      # Grouping happened on coarse class ids, not fine ones.
      groups = jnp.array([
          1, 1, 0, 1, 1, 0, 1, 1, 0, 0,
          0, 0, 0, 0, 1, 1, 0, 0, 1, 1,
          0, 1, 0, 1, 1, 0, 1, 1, 0, 1,
          1, 1, 1, 1, 1, 0, 1, 0, 1, 0,
          0, 0, 1, 1, 1, 1, 0, 1, 0, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
          1, 0, 1, 1, 1, 1, 1, 1, 0, 0,
          1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
          1, 0, 1, 1, 0, 0, 0, 0, 1, 0,
          0, 1, 1, 1, 0, 1, 1, 1, 0, 1,
      ], int)
    elif key == 'hierarchy':
      # These are essentially the coarse labels of CIFAR100.
      groups = jnp.array([
          4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
          3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
          6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
          0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
          5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
          16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
          10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
          2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
          16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
          18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
      ], int)
    else:
      raise NotImplementedError
  else:
    raise ValueError('No loss matrices defined for dataset %s.' % dataset)

  return groups


def _evaluate_accuracy(
    logits: jnp.ndarray, labels: jnp.ndarray) -> pd.DataFrame:
  """Helper to compute accuracy on single dataset.

  Args:
    logits: predicted logits
    labels: ground truth labels

  Returns:
    Accuracy and list of class-conditional accuracies
  """
  classes = logits.shape[1]
  probabilities = jax.nn.softmax(logits, axis=1)
  accuracy = float(cpeval.compute_accuracy(probabilities, labels))
  accuracies = []
  for k in range(classes):
    accuracies.append(float(cpeval.compute_conditional_accuracy(
        probabilities, labels, labels, k)))
  columns = ['accuracy'] + [f'accuracy_{i}' for i in range(classes)]
  data = np.array([accuracy] + accuracies)
  return pd.DataFrame(np.expand_dims(data, axis=0), columns=columns)


def evaluate_accuracy(model: Dict[str, Any]) -> Dict[str, Any]:
  """Compute accuracy on val/test sets.

  Args:
    model: dictionary containing val/test logits and labels

  Returns:
    Accuracies as dictionary split in validation and test results
  """
  res = {}
  if model['val_labels'].size > 0:
    res['val'] = _evaluate_accuracy(model['val_logits'], model['val_labels'])
  res['test'] = _evaluate_accuracy(model['test_logits'], model['test_labels'])
  return res


def evaluate_coverage(
    data: Dict[str, Any], confidence_sets: jnp.ndarray,
    labels: jnp.ndarray) -> pd.DataFrame:
  """Compute coverage on validation or test data.

  Computes marginal, class- and size-conditional coverages.

  Args:
    data: data information with groups and loss matrix
    confidence_sets: predicted confidence sets
    labels: corresponding ground truth labels

  Returns:
    Results as dictionary
  """
  classes = confidence_sets.shape[1]
  coverage = float(cpeval.compute_coverage(confidence_sets, labels))
  values = {'coverage': coverage}

  # Setup groups for which we compute conditional coverage.
  groups = {
      'class': (labels, classes),
  }
  for key in data['groups']:
    groups[key] = (data['groups'][key][labels],
                   jnp.max(data['groups'][key]) + 1)

  compute_conditional_coverage = jax.jit(cpeval.compute_conditional_coverage)
  for key in groups:
    group_labels, num_groups = groups[key][0], groups[key][1]
    for k in range(num_groups):
      coverage_k = float(compute_conditional_coverage(
          confidence_sets, labels, group_labels, k))
      values['%s_coverage_%d' % (key, k)] = coverage_k

  return pd.DataFrame(
      np.expand_dims(np.array(list(values.values())), axis=0),
      columns=list(values.keys()))


def evaluate_miscoverage(
    data: Dict[str, Any], confidence_sets: jnp.ndarray,
    labels: jnp.ndarray) -> pd.DataFrame:
  """Compute mis-coverage.

  Args:
    data: data information with groups and loss matrix
    confidence_sets: predicted confidence sets
    labels: corresponding ground truth labels

  Returns:
    Results as dictionary
  """
  groups = {}
  values = {}
  for key in data['groups']:
    groups[key] = (data['groups'][key],
                   jnp.max(data['groups'][key]) + 1)

  compute_conditional_miscoverage = jax.jit(
      cpeval.compute_conditional_miscoverage)
  for key in groups:
    group_indices, num_groups = groups[key][0], groups[key][1]
    group_labels = group_indices[labels]

    # For each example, we need to pick the right labels NOT to be included
    # in the confidence sets:
    one_hot_labels = (jnp.expand_dims(
        group_labels, axis=1) != jnp.expand_dims(group_indices, axis=0))
    one_hot_labels = one_hot_labels.astype(int)
    miscoverage = float(compute_conditional_miscoverage(
        confidence_sets, one_hot_labels,
        jnp.zeros(confidence_sets.shape[0]), 0))
    values['%s_miscoverage' % key] = miscoverage
    values['%s_miscoverage_n' % key] = confidence_sets.shape[0]

    for k in range(num_groups):
      miscoverage_k = float(compute_conditional_miscoverage(
          confidence_sets, one_hot_labels, group_labels, k))
      values['%s_miscoverage_%d' % (key, k)] = miscoverage_k
      values['%s_miscoverage_%d_n' % (key, k)] = jnp.sum(group_labels == k)

  return pd.DataFrame(
      np.expand_dims(np.array(list(values.values())), axis=0),
      columns=list(values.keys()))


def evaluate_size(
    data: Dict[str, Any], confidence_sets: jnp.ndarray,
    labels: jnp.ndarray) -> pd.DataFrame:
  """Compute size on validation or test data.

  Args:
    data: data information with groups and loss matrix
    confidence_sets: predicted confidence sets
    labels: corresponding ground truth labels

  Returns:
    Results as dictionary
  """
  classes = confidence_sets.shape[1]
  size, count = cpeval.compute_size(confidence_sets)
  size = float(size)

  values = {
      'size': size,
      'count': count,
  }

  # Setup groups for which we compute conditional sizes.
  groups = {
      'class': (labels, classes),
  }
  for key in data['groups']:
    groups[key] = (data['groups'][key][labels],
                   jnp.max(data['groups'][key]) + 1)

  compute_conditional_size = jax.jit(cpeval.compute_conditional_size)
  for key in groups:
    group_labels, num_groups = groups[key][0], groups[key][1]
    for k in range(num_groups):
      size_k, count_k = compute_conditional_size(
          confidence_sets, group_labels, k)
      values['%s_fraction_%d' % (key, k)] = float(count_k)/count
      values['%s_size_%d' % (key, k)] = size_k

  # Counts per confidence set size.
  confidence_set_sizes = jnp.sum(confidence_sets, axis=1)
  for k in range(classes):
    _, count_k = compute_conditional_size(
        confidence_sets, confidence_set_sizes, k)
    values['size_%d' % k] = int(count_k)/float(confidence_sets.shape[0])

  # Additionally compute cumulative size distribution.
  for k in range(classes):
    values['cumulative_size_%d' % k] = values['size_%d' % k]
    if k > 0:
      values['cumulative_size_%d' % k] += values['cumulative_size_%d' % (k - 1)]

  return pd.DataFrame(
      np.expand_dims(np.array(list(values.values())), axis=0),
      columns=list(values.keys()))


def evaluate_confusion(
    logits: jnp.ndarray, confidence_sets: jnp.ndarray,
    labels: jnp.ndarray) -> pd.DataFrame:
  """Evaluate confusion of confidence sets.

  Args:
    logits: predicted logits for top-1 prediction
    confidence_sets: predicted confidence sets
    labels: ground truth labels

  Returns:
    Confusion matrix
  """
  classes = confidence_sets.shape[1]
  predictions = jnp.argmax(logits, axis=1)
  # Regular classification confusion.
  classification_confusion = sklearn.metrics.confusion_matrix(
      labels, predictions)

  # Confusion in a coverage sense.
  coverage_confusion = np.zeros((classes, classes))
  for k in range(classes):
    coverage_confusion[k] = jnp.sum(confidence_sets[labels == k], axis=0)

  # Note that we normalize the confusion matrices as the count is available
  # separately.
  classification_confusion = classification_confusion / logits.shape[0]
  coverage_confusion = coverage_confusion / logits.shape[0]

  values = np.expand_dims(np.concatenate((
      classification_confusion.flatten(),
      coverage_confusion.flatten()
  )), axis=0)
  columns = []
  # Lint does not like double for loops, even in this simple case:
  for i in range(classes):
    for j in range(classes):
      columns.append('classification_confusion_%d_%d' % (i, j))
  for i in range(classes):
    for j in range(classes):
      columns.append('coverage_confusion_%d_%d' % (i, j))
  return pd.DataFrame(values, columns=columns)


def evaluate_metrics(
    data: Dict[str, Any], logits: jnp.ndarray,
    confidence_sets: jnp.ndarray, labels: jnp.ndarray) -> List[pd.DataFrame]:
  """Evaluate metrics on validation or test set.

  Args:
    data: data information with groups and loss matrix
    logits: predicted logits
    confidence_sets: predicted confidence sets
    labels: ground truth labels

  Returns:
    List of Panda dataframes containing evaluation metrics
  """
  accuracy = _evaluate_accuracy(logits, labels)
  coverage = evaluate_coverage(data, confidence_sets, labels)
  miscoverage = evaluate_miscoverage(data, confidence_sets, labels)
  size = evaluate_size(data, confidence_sets, labels)
  results = [accuracy, coverage, miscoverage, size]
  confusion = evaluate_confusion(logits, confidence_sets, labels)
  results.append(confusion)
  return results


def evaluate_conformal_prediction(
    model: Dict[str, Any], calibrate_fn: _CalibrateFn, predict_fn: _PredictFn,
    trials: int, rng: jnp.ndarray) -> Dict[str, Any]:
  """Evaluate conformal prediction using a calibration and prediction method.

  Applies calibration and prediction on trials random splits into validation
  and test sets. Returns standard deviation and average accuracy and
  coverage metrics.

  Calibration and prediction functions need to expect a rng key
  as additional argument to allow randomization if possible.

  Args:
    model: dictionary containing val/test logits and labels
    calibrate_fn: callable to use for calibration
    predict_fn: callable to use for prediction
    trials: number of trials
    rng: random key

  Returns:
    Dictionary of results containing average and standard deviation
    of metrics
  """
  keys = model.keys()
  if 'val_logits' not in keys or 'val_labels' not in keys:
    raise ValueError('val_logits or val_labels not present.')
  if 'test_logits' not in keys or 'test_labels' not in keys:
    raise ValueError('test_logits or test_labels not present.')

  rngs = jax.random.split(rng, 3*trials)
  val_examples = model['val_labels'].shape[0]
  test_examples = model['test_labels'].shape[0]
  num_examples = val_examples + test_examples

  logits = jnp.concatenate(
      (model['val_logits'], model['test_logits']), axis=0)
  # Casting explicitly to int as some calibration functions may involve
  # indexing which raises a hard ot understand error if labels are not integers.
  labels = jnp.concatenate(
      (model['val_labels'], model['test_labels']), axis=0).astype(int)

  val_results = pd.DataFrame()
  test_results = pd.DataFrame()
  for t in range(trials):
    perm_rng = rngs[3*t + 0]
    val_rng = rngs[3*t + 1]
    test_rng = rngs[3*t + 2]

    perm = jax.random.permutation(perm_rng, jnp.arange(num_examples))
    val_logits_t = logits[perm[:val_examples]]
    val_labels_t = labels[perm[:val_examples]]
    test_logits_t = logits[perm[val_examples:]]
    test_labels_t = labels[perm[val_examples:]]

    tau = calibrate_fn(val_logits_t, val_labels_t, val_rng)
    tau_t = np.array([tau]).reshape(1, -1)  # For handling multiple taus.
    columns = ['tau' if i == 0 else 'tau_%d' % i for i in range(tau_t.shape[1])]
    tau_t = pd.DataFrame(tau_t, columns=columns)

    test_confidence_sets_t = predict_fn(test_logits_t, tau, test_rng)
    test_results_t = evaluate_metrics(
        model['data'], test_logits_t, test_confidence_sets_t, test_labels_t)
    test_results_t = pd.concat([tau_t] + test_results_t, axis=1)

    test_results = pd.concat((test_results, test_results_t), axis=0)
    print(f'\t trial {t}: {tau}', flush=True)

  results = {
      'mean': {'val': val_results.mean(0), 'test': test_results.mean(0)},
      'std': {'val': val_results.std(0), 'test': test_results.std(0)},
  }
  print('\t reduced', flush=True)
  return results
