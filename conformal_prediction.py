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

"""Implementation of recent conformal prediction approaches.

Implements conformal prediction from [1,2,3]:

[1] Yaniv Romano, Matteo Sesia, Emmanuel J. Candes.
Classification withvalid and adaptive coverage.
NeurIPS, 2020.
[2] Anastasios N. Angelopoulos, Stephen Bates, Michael Jordan, Jitendra Malik.
Uncertainty sets for image classifiers using conformal prediction.
ICLR, 2021
[3] Mauricio Sadinle, Jing Lei, and Larry A. Wasserman.
Least ambiguous set-valued classifiers with bounded error levels.
ArXiv, 2016.
"""
from typing import Optional, Callable, Any

import jax
import jax.numpy as jnp


_QuantileFn = Callable[[Any, float], float]
_CalibrateFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Any]
_PredictFn = Callable[[jnp.ndarray, Any, jnp.ndarray], jnp.ndarray]
_SelectFn = Callable[[jnp.ndarray, jnp.ndarray], float]


def _check_conformal_quantile(array: jnp.ndarray, q: float):
  """Helper to check quantile arguments.

  Args:
    array: input array to compute quantile of
    q: quantile to compute

  Raises:
      ValueError: if shape or q invalid.
  """
  if array.size == 0:
    raise ValueError('Expecting non-empty array.')
  if array.ndim != 1:
    raise ValueError('Expecting array of shape n.')
  if q < 0 or q > 1:
    raise ValueError('Expecting q in [0,1].')


def conformal_quantile(array: jnp.ndarray, q: float) -> float:
  """Corrected quantile for conformal prediction.

  Wrapper for np.quantile, but instead of obtaining the q-quantile,
  it computes the (1 + 1/array.shape[0]) * q quantile. For conformal
  prediction, this is needed to obtain the guarantees for future test
  examples, see [1] Appendix Lemma for details.

  [1] Yaniv Romano, Evan Petterson, Emannuel J. Candes.
  Conformalized quantile regression. NeurIPS, 2019.

  Args:
    array: input array to compute quantile of
    q: quantile to compute

  Returns:
    (1 + 1/array.shape[0]) * q quantile of array.
  """
  # Using midpoint here to be comparable to the smooth implementation
  # in smooth_conformal_prediction which uses smooth sort to compute quantiles.
  return jnp.quantile(
      array, (1 + 1./array.shape[0]) * q, method='midpoint')


def conformal_quantile_with_checks(array: jnp.ndarray, q: float) -> float:
  """conformal_quantile with extra argument checks raising ValueError."""
  _check_conformal_quantile(array, q)
  return conformal_quantile(array, q)


def _check_predict(probabilities: jnp.ndarray):
  """Helper to check probabilities for prediction.

  Args:
    probabilities: predicted probabilities on test set

  Raises:
    ValueError if shape is incorrect.
  """
  if probabilities.ndim != 2:
    raise ValueError('Expecting probabilities of shape n_examples x n_classes.')
  if probabilities.size == 0:
    raise ValueError('probabilities is empty.')


def _check_calibrate(
    probabilities: jnp.ndarray,
    labels: jnp.ndarray,
    alpha: Optional[float] = None):
  """Helper to check shape of probabilities, labels and alpha for calibration.

  Args:
    probabilities: predicted probabilities on a validation set
    labels: ground truth labels on validation set
    alpha: confidence level

  Raises:
   ValueError if shapes do not match.
  """
  if probabilities.ndim != 2:
    raise ValueError('Expecting probabilities of shape n_examples x n_classes.')
  if labels.ndim != 1:
    raise ValueError('Expecting labels of shape n_examples.')
  if not jnp.issubdtype(labels.dtype, jnp.integer):
    raise ValueError('Expecting labels to be integers.')
  if jnp.max(labels) >= probabilities.shape[1]:
    raise ValueError('More labels than predicted in probabilities.')
  if probabilities.size == 0:
    raise ValueError('probabilities is empty.')
  if probabilities.shape[0] != labels.shape[0]:
    raise ValueError(
        'Number of predicted probabilities does not match number of labels.')
  if alpha is not None:
    if alpha < 0 or alpha > 1:
      raise ValueError('Expecting alpha to be in [0, 1].')


def calibrate_threshold(
    probabilities: jnp.ndarray,
    labels: jnp.ndarray,
    alpha: float = 0.1,
    quantile_fn: _QuantileFn = conformal_quantile) -> float:
  """Probability/logit thresholding baseline calibration procedure.

  Finds a threshold based on input probabilities or logits. Confidence sets
  are defined as all classes above the threshold.

  Args:
    probabilities: predicted probabilities on validation set
    labels: ground truth labels on validation set
    alpha: confidence level
    quantile_fn: function to compute conformal quantile

  Returns:
    Threshold used to construct confidence sets
  """
  conformity_scores = probabilities[
      jnp.arange(probabilities.shape[0]), labels.astype(int)]
  return quantile_fn(conformity_scores, alpha)


def calibrate_threshold_with_checks(
    probabilities: jnp.ndarray,
    labels: jnp.ndarray,
    alpha: float = 0.1,
    quantile_fn: _QuantileFn = conformal_quantile_with_checks) -> float:
  """calibrate_threshold with extra argument checks raising ValueError."""
  _check_calibrate(probabilities, labels, alpha)
  return calibrate_threshold(probabilities, labels, alpha, quantile_fn)


def predict_threshold(probabilities: jnp.ndarray, tau: float) -> jnp.ndarray:
  """Probability/logit threshold baseline.

  Predicts all classes with probabilities/logits above given threshold
  as confidence sets.

  Args:
    probabilities: predicted probabilities on test set
    tau: threshold for probabilities or logits

  Returns:
    Confidence sets as 0-1array of same size as probabilities.
  """
  confidence_sets = (probabilities >= tau)
  return confidence_sets.astype(int)


def predict_threshold_with_checks(
    probabilities: jnp.ndarray, tau: float) -> jnp.ndarray:
  """predict_threshold with extra argument checks raising ValueError."""
  _check_predict(probabilities)
  # tau can be unconstrained (i.e., also negative) as it might have been
  # calibrated on logits.
  return predict_threshold(probabilities, tau)


def _check_reg(classes: int, k_reg: Optional[int], lambda_reg: Optional[float]):
  """Helper for checking valid regularization arguments.

  Args:
    classes: number of classes
    k_reg: target size of confidence sets
    lambda_reg: strength of regularization

  Raises:
    Value Error if regularization arguments are incorrect.
  """
  if k_reg is not None and lambda_reg is not None:
    if lambda_reg < 0:
      raise ValueError('Expecting k_lambda to be a float >= 0.')
    if k_reg < 0 or k_reg > classes:
      raise ValueError('Expecting k_reg to be an int in [0, n_classes].')


def calibrate_raps(
    probabilities: jnp.ndarray,
    labels: jnp.ndarray,
    alpha: float = 0.1,
    k_reg: Optional[int] = None,
    lambda_reg: Optional[float] = None,
    rng: Optional[jnp.array] = None,
    quantile_fn: _QuantileFn = conformal_quantile) -> float:
  """Implementation of calibration for adaptive prediction sets.

  Following [1] and [2], this function implements adaptive prediction sets (APS)
  -- i.e., conformal classification. This methods estimates tau as outlined in
  [2] but without the confidence set size regularization.

  [1] Yaniv Romano, Matteo Sesia, Emmanuel J. Candes.
  Classification withvalid and adaptive coverage.
  NeurIPS, 2020.
  [2] Anastasios N. Angelopoulos, Stephen Bates, Michael Jordan, Jitendra Malik.
  Uncertainty sets for image classifiers using conformal prediction.
  ICLR, 2021

  Args:
    probabilities: predicted probabilities on validation set
    labels: ground truth labels on validation set
    alpha: confidence level
    k_reg: target confidence set size for regularization
    lambda_reg: regularization weight
    rng: random key for uniform variables
    quantile_fn: function to compute conformal quantile

  Returns:
    Threshold tau such that with probability 1 - alpha, the confidence set
    constructed from tau includes the true label
  """
  reg = k_reg is not None and lambda_reg is not None

  sorting = jnp.argsort(-probabilities, axis=1)
  reverse_sorting = jnp.argsort(sorting)
  indices = jnp.indices(probabilities.shape)
  sorted_probabilities = probabilities[indices[0], sorting]
  cum_probabilities = jnp.cumsum(sorted_probabilities, axis=1)

  rand = jnp.zeros((sorted_probabilities.shape[0]))
  if rng is not None:
    rand = jax.random.uniform(rng, shape=(sorted_probabilities.shape[0],))
  cum_probabilities -= jnp.expand_dims(rand, axis=1) * sorted_probabilities

  conformity_scores = cum_probabilities[
      jnp.arange(cum_probabilities.shape[0]),
      reverse_sorting[jnp.arange(reverse_sorting.shape[0]), labels]]

  if reg:
    # in [2], it seems that L_i can be zero (i.e., true class has highest
    # probability), but we add + 1 in the second line for validation
    # as the true class is included by design and only
    # additional classes should be regularized
    conformity_reg = reverse_sorting[jnp.arange(reverse_sorting.shape[0]),
                                     labels]
    conformity_reg = conformity_reg - k_reg + 1
    conformity_reg = lambda_reg * jnp.maximum(conformity_reg, 0)
    conformity_scores += conformity_reg

  tau = quantile_fn(conformity_scores, 1 - alpha)
  return tau


def calibrate_raps_with_checks(
    probabilities: jnp.ndarray,
    labels: jnp.ndarray,
    alpha: float = 0.1,
    k_reg: Optional[int] = None,
    lambda_reg: Optional[float] = None,
    rng: Optional[jnp.array] = None,
    quantile_fn: _QuantileFn = conformal_quantile) -> float:
  """calibrate_raps with extra argument checks raising ValueError."""
  _check_calibrate(probabilities, labels, alpha)
  _check_reg(probabilities.shape[1], k_reg, lambda_reg)
  return calibrate_raps(
      probabilities, labels, alpha, k_reg, lambda_reg, rng, quantile_fn)


def predict_raps(
    probabilities: jnp.ndarray,
    tau: float,
    k_reg: Optional[int] = None,
    lambda_reg: Optional[float] = None,
    rng: Optional[jnp.array] = None) -> jnp.ndarray:
  """Get confidence sets using tau computed via aps_calibrate.

  Given threshold tau, construct confidence sets as the top-k classes
  such that the sum of probabilities is still below tau and add the top-(k+1)
  class depending on uniform random variables.

  See calibrate_raps for details and references.

  Args:
    probabilities: predicted probabilities on test set
    tau: threshold
    k_reg: target confidence set size for regularization
    lambda_reg: regularization weight
    rng: random key for uniform variables

  Returns:
    Confidence sets as 0-1array of same size as probabilities.
  """
  reg = k_reg is not None and lambda_reg is not None

  sorting = jnp.argsort(-probabilities, axis=1)
  indices = jnp.indices(probabilities.shape)
  sorted_probabilities = probabilities[indices[0], sorting]
  cum_probabilities = jnp.cumsum(sorted_probabilities, axis=1)

  if reg:
    # in [2], L is the number of classes for which cumulative probability
    # mass and regularizer are below tau + 1, we account for that in
    # the first line by starting to count at 1
    reg_probabilities = jnp.repeat(
        jnp.expand_dims(1 + jnp.arange(cum_probabilities.shape[1]), axis=0),
        cum_probabilities.shape[0], axis=0)
    reg_probabilities = reg_probabilities - k_reg
    reg_probabilities = jnp.maximum(reg_probabilities, 0)
    cum_probabilities += lambda_reg * reg_probabilities

  rand = jnp.ones((sorted_probabilities.shape[0]))
  if rng is not None:
    rand = jax.random.uniform(rng, shape=(sorted_probabilities.shape[0],))
  cum_probabilities -= jnp.expand_dims(rand, axis=1) * sorted_probabilities

  sorted_confidence_sets = (cum_probabilities <= tau)

  # reverse sorting by argsort the sorting indices
  reverse_sorting = jnp.argsort(sorting, axis=1)
  confidence_sets = sorted_confidence_sets[indices[0], reverse_sorting]
  return confidence_sets.astype(int)


def predict_raps_with_checks(
    probabilities: jnp.ndarray,
    tau: float,
    k_reg: Optional[int] = None,
    lambda_reg: Optional[float] = None,
    rng: Optional[jnp.array] = None) -> jnp.ndarray:
  """predict_raps with extra argument checks raising ValueError."""
  _check_predict(probabilities)
  _check_reg(probabilities.shape[1], k_reg, lambda_reg)
  if tau < 0:
    raise ValueError('Expecting threshold tau to be greater or equal to zero.')
  return predict_raps(probabilities, tau, k_reg, lambda_reg, rng)
