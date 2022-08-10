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

"""Smooth implementation of conformal prediction approaches [1] and [2].

This module uses differentiable sorting to implement conformal prediction in a
differentiable manner [1,2], considering both calibration and prediction steps.

[1] Yaniv Romano, Matteo Sesia, Emmanuel J. Candes.
Classification withvalid and adaptive coverage.
NeurIPS, 2020.
[2] Mauricio Sadinle, Jing Lei, and Larry A. Wasserman.
Least ambiguous set-valued classifiers with bounded error levels.
ArXiv, 2016.
"""
import functools
from typing import Optional, Callable, Tuple, Any


import jax
import jax.numpy as jnp

from conformal_training import variational_sorting_net


_SmoothQuantileFn = Callable[[Any, float], float]
_ForwardFn = Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray],
    Tuple[jnp.ndarray, jnp.ndarray]]
_ForwardBackwardFn = Callable[
    [jnp.ndarray, jnp.ndarray],
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]


def _check_conformal_quantile(
    array: jnp.ndarray, q: float,
    sos: variational_sorting_net.VariationalSortingNet, dispersion: float):
  """Helper to check quantile arguments.

  Args:
    array: input array to compute quantile of
    q: quantile to compute
    sos: smooth order stat object for sorting
    dispersion: dispersion for smooth sorting

  Raises:
      ValueErrors if shape or q invalid.
  """
  if array.size == 0:
    raise ValueError('Expecting non-empty array.')
  if array.ndim != 1:
    raise ValueError('Expecting array of shape n.')
  if q < 0 or q > 1:
    raise ValueError('Expecting q in [0,1].')
  if sos.comms['num_wires'] != array.shape[0]:
    raise ValueError('Comm pattern has incorrect number of wires.')
  if dispersion <= 0:
    raise ValueError('Expecting dispersion strictly greater than zero.')


def smooth_conformal_quantile(
    array: jnp.ndarray, q: float,
    sos: variational_sorting_net.VariationalSortingNet,
    dispersion: float) -> float:
  """Smooth implementation of conformal quantile.

  Args:
    array: input array to compute quantile of
    q: quantile to compute
    sos: smooth order stat object
    dispersion: dispersion for smooth sorting

  Returns:
    (1 + 1/array.shape[0]) * q quantile of array.
  """
  return sos.quantile(
      array, dispersion=dispersion, alpha=(1 + 1./array.shape[0]) * q, tau=0.5)


def smooth_conformal_quantile_with_checks(
    array: jnp.ndarray, q: float,
    sos: variational_sorting_net.VariationalSortingNet,
    dispersion: float) -> float:
  """smooth_conformal_quantile with extra argument checks."""
  _check_conformal_quantile(array, q, sos, dispersion)
  return smooth_conformal_quantile(array, q, sos, dispersion)


def _check_probabilities(probabilities: jnp.ndarray):
  """Helper for checking probabilities for prediction or calibration.

  Args:
    probabilities: predicted probabilities on test or validation set

  Raises:
    ValueError if invalid arguments
  """
  if len(probabilities.shape) != 2:
    raise ValueError('Expecting probabilities of shape n_examples x n_classes.')
  if probabilities.size == 0:
    raise ValueError('probabilities is empty.')


def _check_sos(
    probabilities: jnp.ndarray,
    sos: variational_sorting_net.VariationalSortingNet,
    dispersion: float):
  """Helper for checking arguments for prediction or calibration.

  Args:
    probabilities: predicted probabilities on test or validation set
    sos: smooth order network
    dispersion: dispersion to use for smooth sort

  Raises:
    ValueError if invalid arguments
  """
  if sos.comms['num_wires'] != probabilities.shape[1]:
    raise ValueError('VariationalSortingNet used to sort n_classes elements, '
                     'comm pattern has incorrect number of wires.')
  if dispersion <= 0:
    raise ValueError('Expecting dispersion strictly greater than zero.')


def _check_predict(tau: float, temperature: float):
  """Helper for checking arguments for prediction.

  Args:
    tau: threshold
    temperature: temperature for smooth thresholding

  Raises:
    ValueError if invalid arguments
  """
  if tau < 0:
    raise ValueError('Expecting tau to be >= 0.')
  if temperature <= 0:
    raise ValueError('Expecting temperature strictly greater than zero.')


def _check_calibrate(
    probabilities: jnp.ndarray,
    labels: jnp.ndarray,
    alpha: float):
  """Helper for checking argumetns for calibration.

  Args:
    probabilities: predicted probabilities on validation set
    labels: ground truth labels on validation set
    alpha: confidence level

  Raises:
    ValueError if arguments invalid
  """
  if len(labels.shape) != 1:
    raise ValueError('Expecting labels of shape n_examples.')
  if probabilities.shape[0] != labels.shape[0]:
    raise ValueError(
        'Number of predicted probabilities does not match number of labels.')
  if jnp.max(labels) >= probabilities.shape[1]:
    raise ValueError('More labels than predicted in probabilities.')
  if not jnp.issubdtype(labels.dtype, jnp.integer):
    raise ValueError('Expecting labels to be integers.')
  if alpha < 0 or alpha > 1:
    raise ValueError('Expecting alpha to be in [0, 1].')


def _check_groups(probabilities, groups):
  """Helper for checking groups in subset aware class-conditional prediction.

  Args:
    probabilities: predicted probabilities
    groups: class group labels

  Raises:
    Value Error if groups are incorrect.
  """
  if groups.ndim != 1:
    raise ValueError('Expecting group labels of shape n_classes.')
  if not jnp.issubdtype(groups.dtype, jnp.integer):
    raise ValueError('Expecting group labels to be integers.')
  if groups.size != probabilities.shape[1]:
    raise ValueError('Number of group labels is not n_classes.')


def smooth_predict_threshold(
    probabilities: jnp.ndarray, tau: float, temperature: float) -> jnp.ndarray:
  """Smooth implementation of predict_threshold.

  Uses a sigmoid to implement soft thresholding.

  Args:
    probabilities: predicted probabilities or logits
    tau: threshold
    temperature: temperature for soft-thresholding

  Returns:
    Confidence sets
  """
  return jax.nn.sigmoid((probabilities - tau) / temperature)


def smooth_predict_threshold_with_checks(
    probabilities: jnp.ndarray, tau: float, temperature: float) -> jnp.ndarray:
  """smooth_predict_threshold with extra argument checks."""
  _check_probabilities(probabilities)
  _check_predict(tau, temperature)
  return smooth_predict_threshold(probabilities, tau, temperature)


def _get_sos_fns(
    sos: variational_sorting_net.VariationalSortingNet,
    dispersion: float) -> Tuple[_ForwardFn, _ForwardBackwardFn]:
  """Get forward and backward functions with given dispersion from sos.

  Args:
    sos: smooth order statistic object to use forward and backward from
    dispersion: dispersion to use for forward and backward

  Returns:
    Partials for forward and forward with backward
  """
  forward_fn = functools.partial(
      sos.forward_only, dispersion=dispersion,
      lower=0, upper=None, key=None)
  forward_backward_fn = functools.partial(
      sos.forward_backward, v=None, dispersion=dispersion,
      lower=0, upper=None, key=None)
  return forward_fn, forward_backward_fn


def smooth_predict_aps(
    probabilities: jnp.ndarray,
    tau: float, sos: variational_sorting_net.VariationalSortingNet,
    rng: Optional[jnp.ndarray] = None,
    temperature: float = 0.01, dispersion: float = 0.001) -> jnp.ndarray:
  """Smooth version of predict_raps without regularization.

  Uses variational sorting networks to perform smooth sorting and sigmoid for
  thresholding. The final confidence sets are fully differentiable with respect
  to the input probabilities.

  Args:
    probabilities: predicted probabilities on test set
    tau: threshold
    sos: smooth order network
    rng: PRNG key for sampling random variables
    temperature: temperature for soft thresholding, the lower the harder
      the thresholding
    dispersion: dispersion to use for smooth sort

  Returns:
    Confidence sets as arrays in [0,  1] after soft tresholding with given
    temperature.

  Raises:
    ValueError if probabilities have incorrect shape or tau is invalid.
  """

  forward_fn, forward_backward_fn = _get_sos_fns(sos, dispersion)

  def smooth_sort_fn(p, d):
    """Helper to vmap differentiable sorting across all examples.

    Args:
      p: vector of probabilities
      d: single number to put on the diagonal of the upper triangular matrix L

    Returns:
      Confidence sets for given probabilities
    """
    # Diagonal is set to zero by default, which is basically equivalent to
    # computing the cumulative sorted probability and afterwards
    # subtracting the individual (sorted) probabilities again.
    # This is done as, without randomization, we want the class
    # that just exceeds the threshold, to be included in the confidence set.
    matrix_l = jnp.triu(jnp.ones((p.shape[0], p.shape[0])))
    matrix_l = matrix_l.at[jnp.diag_indices(matrix_l.shape[0])].set(d)

    _, cum_sorted_p = forward_fn(-p, p, matrix_l)
    sorted_confidence_set = jax.nn.sigmoid(-(cum_sorted_p - tau)/temperature)
    _, confidence_set, _ = forward_backward_fn(-p, sorted_confidence_set)
    return confidence_set

  if rng is not None:
    diagonals = jax.random.uniform(rng, (probabilities.shape[0],))
  else:
    diagonals = jnp.zeros(probabilities.shape[0])
  smooth_sort_vmap = jax.vmap(smooth_sort_fn, (0, 0), 0)
  return smooth_sort_vmap(probabilities, diagonals)


def smooth_predict_aps_with_checks(
    probabilities: jnp.ndarray,
    tau: float, sos: variational_sorting_net.VariationalSortingNet,
    rng: Optional[jnp.ndarray] = None,
    temperature: float = 0.01, dispersion: float = 0.001) -> jnp.ndarray:
  """smooth_predict_aps with extra argument checks raising ValueError."""
  _check_probabilities(probabilities)
  _check_sos(probabilities, sos, dispersion)
  _check_predict(tau, temperature)
  return smooth_predict_aps(
      probabilities, tau, sos, rng, temperature, dispersion)


def smooth_calibrate_threshold(
    probabilities: jnp.ndarray, labels: jnp.ndarray, alpha: float,
    smooth_quantile_fn: _SmoothQuantileFn) -> float:
  """Smooth calibrate_threshold version.

  Args:
    probabilities: predicted probabilities or logits
    labels: corresponding ground truth labels
    alpha: confidence level
    smooth_quantile_fn: smooth quantile function to use

  Returns:
    Threshold
  """
  conformity_scores = probabilities[
      jnp.arange(probabilities.shape[0]), labels.astype(int)]
  return smooth_quantile_fn(conformity_scores, alpha)


def smooth_calibrate_threshold_with_checks(
    probabilities: jnp.ndarray, labels: jnp.ndarray, alpha: float,
    smooth_quantile_fn: _SmoothQuantileFn) -> float:
  """smooth_calibrate_threshold with extra argument checks."""
  _check_probabilities(probabilities)
  _check_calibrate(probabilities, labels, alpha)
  return smooth_calibrate_threshold(
      probabilities, labels, alpha, smooth_quantile_fn)


def smooth_calibrate_aps(
    probabilities: jnp.ndarray,
    labels: jnp.ndarray,
    alpha: float,
    sos: variational_sorting_net.VariationalSortingNet,
    dispersion: float,
    smooth_quantile_fn: _SmoothQuantileFn,
    rng: Optional[jnp.ndarray] = None,
) -> float:
  """Smooth implementation of calibrate_raps without regularization.

  Args:
    probabilities: predicted probabilities on validation set
    labels: ground truth labels on validation set
    alpha: confidence level
    sos: smooth order network for probabilities, i.e.,
      has to allow sorting n_classes elements
    dispersion: dispersion to use for smooth sort.
    smooth_quantile_fn: smooth conformal quantile function to use
    rng: PRNG key for sampling random variables

  Returns:
    Threshold.

  Raises:
    ValueError if probabilities have incorrect shape or alpha is invalid.
  """

  forward_fn, forward_backward_fn = _get_sos_fns(sos, dispersion)

  def smooth_sort_fn(p, d, l):
    matrix_l = jnp.triu(jnp.ones((p.shape[0], p.shape[0])))
    matrix_l = matrix_l.at[jnp.diag_indices(matrix_l.shape[0])].set(d)
    _, cum_sorted_p = forward_fn(
        -p, p, matrix_l)
    _, cum_p, _ = forward_backward_fn(
        -p, cum_sorted_p)
    return  cum_p[l]

  if rng is not None:
    diagonals = jax.random.uniform(rng, (probabilities.shape[0],))
  else:
    diagonals = jnp.ones(probabilities.shape[0])
  smooth_sort_vmap = jax.vmap(smooth_sort_fn, (0, 0, 0), 0)
  scores = smooth_sort_vmap(probabilities, diagonals, labels)
  return smooth_quantile_fn(scores, 1 - alpha)


def smooth_calibrate_aps_with_checks(
    probabilities: jnp.ndarray,
    labels: jnp.ndarray,
    alpha: float,
    sos: variational_sorting_net.VariationalSortingNet,
    dispersion: float,
    smooth_quantile_fn: _SmoothQuantileFn,
    rng: Optional[jnp.ndarray] = None,
) -> float:
  """smooth_calibrate_aps with additional argument checks."""
  _check_probabilities(probabilities)
  _check_sos(probabilities, sos, dispersion)
  _check_calibrate(probabilities, labels, alpha)
  return smooth_calibrate_aps(
      probabilities, labels, alpha, sos,
      dispersion, smooth_quantile_fn, rng)
