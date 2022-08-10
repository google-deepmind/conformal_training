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

"""Training loop for coverage training, i.e., with confidence set prediction."""
import functools
import itertools
from typing import Tuple, Dict, Any, Callable, Union

from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections as collections

import sorting_nets
import variational_sorting_net
import conformal_prediction as cp
import data as cpdata
import evaluation as cpeval
import smooth_conformal_prediction as scp
import train_normal as cpnormal
import train_utils as cputils


SizeLossFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
CoverageLossFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
SmoothPredictFn = Callable[[jnp.ndarray, Any, jnp.ndarray], jnp.ndarray]

_CalibrateFn = Callable[
    [Union[Any, jnp.ndarray], Union[Any, jnp.ndarray], Union[Any, jnp.ndarray]],
    Union[Any, jnp.ndarray]]
_LossTransformFn = Callable[[jnp.ndarray], jnp.ndarray]


class TrainCoverage(cpnormal.TrainNormal):
  """Coverage training routine following [1] but adapted to also work with APS.

  Trains a model by predicting confidence sets using some soft confidence set
  prediction method. [1] uses simple soft-thresholding with a fixed threshold
  but a smooth implementation of [2] can also be used. See
  smooth_conformal_prediction.

  [1] Anthony Bellotti.
  Optimized conformal classification using gradient descent approximation.
  ArXiv, 2021.
  [2] Yaniv Romano, Matteo Sesia, Emmanuel J. Candes.
  Classification withvalid and adaptive coverage.
  NeurIPS, 2020.
  """

  def __init__(self, config, data, optimizer):
    """Initialize coverage training.

    Args:
      config: training configuration
      data: datasets and information
      optimizer: optimizer to use
    """
    super(TrainCoverage, self).__init__(config, data, optimizer)

    self.fixed_smooth_predict_fn = None
    """(callable) Fixed smooth prediction function to get confidence sets."""
    self.calibrate_fn = None
    """(callable) Conformal prediction calibration function for fine-tuning."""
    self.coverage_loss_fn = None
    """(callable) Loss function for confidence sets."""
    self.size_loss_fn = None
    """(callable) Size loss for confidence sets."""
    self.loss_transform_fn = None
    """(callable) Monotonic transform of coverage + size loss."""
    self.tau = None
    """ (float) For fine-tuning, tau needs to be calibrated. """

  def compute_loss_and_error(
      self,
      trainable_params: cputils.FlatMapping,
      fixed_params: cputils.FlatMapping,
      inputs: jnp.ndarray,
      labels: jnp.ndarray,
      model_state: cputils.FlatMapping,
      training: bool,
      rng: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, Tuple[cputils.FlatMapping, Dict[str, Any]]]:
    """Compute coverage loss and size loss.

    Args:
      trainable_params: trainable model parameters
      fixed_params: model parameters fixed for fine-tuning
      inputs: input examples
      labels: ground truth examples
      model_state: model state
      training: training mode
      rng: random key

    Returns:
      Tuple consisting of loss and another tuple of new model state and a
      dictionary with additional information
    """
    forward_rng, predict_rng = None, None
    if rng is not None:
      forward_rng, predict_rng = jax.random.split(rng, 2)

    params = hk.data_structures.merge(trainable_params, fixed_params)
    logits, new_model_state = self.model.apply(
        params, model_state, forward_rng, inputs, training=training)

    confidence_sets = self.fixed_smooth_predict_fn(
        logits, self.tau, predict_rng)
    coverage_loss = self.coverage_loss_fn(confidence_sets, labels)
    size_loss = self.size_loss_fn(confidence_sets, logits, labels)
    size_loss *= self.config.coverage.size_weight

    weight_decay_loss = cputils.compute_weight_decay(params)
    weight_decay_loss *= self.config.weight_decay
    cross_entropy_loss = cputils.compute_cross_entropy_loss(logits, labels)
    cross_entropy_loss *= self.config.coverage.cross_entropy_weight
    loss = self.loss_transform_fn(coverage_loss + size_loss + 1e-8)
    loss += cross_entropy_loss
    loss += weight_decay_loss

    confidence_sets = jnp.greater(
        confidence_sets, jnp.ones_like(confidence_sets) * 0.5)
    error = 1 - cpeval.compute_accuracy(logits, labels)
    coverage = cpeval.compute_coverage(confidence_sets, labels)
    size, _ = cpeval.compute_size(confidence_sets)

    return loss, (new_model_state, {
        'coverage_loss': coverage_loss,
        'size_loss': size_loss,
        'cross_entropy_loss': cross_entropy_loss,
        'weight_decay': weight_decay_loss,
        'error': error,
        'coverage': coverage,
        'size': size,
    })

  def get_sos(
      self, length: int) -> variational_sorting_net.VariationalSortingNet:
    """Set up smooth order stat object for given array length.

    Args:
      length: length of array to be sorted

    Returns:
      Smooth order stat object
    """
    comm = sorting_nets.comm_pattern_batcher(
        length, make_parallel=True)
    sos = variational_sorting_net.VariationalSortingNet(
        comm, smoothing_strategy='entropy_reg', sorting_strategy='hard')
    return sos

  def get_class_groups(
      self, config: collections.ConfigDict) -> Tuple[jnp.ndarray, int]:
    """Get class groups for predict/calibrate from configuration.

    Args:
      config: sub-configuration to get groups from

    Returns:
      Class groups, number of groups
    """
    classes = self.data['classes']
    if config.class_groups:
      groups = jnp.array(config.class_groups)
    else:
      groups = jnp.arange(classes)
    if groups.size != classes:
      raise ValueError('Loss matrix has to be num_classes x num_classes')
    logging.info('Class groups to be used:')
    logging.info(groups)
    return groups, jnp.max(groups) + 1

  def select_calibrate(
      self, config: collections.ConfigDict) -> _CalibrateFn:
    """Select calibration function.

    Args:
      config: sub-configuration to determine calibration function

    Returns:
      Calibration function
    """
    if config.method == 'threshold':
      def calibrate_fn(logits, labels, unused_rng):
        return cp.calibrate_threshold(logits, labels, alpha=config.alpha)
    elif config.method == 'threshold_p':
      def calibrate_fn(logits, labels, unused_rng):
        probabilities = jax.nn.softmax(logits, axis=1)
        return cp.calibrate_threshold(probabilities, labels, alpha=config.alpha)
    elif config.method == 'threshold_logp':
      def calibrate_fn(logits, labels, unused_rng):
        log_probabilities = jax.nn.log_softmax(logits, axis=1)
        return cp.calibrate_threshold(
            log_probabilities, labels, alpha=config.alpha)
    elif config.method == 'aps':
      def calibrate_fn(logits, labels, rng):
        probabilities = jax.nn.softmax(logits, axis=1)
        return cp.calibrate_raps(
            probabilities, labels, alpha=config.alpha,
            k_reg=None, lambda_reg=None, rng=rng)
    else:
      raise ValueError('Invalid calibration method.')
    return calibrate_fn

  def select_smooth_predict(
      self, config: collections.ConfigDict) -> SmoothPredictFn:
    """Select smooth confidence set prediction and calibration functions.

    See smooth_conformal_prediction for options.

    Args:
      config: sub-configuration for selecting prediction/calibration function

    Returns:
      Smooth prediction function
    """
    if config.method == 'threshold':
      def smooth_predict_fn(logits, tau, unused_rng):
        return scp.smooth_predict_threshold(
            logits, tau,
            temperature=config.temperature)
    elif config.method == 'threshold_p':
      def smooth_predict_fn(logits, tau, unused_rng):
        probabilities = jax.nn.softmax(logits, axis=1)
        return scp.smooth_predict_threshold(
            probabilities, tau,
            temperature=config.temperature)
    elif config.method == 'threshold_logp':
      def smooth_predict_fn(logits, tau, unused_rng):
        log_probabilities = jax.nn.log_softmax(logits, axis=1)
        return scp.smooth_predict_threshold(
            log_probabilities, tau,
            temperature=config.temperature)
    elif config.method == 'aps':
      sos = self.get_sos(self.data['classes'])
      def smooth_predict_fn(logits, tau, rng):
        probabilities = jax.nn.softmax(logits, axis=1)
        return scp.smooth_predict_aps(
            probabilities, tau,
            temperature=config.temperature,
            sos=sos, rng=rng if config.rng else None,
            dispersion=config.dispersion)
    else:
      raise ValueError('Invalid smooth prediction method.')
    return smooth_predict_fn

  def get_loss_matrix(self, config: collections.ConfigDict) -> jnp.ndarray:
    """Get loss matrix for coverage loss from configuration.

    Args:
      config: sub-configuration to get loss matrix from

    Returns:
      Loss matrix
    """
    classes = self.data['classes']
    if config.loss_matrix:
      loss_matrix = jnp.array(config.loss_matrix).reshape(classes, classes)
    else:
      loss_matrix = jnp.identity(classes)
    if loss_matrix.shape[0] != classes or loss_matrix.shape[1] != classes:
      raise ValueError('Loss matrix has to be num_classes x num_classes')
    logging.info('Loss matrix for classification loss to be used:')
    logging.info(loss_matrix)
    return loss_matrix

  def select_coverage_loss(
      self, config: collections.ConfigDict) -> CoverageLossFn:
    """Select coverage loss to use for training.

    Args:
      config: sub-configuration to select coverage loss

    Returns:
      Coverage loss
    """
    loss_matrix = self.get_loss_matrix(config)
    if config.coverage_loss == 'none':
      def coverage_loss_fn(unused_confidence_sets, unused_labels):
        return 0.
    elif config.coverage_loss == 'absolute_coverage':
      coverage_loss_fn = functools.partial(
          cputils.compute_coverage_loss,
          alpha=config.target_alpha, transform=jnp.abs)
    elif config.coverage_loss == 'squared_coverage':
      coverage_loss_fn = functools.partial(
          cputils.compute_coverage_loss,
          alpha=config.target_alpha, transform=jnp.square)
    elif config.coverage_loss == 'classification':
      coverage_loss_fn = functools.partial(
          cputils.compute_general_classification_loss,
          loss_matrix=loss_matrix)
    elif config.coverage_loss == 'bce':
      coverage_loss_fn = functools.partial(
          cputils.compute_general_binary_cross_entropy_loss,
          loss_matrix=loss_matrix)
    else:
      raise ValueError('Invalid coverage loss.')
    return coverage_loss_fn

  def select_size_loss(
      self, config: collections.ConfigDict) -> SizeLossFn:
    """Select size loss to use.

    Args:
      config: sub-configuration to select size loss

    Returns:
      Size loss
    """
    if config.size_transform == 'identity':
      size_transform_fn = lambda x: x
    elif config.size_transform == 'log':
      size_transform_fn = jnp.log
    elif config.size_transform == 'square':
      size_transform_fn = jnp.square
    elif config.size_transform == 'abs':
      size_transform_fn = jnp.abs
    else:
      raise ValueError('Invalid size transform')

    if config.size_loss == 'valid':
      selected_size_loss_fn = functools.partial(
          cputils.compute_hinge_size_loss, target_size=1,
          transform=size_transform_fn)
    elif config.size_loss == 'normal':
      selected_size_loss_fn = functools.partial(
          cputils.compute_hinge_size_loss, target_size=0,
          transform=size_transform_fn)
    elif config.size_loss == 'valid_bounded':
      selected_size_loss_fn = functools.partial(
          cputils.compute_hinge_bounded_size_loss, target_size=1,
          bound_size=config.size_bound, bound_weight=config.size_bound_weight,
          transform=size_transform_fn)
    elif config.size_loss == 'normal_bounded':
      selected_size_loss_fn = functools.partial(
          cputils.compute_hinge_bounded_size_loss, target_size=0,
          bound_size=config.size_bound, bound_weight=config.size_bound_weight,
          transform=size_transform_fn)
    elif config.size_loss == 'probabilistic':
      selected_size_loss_fn = cputils.compute_probabilistic_size_loss
    else:
      raise ValueError('Invalid size loss.')

    classes = self.data['classes']
    if config.size_weights:
      size_weights = jnp.array(config.size_weights)
    else:
      size_weights = jnp.ones(classes)
    if size_weights.shape[0] != classes:
      raise ValueError('Could not use size weights due to invalid shape: %d' % (
          size_weights.shape[0]))
    logging.info('Size weights by class for size loss to be used:')
    logging.info(size_weights)

    def size_loss_fn(confidence_sets, unused_logits, labels):
      """Wrapper for size loss as most size losses only need confidence_sets."""
      weights = size_weights[labels]
      return selected_size_loss_fn(confidence_sets, weights=weights)

    return size_loss_fn

  def select_loss_transform(
      self, config: collections.ConfigDict) -> _LossTransformFn:
    """Select loss transform to apply.

    Args:
      config: sub-configuration to select loss transform

    Returns:
      Loss transform
    """
    if config.loss_transform == 'identity':
      loss_transform_fn = lambda array: array
    elif config.loss_transform == 'log':
      loss_transform_fn = jnp.log
    elif config.loss_transform == 'inverse':
      loss_transform_fn = lambda array: -1./array
    elif config.loss_transform == 'inverse_square':
      loss_transform_fn = lambda array: -1./(array**2)
    else:
      raise ValueError('Invalid loss transform.')
    return loss_transform_fn

  def get_train_fns(self) -> Tuple[cputils.LossFn, functools.partial]:
    """Define loss and update functions for training.

    Returns:
      Loss and update function
    """
    self.fixed_smooth_predict_fn = self.select_smooth_predict(
        self.config.coverage)
    self.coverage_loss_fn = self.select_coverage_loss(self.config.coverage)
    self.size_loss_fn = self.select_size_loss(self.config.coverage)
    self.loss_transform_fn = self.select_loss_transform(self.config.coverage)
    loss_fn = self.compute_loss_and_error
    update_fn = functools.partial(
        cputils.update, loss_fn=loss_fn, optimizer=self.optimizer)
    if self.config.jit:
      loss_fn = jax.jit(loss_fn, static_argnames='training')
      update_fn = jax.jit(update_fn, static_argnames='training')
    return loss_fn, update_fn

  def calibrate(self, params, model_state, rng):
    """Calibrate fixed tau used for coverage training.

    Args:
      params: model parameters
      model_state: model state
      rng: random key sequence

    Returns:
      Calibrated tau
    """

    if self.config.mode != 'coverage':
      raise ValueError(
          'Trying to calibrate tau before training but '
          'not in coverage training mode.')
    if not self.calibrate_fn:
      raise ValueError(
          'Trying to calibrate for fine-tuning but calibrate_fn not defined; '
          'in coverage training mode this should not happen.')

    # When not fine-tuning, tau can be arbitrary in most cases.
    # For fine-tuning, we calibrate tau once as the model usually
    # performs quite well already.
    tau = self.config.coverage.tau
    if self.config.finetune.enabled:
      val_ds = itertools.islice(
          cpdata.load_batches(self.data['train']),
          0, self.config.coverage.calibration_batches)
      logits = []
      labels = []
      for inputs_b, labels_b in val_ds:
        logits_b, _ = self.model.apply(
            params, model_state, None, inputs_b, training=False)
        logits.append(logits_b)
        labels.append(labels_b)
      logits = jnp.concatenate(logits, axis=0)
      labels = jnp.concatenate(labels, axis=0)
      tau = self.calibrate_fn(logits, labels, next(rng))
      logging.info('Threshold after calibration of pre-trained model: %g', tau)

    return tau

  def run(self, rng: hk.PRNGSequence):
    """Main training procedure but with calibration if fine-tuning.

    Args:
      rng: random key sequence
    """
    trainable_params, fixed_params, model_state = self.setup(rng)
    self.calibrate_fn = self.select_calibrate(self.config.coverage)
    self.tau = self.calibrate(
        hk.data_structures.merge(trainable_params, fixed_params),
        model_state, rng)
    params, model_state = self.train(
        trainable_params, fixed_params, model_state, rng)
    self.test(params, model_state)
