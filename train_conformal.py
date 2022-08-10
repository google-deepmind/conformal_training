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

"""Smooth conformal training with prediction and calibration."""
import functools
from typing import Tuple, Dict, Any, Callable, Union

import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections as collections


import evaluation as cpeval
import smooth_conformal_prediction as scp
import train_coverage as cpcoverage
import train_utils as cputils


SmoothCalibrateFn = Callable[
    [Union[Any, jnp.ndarray], Union[Any, jnp.ndarray], Union[Any, jnp.ndarray]],
    Union[Any, jnp.ndarray]]


class TrainConformal(cpcoverage.TrainCoverage):
  """Conformal training takes into account calibration and prediction."""

  def __init__(self, config, data, optimizer):
    """Initialize conformal training.

    Args:
      config: training configuration
      data: datasets and information
      optimizer: optimizer to use
    """
    super(TrainConformal, self).__init__(config, data, optimizer)

    self.smooth_predict_fn = None
    """(callable) Smooth prediction function to get confidence sets."""
    # We need separate calibration functions for training and testing
    # to allow different batch sizes.
    self.train_smooth_calibrate_fn = None
    """(callable) Training smooth conformal calibration function."""
    self.test_smooth_calibrate_fn = None
    """(callable) Test smooth conformal calibration function."""

  def compute_loss_and_error_with_calibration(
      self,
      trainable_params: cputils.FlatMapping,
      fixed_params: cputils.FlatMapping,
      inputs: jnp.ndarray,
      labels: jnp.ndarray,
      model_state: cputils.FlatMapping,
      training: bool,
      rng: jnp.ndarray,
      # The calibration function needs to be passed as argument because
      # we need to make two copies of compute_loss_and_error: one for
      # training and one for testing. This is because smooth_calibrate_fn
      # depends on the batch size, which we allow to change between training
      # and test set for datasets with very few examples.
      smooth_calibrate_fn: SmoothCalibrateFn,
  ) -> Tuple[jnp.ndarray, Tuple[cputils.FlatMapping, Dict[str, Any]]]:
    """Compute conformal loss with prediction and calibration on split batch.

    Calibrates the conformal predictor on the first half of the batch and
    computes coverage and size loss on the second half of the batch.

    Args:
      trainable_params: trainable model parameters
      fixed_params: model parameters fixed for fine-tuning
      inputs: input examples
      labels: ground truth examples
      model_state: model state
      training: training mode
      rng: random key
      smooth_calibrate_fn: smooth calibration function

    Returns:
      Tuple consisting of loss and another tuple of new model state and a
      dictionary with additional information
    """
    params = hk.data_structures.merge(trainable_params, fixed_params)
    logits, new_model_state = self.model.apply(
        params, model_state, rng, inputs, training=training)

    val_split = int(self.config.conformal.fraction * logits.shape[0])
    val_logits = logits[:val_split]
    val_labels = labels[:val_split]
    test_logits = logits[val_split:]
    test_labels = labels[val_split:]
    val_tau = smooth_calibrate_fn(val_logits, val_labels, rng)

    test_confidence_sets = self.smooth_predict_fn(test_logits, val_tau, rng)
    coverage_loss = self.coverage_loss_fn(test_confidence_sets, test_labels)
    size_loss = self.size_loss_fn(
        test_confidence_sets, test_logits, test_labels)
    size_loss *= self.config.conformal.size_weight

    weight_decay_loss = cputils.compute_weight_decay(params)
    weight_decay_loss *= self.config.weight_decay
    cross_entropy_loss = cputils.compute_cross_entropy_loss(logits, labels)
    cross_entropy_loss *= self.config.conformal.cross_entropy_weight
    loss = self.loss_transform_fn(coverage_loss + size_loss + 1e-8)
    loss += cross_entropy_loss
    loss += weight_decay_loss

    test_confidence_sets = jnp.greater(
        test_confidence_sets, jnp.ones_like(test_confidence_sets) * 0.5)
    error = 1 - cpeval.compute_accuracy(logits, labels)
    coverage = cpeval.compute_coverage(test_confidence_sets, test_labels)
    size, _ = cpeval.compute_size(test_confidence_sets)

    return loss, (new_model_state, {
        'coverage_loss': coverage_loss,
        'size_loss': size_loss,
        'cross_entropy_loss': cross_entropy_loss,
        'weight_decay': weight_decay_loss,
        'error': error,
        'coverage': coverage,
        'size': size,
    })

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
    """To be safe, override as not implemented."""
    raise NotImplementedError

  def select_smooth_calibrate(
      self,
      config: collections.ConfigDict
  ) -> Tuple[SmoothCalibrateFn, SmoothCalibrateFn]:
    """Select smooth confidence set prediction and calibration functions.

    See smooth_conformal_prediction for options.

    Args:
      config: sub-configuration for selecting prediction/calibration function

    Returns:
      Smooth calibration function
    """
    train_calibration_examples = int(
        self.config.conformal.fraction * self.config.batch_size)
    test_calibration_examples = int(
        self.config.conformal.fraction * self.config.test_batch_size)

    def get_smooth_quantile_fn(calibration_examples):
      """Helper to create smooth quantile function for given #examples."""
      return functools.partial(
          scp.smooth_conformal_quantile,
          sos=self.get_sos(calibration_examples),
          dispersion=config.dispersion)

    get_right_smooth_quantile_fn = get_smooth_quantile_fn
    if config.method == 'threshold':
      def smooth_calibrate_fn(logits, labels, unused_rng, quantile_fn):
        return scp.smooth_calibrate_threshold(
            logits, labels, alpha=config.alpha,
            smooth_quantile_fn=quantile_fn)
    elif config.method == 'threshold_p':
      def smooth_calibrate_fn(logits, labels, unused_rng, quantile_fn):
        probabilities = jax.nn.softmax(logits, axis=1)
        return scp.smooth_calibrate_threshold(
            probabilities, labels, alpha=config.alpha,
            smooth_quantile_fn=quantile_fn)
    elif config.method == 'threshold_logp':
      def smooth_calibrate_fn(logits, labels, unused_rng, quantile_fn):
        log_probabilities = jax.nn.log_softmax(logits, axis=1)
        return scp.smooth_calibrate_threshold(
            log_probabilities, labels, alpha=config.alpha,
            smooth_quantile_fn=quantile_fn)
    elif config.method == 'aps':
      sos = self.get_sos(self.data['classes'])
      def smooth_calibrate_fn(logits, labels, rng, quantile_fn):
        probabilities = jax.nn.softmax(logits, axis=1)
        return scp.smooth_calibrate_aps(
            probabilities, labels,
            alpha=config.alpha,
            sos=sos, rng=rng if config.rng else None,
            dispersion=config.dispersion,
            smooth_quantile_fn=quantile_fn)
    else:
      raise ValueError('Invalid smooth calibration method.')

    train_smooth_calibrate_fn = functools.partial(
        smooth_calibrate_fn,
        quantile_fn=get_right_smooth_quantile_fn(train_calibration_examples))
    test_smooth_calibrate_fn = functools.partial(
        smooth_calibrate_fn,
        quantile_fn=get_right_smooth_quantile_fn(test_calibration_examples))
    return train_smooth_calibrate_fn, test_smooth_calibrate_fn

  def get_conformal_config(self):
    """Overridable helper to select the right config.

    Returns:
      Configuration for conformal training
    """
    return self.config.conformal

  def get_train_fns(
      self) -> Tuple[cputils.LossFn, functools.partial]:
    """For conformal training, we use separate training and test loss fn."""
    conformal_config = self.get_conformal_config()
    smooth_calibrate_fns = self.select_smooth_calibrate(conformal_config)
    self.train_smooth_calibrate_fn = smooth_calibrate_fns[0]
    self.test_smooth_calibrate_fn = smooth_calibrate_fns[1]
    self.smooth_predict_fn = self.select_smooth_predict(conformal_config)
    self.coverage_loss_fn = self.select_coverage_loss(conformal_config)
    self.size_loss_fn = self.select_size_loss(conformal_config)
    self.loss_transform_fn = self.select_loss_transform(conformal_config)
    train_loss_fn = functools.partial(
        self.compute_loss_and_error_with_calibration,
        smooth_calibrate_fn=self.train_smooth_calibrate_fn)
    test_loss_fn = functools.partial(
        self.compute_loss_and_error_with_calibration,
        smooth_calibrate_fn=self.test_smooth_calibrate_fn)
    # The training loss is only used within the update function.
    update_fn = functools.partial(
        cputils.update, loss_fn=train_loss_fn, optimizer=self.optimizer)
    if self.config.jit:
      test_loss_fn = jax.jit(test_loss_fn, static_argnames='training')
      update_fn = jax.jit(update_fn, static_argnames='training')
    return test_loss_fn, update_fn

  # Need to override this again from TrainCoverage as we do not need
  # separate calibration when fine-tuning.
  def run(self, rng: hk.PRNGSequence):
    """Main training procedure.

    Args:
      rng: random key sequence
    """
    trainable_params, fixed_params, model_state = self.setup(rng)
    params, model_state = self.train(
        trainable_params, fixed_params, model_state, rng)
    self.test(params, model_state)
