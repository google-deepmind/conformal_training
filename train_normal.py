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

"""Training loop for normal training."""
import functools
from typing import Callable, Tuple, Dict, Any

from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

import conformal_training.data as cpdata
import conformal_training.evaluation as cpeval
import conformal_training.open_source_utils as cpstaging
import conformal_training.train_utils as cputils


ShiftFn = Callable[[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]


class TrainNormal:
  """Normal training routine."""

  def __init__(self, config, data, optimizer):
    """Initialize normal training.

    Args:
      config: training configuration
      data: datasets and information
      optimizer: optimizer to use
    """
    self.config = config
    """ (collections.ConfigDict) Training configuration. """
    self.data = data
    """ (Dict[str, any]) Datasets and information."""
    self.model = None
    """ (hk.TransformedWithState) Model to train. """
    self.optimizer = optimizer
    """ (optax.GradientTransformation) Optimizer for training. """
    # Mainly for conformal training and backwards compatibility, we use
    # the same batch size for training and testing by default.
    if self.config.test_batch_size is None:
      self.config.test_batch_size = self.config.batch_size

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
    """Compute cross-entropy loss with weight decay and error rate.

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
    params = hk.data_structures.merge(trainable_params, fixed_params)
    logits, new_model_state = self.model.apply(
        params, model_state, rng, inputs, training=training)
    cross_entropy_loss = cputils.compute_cross_entropy_loss(logits, labels)
    weight_decay_loss = cputils.compute_weight_decay(params)
    weight_decay_loss *= self.config.weight_decay
    error = 1 - cpeval.compute_accuracy(logits, labels)
    loss = cross_entropy_loss + weight_decay_loss
    return loss, (new_model_state, {
        'cross_entropy': cross_entropy_loss,
        'weight_decay': weight_decay_loss,
        'error': error,
    })

  def get_train_fns(self) -> Tuple[cputils.LossFn, functools.partial]:
    """Get training loss and update function.

    Returns:
      Loss and update function
    """
    loss_fn = self.compute_loss_and_error
    update_fn = functools.partial(
        cputils.update, loss_fn=loss_fn, optimizer=self.optimizer)
    if self.config.jit:
      loss_fn = jax.jit(loss_fn, static_argnames='training')
      update_fn = jax.jit(update_fn, static_argnames='training')
    return loss_fn, update_fn

  def setup(
      self, rng: hk.PRNGSequence
  ) -> Tuple[cputils.FlatMapping, cputils.FlatMapping, cputils.FlatMapping]:
    """Set up model.

    Args:
      rng: random key sequence

    Returns:
      Trainable parameters, fixed parameters and model state
    """
    def update_flatmapping(base_mapping, mapping, excluded_layers):
      """Helper to update params and model state with loaded ones."""
      mapping = hk.data_structures.to_mutable_dict(mapping)
      for key in base_mapping.keys():
        include = True
        for excluded_layer in excluded_layers:
          if key.find(excluded_layer) >= 0:
            include = False
        if include:
          mapping[key] = base_mapping[key]
      return hk.data_structures.to_haiku_dict(mapping)
    def partition_params(module_name, unused_name, unused_value):
      """Helper to partition parameters into trainable and fixed."""
      return (self.config.finetune.layers is None
              or module_name in include_layers)
    def log_params(params):
      """Helper to log a set of parameters."""
      for module_name, name, _ in hk.data_structures.traverse(params):
        logging.info('%s.%s', module_name, name)

    if self.config.finetune.enabled:
      # Layers to be fine-tuned:
      include_layers = self.config.finetune.layers or ''
      include_layers = include_layers.split(',')

      path, self.model, base_params, base_model_state = cputils.load_model(
          self.config, self.data)

      logging.info('Loaded pre-trained model from %s.', path)
      # We re-initialize the whole model and set the loaded parameters
      # only for those layers that are not supposed to be fine-tuned.
      if self.config.finetune.reinitialize:
        params, model_state = cputils.init_model(
            self.data, self.model, rng)
        params = update_flatmapping(base_params, params, include_layers)
        model_state = update_flatmapping(
            base_model_state, model_state, include_layers)
      else:
        params = base_params
        model_state = base_model_state

      trainable_params, fixed_params = hk.data_structures.partition(
          partition_params, params)

    # For training from scratch we just set all parameters as trainable.
    else:
      self.model = cputils.create_model(self.config, self.data)
      trainable_params, model_state = cputils.init_model(
          self.data, self.model, rng)
      fixed_params = {}
      logging.info('Created model %s.', self.config.architecture)

    logging.info('Trainable parameters:')
    log_params(trainable_params)
    logging.info('Fixed parameteers:')
    log_params(fixed_params)

    return trainable_params, fixed_params, model_state

  def train(
      self, trainable_params: cputils.FlatMapping,
      fixed_params: cputils.FlatMapping,
      model_state: cputils.FlatMapping, rng: hk.PRNGSequence
  ) -> Tuple[cputils.FlatMapping, cputils.FlatMapping]:
    """Normal training loop.

    Args:
      trainable_params: parameters to train
      fixed_params: fixed parameters in the case of fine-tuning
      model_state: model state
      rng: random key sequence

    Returns:
      Parameters and model state
    """
    optimizer_state = self.optimizer.init(trainable_params)
    logging.info('Initialized optimizer for training.')

    loss_fn, update_fn = self.get_train_fns()
    checkpoint = cpstaging.create_checkpoint(self.config)
    cputils.update_checkpoint(
        checkpoint, trainable_params, fixed_params,
        model_state, optimizer_state, 0)
    checkpoint.restore_or_save()

    while checkpoint.state.epoch < self.config.epochs:
      logging.info('Epoch %d:', checkpoint.state.epoch)
      for b, (inputs, labels) in enumerate(
          cpdata.load_batches(self.data['train'])):
        loss, trainable_params, new_model_state, optimizer_state, mixed = update_fn(
            trainable_params, fixed_params, inputs, labels, model_state,
            True, optimizer_state, next(rng))
        if not self.config.finetune.enabled or self.config.finetune.model_state:
          model_state = new_model_state
        log_mixed = ' '.join(['%s=%g' % (k, v) for (k, v) in mixed.items()])
        logging.info('Epoch %d, batch %d: loss=%g %s',
                     checkpoint.state.epoch, b, loss, log_mixed)

      count = 0
      values = {}
      for inputs, labels in cpdata.load_batches(self.data['test']):
        loss_b, (_, mixed) = loss_fn(
            trainable_params, fixed_params,
            inputs, labels, model_state, False, next(rng))
        mixed['loss'] = loss_b
        values = {k: values.get(k, 0) + v for (k, v) in mixed.items()}
        count += 1

      # Compute averages for each logged value.
      values = {k: v/count for (k, v) in values.items()}
      log_mixed = ' '.join(['%s=%g' % (k, v) for (k, v) in values.items()])
      logging.info('Epoch %d, test: %s', checkpoint.state.epoch, log_mixed)

      cputils.update_checkpoint(
          checkpoint, trainable_params, fixed_params,
          model_state, optimizer_state, checkpoint.state.epoch + 1)
      if checkpoint.state.epoch % self.config.checkpoint_frequency == 0:
        checkpoint.save()

    params = hk.data_structures.merge(trainable_params, fixed_params)
    return params, model_state

  def _test_dataset(
      self, params: cputils.FlatMapping, model_state: cputils.FlatMapping,
      dataset: tf.data.Dataset, name: str, epochs: int, shift_fn: ShiftFn):
    """Helper to evaluate model on given dataset.

    Args:
      params: trained parameters of the model
      model_state: model state
      dataset: dataset to evaluate
      name: identifier for dataset
      epochs: number of epochs to run on dataset
      shift_fn: shift function to apply distribution shift to images
    """
    rng = hk.PRNGSequence(0)
    writer = cpstaging.create_writer(self.config, 'eval_%s' % name)
    for epoch in range(epochs):
      logits = []
      labels = []
      for inputs_b, labels_b in cpdata.load_batches(dataset):
        inputs_b, _ = shift_fn(inputs_b, next(rng))
        logits_b, _ = self.model.apply(
            params, model_state, None, inputs_b, training=False)
        logits.append(logits_b)
        labels.append(labels_b)
      logits = jnp.concatenate(logits, axis=0)
      labels = jnp.concatenate(labels, axis=0)
      error = 1 - cpeval.compute_accuracy(
          jax.nn.softmax(logits, axis=1), labels)
      logging.info('Evaluation, %s: %d examples [epoch=%d], error=%g',
                   name, logits.shape[0], epoch, error)
      writer.write({
          'logits': np.array(logits, np.float32),
          'labels': np.array(labels, np.float32),
      })

  def test(self, params: cputils.FlatMapping, model_state: cputils.FlatMapping):
    """Test trained model on training, validation and test sets.

    Args:
      params: trained parameters of the model
      model_state: model state
    """
    no_shift_fn = lambda inputs, rng: (inputs, None)
    num_epochs_per_dataset = {
        'val': (self.data['val'], 1, no_shift_fn),
        'test': (self.data['test'], 1, no_shift_fn),
        'train_clean': (self.data['train_clean'], 1, no_shift_fn),
        # Without data augmentation we might not need to do multiple passes.
        'train_ordered': (self.data['train_ordered'], -1, no_shift_fn),
    }
    for name, (dataset, num_epochs, shift_fn) in num_epochs_per_dataset.items():
      # Check for None in case we train without validation examples.
      if dataset is not None:
        self._test_dataset(
            params, model_state, dataset, name, num_epochs, shift_fn)

  def run(self, rng: hk.PRNGSequence):
    """Main training procedure.

    Args:
      rng: random key sequence
    """
    trainable_params, fixed_params, model_state = self.setup(rng)
    params, model_state = self.train(
        trainable_params, fixed_params, model_state, rng)
    self.test(params, model_state)
