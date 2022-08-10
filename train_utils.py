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

"""Training utilities common across different training schemes."""
from typing import Dict, Any, Tuple, List, Callable, Union

import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections as collections
import optax

import conformal_training.data as cpdata
import conformal_training.models as cpmodels
import conformal_training.open_source_utils as cpstaging


FlatMapping = Union[hk.Params, hk.State]
LossFn = Callable[
    [FlatMapping, FlatMapping, jnp.ndarray,
     jnp.ndarray, FlatMapping, bool, jnp.ndarray],
    Tuple[jnp.ndarray, Tuple[FlatMapping, Dict[str, Any]]]
]


def create_model(
    config: collections.ConfigDict,
    data: Dict[str, Any]) -> hk.TransformedWithState:
  """Helper to get model based on configuration and data.

  Args:
    config: training configuration
    data: data from get_data

  Returns:
    Created model.
  """
  model_config = config[config.architecture]
  whitening = [data['means'], data['stds']] if config.whitening else None
  if config.architecture == 'mlp':
    mlp_units = [model_config.units]*model_config.layers
    model = cpmodels.create_mlp(
        data['classes'], activation=model_config.activation, units=mlp_units,
        whitening=whitening)
  elif config.architecture == 'cnn':
    cnn_channels = [
        model_config.channels*2**i for i in range(model_config.layers)]
    cnn_kernels = [model_config.kernels for _ in range(model_config.layers)]
    model = cpmodels.create_cnn(
        data['classes'], activation=model_config.activation,
        channels=cnn_channels, kernels=cnn_kernels,
        whitening=whitening)
  elif config.architecture == 'resnet':
    logit_w_init = None if model_config.init_logits else jnp.zeros
    model = cpmodels.create_resnet(
        data['classes'], version=model_config.version,
        channels=model_config.channels, resnet_v2=model_config.resnet_v2,
        whitening=whitening, logit_w_init=logit_w_init)
  else:
    raise ValueError('Invalid architecture selected.')

  return model


def load_model(
    config: collections.ConfigDict, data: Dict[str, Any]
) -> Tuple[str, hk.TransformedWithState, FlatMapping, FlatMapping]:
  """Load a model based on the finetune settings in config.

  Args:
    config: training configuration
    data: data from get_data

  Returns:
    Create model, loaded parameters and model state
  """
  checkpoint, path = cpstaging.load_checkpoint(config.finetune)
  model = create_model(config, data)
  params = checkpoint.state.params
  model_state = checkpoint.state.model_state

  return path, model, params, model_state


def init_model(
    data: Dict[str, Any], model: hk.TransformedWithState, rng: hk.PRNGSequence
) -> Tuple[FlatMapping, FlatMapping]:
  """Initialize model and optimizer.

  Args:
    data: data as from get_data
    model: model to initialize
    rng: random key sequence

  Returns:
    Tuple of model parameters and state
  """
  params, model_state = model.init(
      next(rng), next(cpdata.load_batches(data['train']))[0], training=True)
  return params, model_state


def update_checkpoint(
    checkpoint: cpstaging.Checkpoint,
    trainable_params: FlatMapping, fixed_params: FlatMapping,
    model_state: FlatMapping,
    optimizer_state: List[optax.TraceState], epoch: int):
  """Update checkpoint.

  Args:
    checkpoint: checkpoint to update
    trainable_params: model parameters that are being trained
    fixed_params: model parameters that have been fixed
    model_state: model state
    optimizer_state: optimizer state
    epoch: current epoch
  """
  params = hk.data_structures.merge(trainable_params, fixed_params)
  checkpoint.state.params = params
  checkpoint.state.model_state = model_state
  checkpoint.state.optimizer_state = optimizer_state
  checkpoint.state.epoch = epoch


class LRScheduler:
  """Base class of simple scheduler, allowing to track current learning rate."""

  def __init__(
      self, learning_rate: float, learning_rate_decay: float,
      num_examples: int, batch_size: int, epochs: int) -> None:
    """Constructs a learning rate scheduler.

    Args:
      learning_rate: base learning rate to start with
      learning_rate_decay: learning rate decay to be applied
      num_examples: number of examples per epoch
      batch_size: batch size used for training
      epochs: total number of epochs
    """
    self.base_learning_rate = learning_rate
    self.current_learning_rate = learning_rate
    self.learning_rate_decay = learning_rate_decay
    self.batch_size = batch_size
    self.num_examples = num_examples
    self.epochs = epochs

  def __call__(self, step: int) -> float:
    """Applies learning rate schedule to compute current learning rate.

    Args:
      step: training step to compute learning rate for.

    Returns:
      Updated learning rate.
    """
    raise NotImplementedError


class ExponentialLRScheduler(LRScheduler):
  """Exponential learning rate schedule."""

  def __call__(self, step: int) -> float:
    steps_per_epoch = jnp.ceil(self.num_examples / self.batch_size)
    self.current_learning_rate = self.base_learning_rate * (
        self.learning_rate_decay**(step // steps_per_epoch))
    return self.current_learning_rate


class MultIStepLRScheduler(LRScheduler):
  """Multi-step learning rate schedule."""

  def __call__(self, step: int) -> float:
    steps_per_epoch = jnp.ceil(self.num_examples / self.batch_size)
    epoch = step // steps_per_epoch
    epochs_per_step = self.epochs//5
    learning_rate_step = jnp.maximum(epoch//epochs_per_step - 1, 0)
    self.current_learning_rate = self.base_learning_rate * (
        self.learning_rate_decay**learning_rate_step)
    return self.current_learning_rate


def get_sgd_optimizer(
    momentum: float, nesterov: bool,
    lr_scheduler: LRScheduler) -> optax.GradientTransformation:
  """SGD with momentum and lr schedule.

  Args:
    momentum: momentum parameter
    nesterov: whether to use nesterov updates
    lr_scheduler: learning rate schedule to use

  Returns:
    Optimizer
  """
  return optax.chain(
      (optax.trace(decay=momentum, nesterov=nesterov)
       if momentum is not None else optax.identity()),
      optax.scale_by_schedule(lambda step: -lr_scheduler(step))
  )


def get_adam_optimizer(
    b1: float, b2: float, eps: float,
    lr_scheduler: LRScheduler) -> optax.GradientTransformation:
  """SGD with momentum and lr schedule.

  Args:
    b1: decay rate for first moment
    b2: decay rate for second moment
    eps: small constant applied to denominator (see optax docs9
    lr_scheduler: learning rate schedule to use

  Returns:
    Optimizer
  """
  return optax.chain(
      optax.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=0),
      optax.scale_by_schedule(lambda step: -lr_scheduler(step))
  )


def compute_weight_decay(params: FlatMapping) -> float:
  """Weight decay computation.

  Args:
    params: model parameters

  Returns:
    Weight decay
  """
  return sum(
      jnp.sum(jnp.square(param)) for param in jax.tree_leaves(params))


def compute_cross_entropy_loss(
    logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
  """Compute cross entropy loss.

  Args:
    logits: logits predicted by model
    labels: ground truth labels

  Returns:
    Mean cross entropy loss
  """
  one_hot_labels = jax.nn.one_hot(labels, logits.shape[1])
  return jnp.mean(optax.softmax_cross_entropy(
      logits, one_hot_labels))


def compute_hinge_size_loss(
    confidence_sets: jnp.ndarray, target_size: int,
    transform: Callable[[jnp.ndarray], jnp.ndarray],
    weights: jnp.ndarray) -> jnp.ndarray:
  """Compute hinge size loss.

  Args:
    confidence_sets: predicted confidence sets
    target_size: target confidence set size
    transform: transform to apply on per example computed size
    weights: per-example weights to apply

  Returns:
    Size loss
  """
  return jnp.mean(transform(
      weights * jnp.maximum(jnp.sum(confidence_sets, axis=1) - target_size, 0)))


def compute_hinge_bounded_size_loss(
    confidence_sets: jnp.ndarray, target_size: int,
    bound_size: int, bound_weight: float,
    transform: Callable[[jnp.ndarray], jnp.ndarray],
    weights: jnp.ndarray) -> jnp.ndarray:
  """Compute bounded hinge loss.

  Compared to compute_hinge_size_loss, this loss enforces a higher loss
  when size exceeds bound_size.

  Args:
    confidence_sets: predicted confidence sets
    target_size: target confidence set size
    bound_size: confidence set size at which bound loss starts
    bound_weight: weight of bound loss in (0, 1)
    transform: transform to apply on per example computed size
    weights: per-example weights to apply

  Returns:
    Bounded size loss
  """
  sizes = jnp.sum(confidence_sets, axis=1)
  target_loss = jnp.maximum(sizes - target_size, 0)
  bound_loss = jnp.maximum(sizes - bound_size, 0)
  size_loss = jnp.mean(transform(
      weights * ((1 - bound_weight) * target_loss + bound_weight * bound_loss)))
  return size_loss


def compute_probabilistic_size_loss(
    confidence_sets: jnp.ndarray,
    weights: jnp.ndarray) -> jnp.ndarray:
  """Compute probabilistic size loss.

  This size loss is motivated by interpreting the confidence set predictions
  as Bernoulli probabilities of a specific label being part of it.
  The sum of these Bernoulli variables is distributed according to a
  Poisson binomial distribution. This loss is the negative likelihood
  of this distribution for a size of 1.

  Args:
    confidence_sets: predicted sets
    weights: per-example weights to apply

  Returns:
    Size loss
  """

  classes = confidence_sets.shape[1]
  one_hot_labels = jnp.expand_dims(jnp.identity(classes), axis=0)
  repeated_confidence_sets = jnp.repeat(
      jnp.expand_dims(confidence_sets, axis=2), classes, axis=2)
  loss = one_hot_labels * repeated_confidence_sets + (
      1 - one_hot_labels) * (1 - repeated_confidence_sets)
  loss = jnp.prod(loss, axis=1)
  loss = jnp.sum(loss, axis=1)
  return jnp.mean(weights * loss)


def compute_coverage_loss(
    confidence_sets: jnp.ndarray,
    labels: jnp.ndarray, alpha: float,
    transform: Callable[[jnp.ndarray], jnp.ndarray] = jnp.square
) -> jnp.ndarray:
  """Compute squared coverage loss.

  Computes empirical coverage on batch and a squared loss between empirical
  coverage and target coverage defined as 1 - alpha.

  Args:
    confidence_sets: predicted confidence sets
    labels: ground truth labels
    alpha: confidence level
    transform: transform to apply on error, e.g., square

  Returns:
    Squared coverage loss
  """
  one_hot_labels = jax.nn.one_hot(labels, confidence_sets.shape[1])
  return transform(jnp.mean(jnp.sum(
      confidence_sets * one_hot_labels, axis=1)) - (1 - alpha))


def compute_general_classification_loss(
    confidence_sets: jnp.ndarray, labels: jnp.ndarray,
    loss_matrix: jnp.ndarray) -> jnp.ndarray:
  """Compute general classification loss on confidence sets.

  Besides enforcing that the true label is contained in the confidence set,
  this loss also penalizes any other label in the set according to the
  loss_matrix.

  Args:
    confidence_sets: predicted confidence sets
    labels: ground truth labels
    loss_matrix: symmetric loss matrix

  Returns:
    Classification loss
  """
  one_hot_labels = jax.nn.one_hot(labels, confidence_sets.shape[1])
  l1 = (1 - confidence_sets) * one_hot_labels * loss_matrix[labels]
  l2 = confidence_sets * (1 - one_hot_labels) * loss_matrix[labels]
  loss = jnp.sum(jnp.maximum(l1 + l2, jnp.zeros_like(l1)), axis=1)
  return jnp.mean(loss)


def compute_general_binary_cross_entropy_loss(
    confidence_sets: jnp.ndarray, labels: jnp.ndarray,
    loss_matrix: jnp.ndarray) -> jnp.ndarray:
  """Compute general binary cross-entropy loss.

  Args:
    confidence_sets: predicted confidence sets
    labels: ground truth labels
    loss_matrix: symmetric loss matrix

  Returns:
    Binary cross-entropy loss
  """
  one_hot_labels = jax.nn.one_hot(labels, confidence_sets.shape[1])
  l1 = loss_matrix[labels] * one_hot_labels * jnp.log(confidence_sets + 1e-8)
  l2 = loss_matrix[labels] * (1 - one_hot_labels) * jnp.log(
      1 - confidence_sets + 1e-8)
  loss = jnp.sum(jnp.minimum(l1 + l2, jnp.zeros_like(l1)), axis=1)
  return jnp.mean(- loss)


def update(
    trainable_params: FlatMapping, fixed_params: FlatMapping,
    inputs: jnp.ndarray, labels: jnp.ndarray,
    model_state: FlatMapping, training: bool,
    optimizer_state: List[optax.TraceState],
    rng: jnp.ndarray,
    loss_fn: LossFn,
    optimizer: optax.GradientTransformation
) -> Tuple[jnp.ndarray, FlatMapping, FlatMapping,
           List[optax.TraceState], Dict[str, Any]]:
  """Update parameters using the given loss function.

  The loss function is supposed to return the loss, followed by a tuple
  consisting of the new model state and a dictionary that may contain additional
  information or can be empty.

  Args:
    trainable_params: model parameters to update
    fixed_params: model parameters not to update, i.e., fixed
    inputs: input examples
    labels: ground truth examples
    model_state: model state
    training: training mode
    optimizer_state: optimizer state
    rng: random key
    loss_fn: loss function to use
    optimizer: optax optimizer

  Returns:
    Tuple consisting of loss, new parameters, new model state, new optimizer
    state and a dictionary with additional information from the loss function
  """
  (loss, (new_model_state, mixed)), grad = jax.value_and_grad(
      loss_fn, has_aux=True)(trainable_params, fixed_params, inputs, labels,
                             model_state, training, rng)
  updates, new_optimizer_state = optimizer.update(grad, optimizer_state)
  new_params = optax.apply_updates(trainable_params, updates)
  return loss, new_params, new_model_state, new_optimizer_state, mixed
