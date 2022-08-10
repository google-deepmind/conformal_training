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

"""Various simple architectures."""
from typing import List, Optional, Tuple, Sequence, Mapping, Any
import haiku as hk
import jax
from jax import numpy as jnp


def _check_create(
    classes: int = 10, activation: Optional[str] = None,
    whitening: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None):
  """Helper to check arguments for creating models.

  Args:
    classes: number of output classes
    activation: activation function to use or None
    whitening: None or tuple of means and stds to use for whitening

  Raises:
    ValueError: invalid arguments for architecture creation
  """
  if classes < 1:
    raise ValueError('Expecting at least 1 class.')
  if activation is not None:
    if activation not in ['relu', 'tanh']:
      raise ValueError('Unsupported activation.')
  if whitening is not None:
    if len(whitening) != 2:
      raise ValueError(
          'Expecting whitening to be tuple containing means and std.')


def _apply_whitening(
    inputs: jnp.ndarray,
    whitening: Optional[Tuple[jnp.ndarray, jnp.ndarray]]) -> jnp.ndarray:
  """Apply data whitening.

  Args:
    inputs: inputs
    whitening: mean and std for whitening as tuple

  Returns:
    Whitened inputs
  """
  if whitening is not None:
    inputs = (inputs - whitening[0].reshape((1, 1, 1, -1)))
    inputs = inputs / whitening[1].reshape((1, 1, 1, -1))
  return inputs


def create_mlp(
    classes: int = 10,
    activation: str = 'relu',
    units: Optional[List[int]] = None,
    whitening: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
)-> hk.TransformedWithState:
  """Simple MLP architecture.

  Create an MLP with the given output classes and hidden layers.
  Defaults to a linear model.

  Args:
    classes: number of output classes
    activation: activation function to use
    units: list of hidden units per hidden layer
    whitening: None or tuple of means and stds to use for whitening

  Returns:
    Created jax function representing the MLP.

  Raises:
    ValueError: invalid architecture arguments
  """
  if units is None:
    units = []
  _check_create(classes, activation=activation, whitening=whitening)

  def forward(inputs, training):
    inputs = _apply_whitening(inputs, whitening)
    inputs = jnp.reshape(inputs, [inputs.shape[0], -1])
    for unit in units:
      inputs = hk.Linear(unit)(inputs)
      inputs = hk.BatchNorm(True, True, 0.9)(inputs, training)
      inputs = getattr(jax.nn, activation)(inputs)
    inputs = hk.Linear(classes)(inputs)
    return inputs

  return hk.transform_with_state(forward)


def create_cnn(
    classes: int = 10, activation: str = 'relu',
    channels: Optional[List[int]] = None,
    kernels: Optional[List[int]] = None,
    whitening: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
) -> hk.TransformedWithState:
  """Simple CNN architecture.

  Create a CNN with several convolutional layers, followed by
  batch normalization, ReLU and max pooling and a final fully connected layer.

  Args:
    classes: number of output classes
    activation: activation function to use
    channels: convolutional channels of each convolutional stage
    kernels: kernel size for each convolutional layer
    whitening: None or tuple of means and stds to use for whitening

  Returns:
    Created jax function representing the CNN

  Raises:
    ValueError: invalid architecture arguments
  """
  if channels is None:
    channels = [32, 64, 128]
  if kernels is None:
    kernels = [3, 3, 3]
  if not channels:
    raise ValueError('Expecting at least on convolutional channels.')
  if len(channels) != len(kernels):
    raise ValueError('Expecting same number of channels and kernels.')
  _check_create(classes, activation=activation, whitening=whitening)

  def forward(inputs, training):
    inputs = _apply_whitening(inputs, whitening)
    for l in range(len(channels)):
      c = channels[l]
      k = kernels[l]
      inputs = hk.Conv2D(output_channels=c, kernel_shape=[k, k])(inputs)
      inputs = hk.BatchNorm(True, True, 0.9)(inputs, training)
      inputs = getattr(jax.nn, activation)(inputs)
      # window_shape and strides needs to be tuple to avoid deprecated warning.
      inputs = hk.MaxPool(
          window_shape=(2, 2, 1), strides=(2, 2, 1), padding='SAME')(inputs)

    inputs = jnp.reshape(inputs, [inputs.shape[0], -1])
    inputs = hk.Linear(classes)(inputs)
    return inputs

  # transform_with_state necessary because of batch norm.
  return hk.transform_with_state(forward)


class ResNet(hk.nets.ResNet):
  """Overwrite Haiku's ResNet model for Cifar10."""

  def __init__(
      self,
      blocks_per_group: Sequence[int],
      num_classes: int,
      bn_config: Optional[Mapping[str, float]] = None,
      resnet_v2: bool = False,
      bottleneck: bool = True,
      channels_per_group: Sequence[int] = (256, 512, 1024, 2048),
      use_projection: Sequence[bool] = (True, True, True, True),
      logits_config: Optional[Mapping[str, Any]] = None,
      name: Optional[str] = None,
  ):
    """Constructs a ResNet model.

    In contrast to Haiku's original implementation, the first convolutional
    layer uses 3x3 convolutional kernel with stride 1.

    Args:
      blocks_per_group: A sequence of length 4 that indicates the number of
        blocks created in each group.
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers. By default the
        ``decay_rate`` is ``0.9`` and ``eps`` is ``1e-5``.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
        ``False``.
      bottleneck: Whether the block should bottleneck or not. Defaults to
        ``True``.
      channels_per_group: A sequence of length 4 that indicates the number
        of channels used for each block in each group.
      use_projection: A sequence of length 4 that indicates whether each
        residual block should use projection.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
    """
    super(ResNet, self).__init__(
        blocks_per_group, num_classes, bn_config, resnet_v2, bottleneck,
        channels_per_group, use_projection, logits_config, name)

    self.initial_conv = hk.Conv2D(
        output_channels=64, kernel_shape=3, stride=1,
        with_bias=False, padding='SAME', name='initial_conv')


def _check_create_resnet(version: int, channels: int):
  """Helper to check arguments for creating resnets.

  Args:
    version: resnet version
    channels: resnet channels to start with

  Raises:
    ValueError: invalid arguments for architecture creation
  """
  if version not in [18, 34, 50, 101, 152, 200]:
    raise ValueError('Only ResNet-[18, 34, 50, 101, 152, 200] supported.')
  if channels < 1:
    raise ValueError('Expecting at least one channel to start with.')


def create_resnet(
    classes: int = 10,
    version: Optional[int] = 18,
    channels: Optional[int] = None,
    resnet_v2: Optional[bool] = False,
    whitening: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    logit_w_init: Optional[hk.initializers.Initializer] = jnp.zeros
) -> hk.TransformedWithState:
  """Simple wrapper for Haiku's ResNet implementation.

  Creats a ResNet-version with the given channels in the first block
  and whitening if desired. See Haiku doc for details on structure and
  resnet_v2.

  Args:
    classes: number of output classes
    version: version, i.e., depth of ResNet
    channels: number of channels in first block
    resnet_v2: whether to use ResNet v2
    whitening: None or tuple of means and stds to use for whitening
    logit_w_init: logit weights initializer

  Returns:
    Created jax function representing the ResNet

  Raises:
    ValueError: invalid architecture arguments
  """
  if version not in [18, 34, 50, 101, 152, 200]:
    raise ValueError('Only ResNet-[18, 34, 50, 101, 152, 200] supported.')
  if channels < 1:  # pytype: disable=unsupported-operands
    raise ValueError('Expecting at least one channel to start with.')
  _check_create_resnet(version, channels)
  _check_create(classes, activation=None, whitening=whitening)

  resnet_config = ResNet.CONFIGS[version]
  # channels defines the number of channels for first block; the remaining
  # blocks' channels are derived by doubling.
  resnet_config['channels_per_group'] = tuple([
      channels*2**i for i in range(len(resnet_config['blocks_per_group']))
  ])
  # The very first convolutional in Haiku ResNets is hard-coded to 64.
  # So if channels is not 64, we need to add a projection.
  if channels != 64:
    resnet_config['use_projection'] = tuple([True]*4)

  def forward(inputs, training):
    inputs = _apply_whitening(inputs, whitening)
    net = ResNet(
        num_classes=classes, resnet_v2=resnet_v2,
        logits_config={'w_init': logit_w_init}, **resnet_config)
    return net(inputs, training)

  return hk.transform_with_state(forward)
