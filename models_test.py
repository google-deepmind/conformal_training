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

"""Tests for models."""
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np

import conformal_training.models as cpmodels


class ModelsTest(parameterized.TestCase):

  def _test_model(self, classes, model, jit=False):
    def forward(params, model_state, rng, inputs, training):
      outputs, _ = model.apply(params, model_state, rng, inputs, training)
      return outputs
    if jit:
      forward = jax.jit(forward, static_argnums=4)

    batch_size = 10
    inputs = np.random.rand(batch_size, 32, 32, 3).astype(jnp.float32)
    rng = jax.random.PRNGKey(0)
    params, model_state = model.init(rng, inputs, training=True)

    outputs = forward(params, model_state, rng, inputs, training=True)
    self.assertEqual(outputs.shape, (batch_size, classes))
    outputs = forward(params, model_state, rng, inputs, training=False)
    self.assertEqual(outputs.shape, (batch_size, classes))

  @parameterized.parameters([
      dict(classes=10, activation='relu', units=[16, 16]),
      dict(classes=10, activation='relu', units=[16, 16], jit=True),
      dict(classes=100, activation='relu', units=[16, 16]),
      dict(classes=10, activation='tanh', units=[16, 16]),
      dict(classes=10, activation='relu', units=[16]),
  ])
  def test_mlp_classes(self, classes, activation, units, jit=False):
    model = cpmodels.create_mlp(classes, activation=activation, units=units)
    self._test_model(classes, model, jit=jit)

  @parameterized.parameters([
      dict(classes=10, activation='a', units=[128]),
      dict(classes=0, activation='relu', units=[128]),
  ])
  def test_mlp_errors(self, classes, activation, units):
    with self.assertRaises(ValueError):
      cpmodels.create_mlp(classes, activation, units)

  @parameterized.parameters([
      dict(classes=10, activation='relu',
           channels=[8, 16, 32], kernels=[3, 3, 3]),
      dict(classes=10, activation='relu',
           channels=[8, 16, 32], kernels=[3, 3, 3], jit=True),
      dict(classes=100, activation='relu',
           channels=[8, 16, 32], kernels=[3, 3, 3]),
      dict(classes=10, activation='tanh',
           channels=[8, 16, 32], kernels=[3, 3, 3]),
      dict(classes=10, activation='relu',
           channels=[8, 16], kernels=[3, 3]),
      dict(classes=10, activation='relu',
           channels=[8, 16, 32], kernels=[5, 5, 5]),
  ])
  def test_cnn_classes(self, classes, activation, channels, kernels, jit=False):
    model = cpmodels.create_cnn(
        classes, activation=activation, channels=channels, kernels=kernels)
    self._test_model(classes, model, jit=jit)

  @parameterized.parameters([
      dict(classes=10, activation='relu', channels=[], kernels=[3, 3, 3]),
      dict(classes=10, activation='relu', channels=[64, 128, 256], kernels=[]),
      dict(classes=10, activation='a', channels=[64, 128], kernels=[3, 3]),
      dict(classes=10, activation='relu', channels=[64, 128],
           kernels=[3, 3, 3]),
      dict(classes=0, activation='relu', channels=[64, 128, 256],
           kernels=[3, 3, 3]),
  ])
  def test_cnn_errors(self, classes, activation, channels, kernels):
    with self.assertRaises(ValueError):
      cpmodels.create_cnn(classes, activation, channels, kernels)

  @parameterized.parameters([
      dict(classes=10, version=18),
      dict(classes=10, version=18, jit=True),
      dict(classes=100, version=18),
      dict(classes=10, version=18, channels=32),
  ])
  def test_resnet_classes(
      self, classes, version, resnet_v2=False, channels=64, jit=True):
    model = cpmodels.create_resnet(
        classes, version, channels, resnet_v2)
    self._test_model(classes, model, jit=jit)

  @parameterized.parameters([
      dict(classes=10, version=17, channels=64),
      dict(classes=10, version=18, channels=0),
      dict(classes=0, version=18, channels=64),
  ])
  def test_resnet_errors(self, classes, version, channels):
    with self.assertRaises(ValueError):
      cpmodels.create_resnet(classes, version, channels)

  def test_resnet_initial_conv_params(self):
    batch_size = 100
    model = cpmodels.create_resnet(10, 18, 64, False)
    inputs = np.random.rand(batch_size, 32, 32, 3).astype(jnp.float32)
    params, _ = model.init(
        jax.random.PRNGKey(0), inputs, training=True)
    chex.assert_shape(params['res_net/~/initial_conv_1']['w'], (3, 3, 3, 64))


if __name__ == '__main__':
  absltest.main()
