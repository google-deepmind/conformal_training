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

"""Tests for training utilities."""
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

import train_utils as cputils


class TrainUtilsTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(learning_rate_decay=0.1, num_examples=50000, batch_size=128,
           epochs=150),
      dict(learning_rate_decay=0.1, num_examples=50000, batch_size=64,
           epochs=150),
      dict(learning_rate_decay=0.1, num_examples=10000, batch_size=128,
           epochs=150),
      dict(learning_rate_decay=0.1, num_examples=50000, batch_size=128,
           epochs=250),
  ])
  def test_multi_step_lr_scheduler(
      self, learning_rate_decay, num_examples, batch_size, epochs):
    learning_rate = 0.1
    lr_scheduler = cputils.MultIStepLRScheduler(
        learning_rate, learning_rate_decay, num_examples, batch_size, epochs)

    # Test final and initial learning rate.
    first_step = 0
    self.assertAlmostEqual(
        lr_scheduler(first_step), learning_rate)
    final_step = num_examples*epochs//batch_size
    self.assertAlmostEqual(
        lr_scheduler(final_step), learning_rate * learning_rate_decay**3)

    # Check each learning rate drop individually.
    steps_per_epoch = np.ceil(num_examples/batch_size)
    first_drop_epoch = epochs // 5 * 2
    first_drop_step = first_drop_epoch * steps_per_epoch
    self.assertAlmostEqual(lr_scheduler(first_drop_step - 1), learning_rate)
    self.assertAlmostEqual(
        lr_scheduler(first_drop_step), learning_rate * learning_rate_decay)
    second_drop_epoch = epochs // 5 * 3
    second_drop_step = second_drop_epoch * steps_per_epoch
    self.assertAlmostEqual(
        lr_scheduler(second_drop_step - 1), learning_rate * learning_rate_decay)
    self.assertAlmostEqual(
        lr_scheduler(second_drop_step), learning_rate * learning_rate_decay**2)
    third_drop_epoch = epochs // 5 * 4
    third_drop_step = third_drop_epoch * steps_per_epoch
    self.assertAlmostEqual(
        lr_scheduler(third_drop_step - 1),
        learning_rate * learning_rate_decay**2)
    self.assertAlmostEqual(
        lr_scheduler(third_drop_step), learning_rate * learning_rate_decay**3)

  def test_compute_general_classification_loss(self):
    confidence_sets = jnp.zeros((100, 10))
    loss_matrix = jnp.eye(10)
    labels = jnp.zeros(100).astype(int)

    loss = cputils.compute_general_classification_loss(
        confidence_sets, labels, loss_matrix)
    self.assertAlmostEqual(loss, 1.)

    confidence_sets = confidence_sets.at[:, 0].set(1)
    loss = cputils.compute_general_classification_loss(
        confidence_sets, labels, loss_matrix)
    self.assertAlmostEqual(loss, 0.)

    confidence_sets = confidence_sets.at[:, 1].set(1)
    loss = cputils.compute_general_classification_loss(
        confidence_sets, labels, loss_matrix)
    self.assertAlmostEqual(loss, 0.)

    loss_matrix = jnp.ones((10, 10))
    loss = cputils.compute_general_classification_loss(
        confidence_sets, labels, loss_matrix)
    self.assertAlmostEqual(loss, 1.)

    confidence_sets = confidence_sets.at[:, 1].set(0)
    loss = cputils.compute_general_classification_loss(
        confidence_sets, labels, loss_matrix)
    self.assertAlmostEqual(loss, 0.)

  def test_compute_general_binary_cross_entropy_loss(self):
    confidence_sets = jnp.zeros((100, 10))
    loss_matrix = jnp.eye(10)
    labels = jnp.zeros(100).astype(int)

    loss = cputils.compute_general_binary_cross_entropy_loss(
        confidence_sets, labels, loss_matrix)
    self.assertAlmostEqual(loss, - jnp.log(1e-8))

    confidence_sets = confidence_sets.at[:, 0].set(1)
    loss = cputils.compute_general_binary_cross_entropy_loss(
        confidence_sets, labels, loss_matrix)
    self.assertAlmostEqual(loss, 0.)

    confidence_sets = confidence_sets.at[:, 1].set(1)
    loss = cputils.compute_general_binary_cross_entropy_loss(
        confidence_sets, labels, loss_matrix)
    self.assertAlmostEqual(loss, 0.)

    loss_matrix = jnp.ones((10, 10))
    loss = cputils.compute_general_binary_cross_entropy_loss(
        confidence_sets, labels, loss_matrix)
    self.assertAlmostEqual(loss, - jnp.log(1e-8), places=3)

    confidence_sets = confidence_sets.at[:, 1].set(0)
    loss = cputils.compute_general_binary_cross_entropy_loss(
        confidence_sets, labels, loss_matrix)
    self.assertAlmostEqual(loss, 0.)

  @parameterized.parameters([
      dict(num_classes=5, target_size=0),
      dict(num_classes=5, target_size=1),
      dict(num_classes=5, target_size=5),
  ])
  def test_compute_hinge_size_loss(self, num_classes, target_size):
    for k in range(num_classes):
      confidence_sets = np.zeros((1, num_classes))
      confidence_sets[:, :k] = 1
      self.assertEqual(np.sum(confidence_sets), k)
      size_loss = cputils.compute_hinge_size_loss(
          jnp.array(confidence_sets), target_size=target_size,
          transform=lambda x: x, weights=jnp.ones(confidence_sets.shape[0]))
      expected_loss = max(k - target_size, 0)
      self.assertAlmostEqual(size_loss, expected_loss)
      size_loss = cputils.compute_hinge_size_loss(
          jnp.array(confidence_sets), target_size=target_size,
          transform=jnp.log, weights=jnp.ones(confidence_sets.shape[0]))
      self.assertAlmostEqual(size_loss, np.log(expected_loss), places=3)

  @parameterized.parameters([
      dict(num_classes=5, target_size=0, bound_size=0, bound_weight=0.5),
      dict(num_classes=5, target_size=1, bound_size=3, bound_weight=0.5),
      dict(num_classes=5, target_size=1, bound_size=3, bound_weight=0.99),
      dict(num_classes=5, target_size=5, bound_size=7, bound_weight=0.5),
  ])
  def test_compute_hinge_bounded_size_loss(
      self, num_classes, target_size, bound_size, bound_weight):
    for k in range(num_classes):
      confidence_sets = np.zeros((1, num_classes))
      confidence_sets[:, :k] = 1
      self.assertEqual(np.sum(confidence_sets), k)
      size_loss = cputils.compute_hinge_bounded_size_loss(
          jnp.array(confidence_sets), target_size=target_size,
          bound_size=bound_size, bound_weight=bound_weight,
          transform=lambda x: x, weights=jnp.ones(confidence_sets.shape[0]))
      expected_loss = (1 - bound_weight) * max(k - target_size, 0)
      expected_loss += bound_weight * max(k - bound_size, 0)
      self.assertAlmostEqual(size_loss, expected_loss)
      size_loss = cputils.compute_hinge_bounded_size_loss(
          jnp.array(confidence_sets), target_size=target_size,
          bound_size=bound_size, bound_weight=bound_weight,
          transform=jnp.log, weights=jnp.ones(confidence_sets.shape[0]))
      self.assertAlmostEqual(size_loss, np.log(expected_loss), places=3)


if __name__ == '__main__':
  absltest.main()
