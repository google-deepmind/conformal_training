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

"""Tests for evaluation metrics."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

import evaluation as cpeval
import test_utils as cptutils


class EvaluationTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(num_examples=100000, num_classes=10),
      dict(num_examples=100000, num_classes=100),
      dict(num_examples=1000000, num_classes=100)
  ])
  def test_compute_accuracy_random(self, num_examples, num_classes):
    labels = cptutils.get_labels(num_examples, num_classes)
    probabilities = cptutils.get_probabilities(labels, 0)
    accuracy = cpeval.compute_accuracy_with_checks(probabilities, labels)
    expected_accuracy = 1./num_classes
    self.assertGreaterEqual(accuracy, expected_accuracy - 2e-2)
    self.assertGreaterEqual(expected_accuracy + 2e-2, accuracy)

  @parameterized.parameters([
      dict(num_examples=1000000, num_classes=10, num_selected=0),
      dict(num_examples=1000000, num_classes=10, num_selected=100000),
      dict(num_examples=1000000, num_classes=10, num_selected=500000),
  ])
  def test_compute_conditional_accuracy_random(
      self, num_examples, num_classes, num_selected):
    labels = cptutils.get_labels(num_examples, num_classes)
    probabilities = cptutils.get_probabilities(labels, 0)
    conditional_labels = jnp.zeros(labels.shape).astype(int)
    conditional_labels = conditional_labels.at[:num_selected].set(1)
    accuracy = cpeval.compute_conditional_accuracy_with_checks(
        probabilities, labels, conditional_labels, 1)
    expected_accuracy = (1./num_classes) if num_selected > 0 else 0
    self.assertGreaterEqual(accuracy, expected_accuracy - 2e-2)
    self.assertGreaterEqual(expected_accuracy + 2e-2, accuracy)

  @parameterized.parameters([
      dict(num_examples=10000, num_classes=10),
      dict(num_examples=10000, num_classes=100),
  ])
  def test_compute_accuracy_correct(self, num_examples, num_classes):
    labels = cptutils.get_labels(num_examples, num_classes)
    probabilities = cptutils.get_probabilities(labels, 1)
    accuracy = cpeval.compute_accuracy_with_checks(probabilities, labels)
    self.assertAlmostEqual(accuracy, 1)

  @parameterized.parameters([
      dict(probabilities=np.array([]), labels=np.ones((100))),
      dict(probabilities=np.zeros((100, 10)), labels=np.array([])),
      dict(probabilities=np.zeros((100, 10)),
           labels=np.ones((100)) * 99),
      dict(probabilities=np.zeros((100)),
           labels=np.ones((100))),
      dict(probabilities=np.zeros((100, 10)),
           labels=np.ones((100, 10))),
      dict(probabilities=np.zeros((100, 10)),
           labels=np.ones((99))),
      dict(probabilities=np.zeros((100, 10)),
           labels=np.ones((99))),
  ])
  def test_compute_accuracy_errors(self, probabilities, labels):
    with self.assertRaises(ValueError):
      cpeval.compute_accuracy_with_checks(
          jnp.array(probabilities), jnp.array(labels))

  @parameterized.parameters([
      dict(probabilities=np.zeros((100, 10)),
           labels=np.ones((100)), conditional_labels=np.array([])),
      dict(probabilities=np.zeros((100, 10)),
           labels=np.ones((100)), conditional_labels=np.ones((99))),
      dict(probabilities=np.zeros((100, 10)),
           labels=np.ones((100)), conditional_labels=np.ones((100, 10))),
  ])
  def test_compute_conditional_accuracy_errors(
      self, probabilities, labels, conditional_labels):
    with self.assertRaises(ValueError):
      cpeval.compute_conditional_accuracy_with_checks(
          jnp.array(probabilities), jnp.array(labels),
          jnp.array(conditional_labels), 1)

  @parameterized.parameters([
      dict(probabilities=np.zeros((100, 10)),
           labels=np.ones((100)), conditional_labels=np.ones((100))),
  ])
  def test_compute_accuracy_jit(
      self, probabilities, labels, conditional_labels):
    compute_conditional_accuracy_fn = jax.jit(
        cpeval.compute_conditional_accuracy)
    compute_conditional_accuracy_fn(
        jnp.array(probabilities), jnp.array(labels),
        jnp.array(conditional_labels), 1)

  @parameterized.parameters([
      dict(num_examples=10000, num_classes=10),
      dict(num_examples=10000, num_classes=100),
      dict(num_examples=100000, num_classes=100),
  ])
  def test_compute_coverage_simple(self, num_examples, num_classes):
    labels = cptutils.get_labels(num_examples, num_classes)

    # Case: all zeros.
    confidence_sets = jnp.zeros((num_examples, num_classes))
    coverage = cpeval.compute_coverage_with_checks(confidence_sets, labels)
    self.assertAlmostEqual(coverage, 0)

    # Case: all ones.
    confidence_sets = jnp.ones((num_examples, num_classes))
    coverage = cpeval.compute_coverage_with_checks(confidence_sets, labels)
    self.assertAlmostEqual(coverage, 1)

    # Case: one hot of true class.
    confidence_sets = jnp.zeros((num_examples, num_classes))
    confidence_sets = confidence_sets.at[
        (jnp.arange(confidence_sets.shape[0]), labels)].set(1)
    self.assertAlmostEqual(coverage, 1)

  @parameterized.parameters([
      dict(num_examples=500000, num_classes=10),
      dict(num_examples=5000000, num_classes=100),
  ])
  def test_compute_coverage_random(self, num_examples, num_classes):
    labels = cptutils.get_labels(num_examples, num_classes)

    # First case, only true label or zeros.
    confidence_sets = jnp.zeros((num_examples, num_classes))
    rand = jnp.array(np.random.random((num_examples)))
    confidence_sets = confidence_sets.at[
        (jnp.arange(confidence_sets.shape[0]), labels)].set(
            (rand <= 0.5).astype(int))
    coverage = cpeval.compute_coverage_with_checks(confidence_sets, labels)
    self.assertAlmostEqual(coverage, 0.5, places=1)

    # First case, everything one except true label for some rows.
    confidence_sets = jnp.ones((num_examples, num_classes))
    confidence_sets = confidence_sets.at[
        (jnp.arange(confidence_sets.shape[0]), labels)].set(
            (rand <= 0.5).astype(int))
    coverage = cpeval.compute_coverage_with_checks(confidence_sets, labels)
    self.assertAlmostEqual(coverage, 0.5, places=1)

  @parameterized.parameters([
      dict(num_examples=5000000, num_classes=10, num_selected=0),
      dict(num_examples=5000000, num_classes=10, num_selected=500000),
  ])
  def test_compute_conditional_coverage_random(
      self, num_examples, num_classes, num_selected):
    confidence_sets = jnp.zeros((num_examples, num_classes))
    labels = cptutils.get_labels(num_examples, num_classes)
    conditional_labels = jnp.zeros(labels.shape).astype(int)
    conditional_labels = conditional_labels.at[:num_selected].set(1)
    rand = jnp.array(np.random.random((num_examples)))
    confidence_sets = confidence_sets.at[
        (jnp.arange(confidence_sets.shape[0]), labels)].set(
            (rand <= 0.5).astype(int))
    coverage = cpeval.compute_conditional_coverage_with_checks(
        confidence_sets, labels, conditional_labels, 1)
    expected_coverage = 0.5 if num_selected > 0 else 1
    self.assertAlmostEqual(coverage, expected_coverage, places=1)

  @parameterized.parameters([
      dict(confidence_sets=np.array([]), labels=np.ones((100))),
      dict(confidence_sets=np.zeros((100, 10)), labels=np.array([])),
      dict(confidence_sets=np.zeros((100, 10)),
           labels=np.ones((100)) * 99),
      dict(confidence_sets=np.zeros((100)),
           labels=np.ones((100))),
      dict(confidence_sets=np.zeros((100, 10)),
           labels=np.ones((100, 10))),
      dict(confidence_sets=np.zeros((100, 10)),
           labels=np.ones((99))),
  ])
  def test_compute_coverage_errors(self, confidence_sets, labels):
    with self.assertRaises(ValueError):
      cpeval.compute_coverage_with_checks(
          jnp.array(confidence_sets), jnp.array(labels))

  @parameterized.parameters([
      dict(confidence_sets=np.zeros((100, 10)),
           labels=np.ones((100)), conditional_labels=np.array([])),
      dict(confidence_sets=np.zeros((100, 10)),
           labels=np.ones((100)), conditional_labels=np.ones((99))),
      dict(confidence_sets=np.zeros((100, 10)),
           labels=np.ones((100)), conditional_labels=np.ones((100, 10))),
  ])
  def test_compute_conditional_coverage_errors(
      self, confidence_sets, labels, conditional_labels):
    with self.assertRaises(ValueError):
      cpeval.compute_conditional_coverage_with_checks(
          jnp.array(confidence_sets), jnp.array(labels),
          jnp.array(conditional_labels), 1)

  @parameterized.parameters([
      dict(confidence_sets=np.zeros((100, 10)),
           labels=np.ones((100)), conditional_labels=np.ones((100))),
  ])
  def test_compute_conditional_coverage_jit(
      self, confidence_sets, labels, conditional_labels):
    compute_conditional_coverage_fn = jax.jit(
        cpeval.compute_conditional_coverage)
    compute_conditional_coverage_fn(
        jnp.array(confidence_sets), jnp.array(labels),
        jnp.array(conditional_labels), 1)

  @parameterized.parameters([
      dict(num_examples=100000, num_classes=10, fraction=0.1),
      dict(num_examples=100000, num_classes=10, fraction=0.5),
  ])
  def test_compute_size(self, num_examples, num_classes, fraction):
    confidence_sets = np.random.random((num_examples, num_classes))
    confidence_sets = jnp.array(confidence_sets <= fraction).astype(int)
    size, count = cpeval.compute_size_with_checks(confidence_sets)
    expected_size = num_classes * fraction
    self.assertEqual(count, num_examples)
    self.assertAlmostEqual(size, expected_size, places=1)

  @parameterized.parameters([
      dict(num_examples=100000, num_classes=10,
           fraction=0.1, num_selected=0),
      dict(num_examples=100000, num_classes=10,
           fraction=0.5, num_selected=50000),
  ])
  def test_compute_conditional_size(
      self, num_examples, num_classes, fraction, num_selected):
    confidence_sets = np.random.random((num_examples, num_classes))
    confidence_sets = jnp.array(confidence_sets <= fraction).astype(int)
    conditional_labels = jnp.zeros(confidence_sets.shape[0]).astype(int)
    conditional_labels = conditional_labels.at[:num_selected].set(1)
    size, count = cpeval.compute_conditional_size_with_checks(
        confidence_sets, conditional_labels, 1)
    expected_size = (num_classes * fraction) if num_selected > 0 else 0
    self.assertEqual(count, num_selected)
    self.assertAlmostEqual(size, expected_size, places=1)

  @parameterized.parameters([
      dict(confidence_sets=np.array([])),
      dict(confidence_sets=np.zeros((100))),
  ])
  def test_compute_size_errors(self, confidence_sets):
    with self.assertRaises(ValueError):
      cpeval.compute_size_with_checks(jnp.array(confidence_sets))

  @parameterized.parameters([
      dict(confidence_sets=np.zeros((100, 10)),
           conditional_labels=np.array([])),
      dict(confidence_sets=np.zeros((100, 10)),
           conditional_labels=np.ones((99))),
      dict(confidence_sets=np.zeros((100, 10)),
           conditional_labels=np.ones((100, 10))),
  ])
  def test_compute_conditional_size_errors(
      self, confidence_sets, conditional_labels):
    with self.assertRaises(ValueError):
      cpeval.compute_conditional_size_with_checks(
          jnp.array(confidence_sets), jnp.array(conditional_labels), 1)

  @parameterized.parameters([
      dict(confidence_sets=np.zeros((100, 10)),
           conditional_labels=np.ones((100))),
  ])
  def test_compute_conditional_size_jit(
      self, confidence_sets, conditional_labels):
    compute_conditional_size_fn = jax.jit(cpeval.compute_conditional_size)
    compute_conditional_size_fn(
        jnp.array(confidence_sets), jnp.array(conditional_labels), 1)


if __name__ == '__main__':
  absltest.main()
