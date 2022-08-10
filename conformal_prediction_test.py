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

"""Tests for conformal prediction calibration and predictions."""
import functools
from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax
import jax.numpy as jnp
import numpy as np

import conformal_prediction as cp
import test_utils as cptutils


class ConformalPredictionTest(parameterized.TestCase):

  def setUp(self):
    super(ConformalPredictionTest, self).setUp()
    np.random.seed(0)

  @parameterized.parameters([
      dict(array=np.array([]), q=0.5),
      dict(array=np.linspace(0, 1, 100), q=-0.1),
      dict(array=np.linspace(0, 1, 100), q=1.1),
      dict(array=np.linspace(0, 1, 100).reshape(2, 50), q=0.5),
  ])
  def test_conformal_quantile_errors(self, array, q):
    with self.assertRaises(ValueError):
      cp.conformal_quantile_with_checks(jnp.array(array), q=q)

  def _test_validation_coverage(self, calibrate, predict, alpha):
    num_examples = 10000
    num_classes = 10
    labels = cptutils.get_labels(num_examples, num_classes)
    probabilities = cptutils.get_probabilities(labels, 5)
    threshold = calibrate(probabilities, labels, alpha)
    confidence_sets = predict(probabilities, threshold)
    # Some methods will have perfect coverage in this case as the true class
    # always represents the largest probability.
    # Others will only have 1 - alpha coverage as coverage is independent of
    # sorting.
    self.assertGreaterEqual(
        jnp.sum(confidence_sets[jnp.arange(confidence_sets.shape[0]), labels]),
        int(num_examples * (1 - alpha)))

  @parameterized.parameters([
      dict(alpha=0.1),
      dict(alpha=0.01),
  ])
  def test_threshold_confidence_sets_validation_coverage(self, alpha):
    calibrate = cp.calibrate_threshold_with_checks
    predict = cp.predict_threshold_with_checks
    self._test_validation_coverage(calibrate, predict, alpha)

  @parameterized.parameters([
      dict(probabilities=np.zeros((100, 10)),
           labels=np.zeros((100)).astype(int)),
  ])
  def test_calibrate_predict_threshold_jit(self, probabilities, labels):
    calibrate_threshold_fn = jax.jit(
        functools.partial(cp.calibrate_threshold, alpha=0.1))
    predict_threshold_fn = jax.jit(cp.predict_threshold)
    tau = calibrate_threshold_fn(probabilities, labels)
    confidence_sets = predict_threshold_fn(probabilities, tau)
    chex.assert_shape(confidence_sets, probabilities.shape)

  @parameterized.parameters([
      dict(probabilities=np.array([]), tau=0),
      dict(probabilities=np.zeros((100)), tau=0),
  ])
  def test_predict_threshold_errors(self, probabilities, tau):
    with self.assertRaises(ValueError):
      cp.predict_threshold_with_checks(jnp.array(probabilities), tau)

  @parameterized.parameters([
      dict(alpha=0.1, k_reg=None, lambda_reg=None),
      dict(alpha=0.01, k_reg=None, lambda_reg=None),
      dict(alpha=0.1, k_reg=1, lambda_reg=0),
      dict(alpha=0.01, k_reg=1, lambda_reg=0),
      dict(alpha=0.1, k_reg=1, lambda_reg=0.5),
      dict(alpha=0.01, k_reg=1, lambda_reg=0.5),
  ])
  def test_raps_confidence_sets_validation_coverage(
      self, alpha, k_reg, lambda_reg):
    calibrate = functools.partial(
        cp.calibrate_raps_with_checks,
        k_reg=k_reg, lambda_reg=lambda_reg)
    predict = functools.partial(
        cp.predict_raps_with_checks, k_reg=k_reg, lambda_reg=lambda_reg)
    self._test_validation_coverage(calibrate, predict, alpha)

  @parameterized.parameters([
      dict(probabilities=np.zeros((100, 10)),
           labels=np.zeros((100)).astype(int)),
  ])
  def test_calibrate_predict_raps_jit(self, probabilities, labels):
    calibrate_raps_fn = jax.jit(functools.partial(
        cp.calibrate_raps, alpha=0.1, k_reg=1, lambda_reg=0.5))
    predict_raps_fn = jax.jit(functools.partial(
        cp.predict_raps, k_reg=1, lambda_reg=0.5))
    rng = jax.random.PRNGKey(0)
    tau = calibrate_raps_fn(probabilities, labels, rng=rng)
    confidence_sets = predict_raps_fn(probabilities, tau, rng=rng)
    chex.assert_shape(confidence_sets, probabilities.shape)

  @parameterized.parameters([
      dict(probabilities=np.array([]),
           labels=np.array([]), alpha=0.1),
      dict(probabilities=np.zeros((100)),
           labels=np.ones((100), dtype=int), alpha=0.1),
      dict(probabilities=np.zeros((100, 10)),
           labels=np.ones((100), dtype=int) * 99, alpha=0.1),
      dict(probabilities=np.zeros((100, 10)),
           labels=np.ones((100)) * 0.5, alpha=0.1),
      dict(probabilities=np.zeros((100, 10)),
           labels=np.ones((100), dtype=int), alpha=-0.1),
      dict(probabilities=np.zeros((100, 10)),
           labels=np.ones((100), dtype=int), alpha=1.1),
  ])
  def test_calibrate_raps_errors(self, probabilities, labels, alpha):
    with self.assertRaises(ValueError):
      cp.calibrate_raps_with_checks(
          jnp.array(probabilities), jnp.array(labels), alpha, rng=None)

  @parameterized.parameters([
      dict(probabilities=np.array([]), tau=0.9),
      dict(probabilities=np.zeros((100)), tau=0.9),
      dict(probabilities=np.zeros((100, 10)), tau=-0.1),
  ])
  def test_predict_raps_errors(self, probabilities, tau):
    with self.assertRaises(ValueError):
      cp.predict_raps_with_checks(jnp.array(probabilities), tau, rng=None)


if __name__ == '__main__':
  absltest.main()
