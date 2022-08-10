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

"""Tests for smooth conformal prediction."""
import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

import sorting_nets
import variational_sorting_net
import conformal_prediction as cp
import smooth_conformal_prediction as scp
import test_utils as cptutils


class SmoothConformalPredictionTest(parameterized.TestCase):

  def setUp(self):
    super(SmoothConformalPredictionTest, self).setUp()
    np.random.seed(0)

  def _get_smooth_order_stats(self, length):
    comm = sorting_nets.comm_pattern_batcher(length, make_parallel=True)
    sos = variational_sorting_net.VariationalSortingNet(
        comm, smoothing_strategy='entropy_reg', sorting_strategy='hard')
    return sos

  @parameterized.parameters([
      dict(num_examples=10000, num_classes=10, tau=0.9)
  ])
  def test_smooth_predict_aps(self, num_examples, num_classes, tau):
    # Randomness is generally not handled equivalently.
    rng = None
    labels = cptutils.get_labels(num_examples, num_classes)
    probabilities = cptutils.get_probabilities(labels, 0)
    confidence_sets = cp.predict_raps(
        probabilities, tau, k_reg=None, lambda_reg=None, rng=rng)
    dispersion = 0.00001
    temperature = 0.00001
    sos = self._get_smooth_order_stats(num_classes)
    smooth_confidence_sets = scp.smooth_predict_aps(
        probabilities, tau, sos, rng=rng,
        temperature=temperature, dispersion=dispersion)
    smooth_confidence_sets = smooth_confidence_sets.at[
        smooth_confidence_sets > 0.5].set(1)
    smooth_confidence_sets = smooth_confidence_sets.at[
        smooth_confidence_sets <= 0.5].set(0)
    np.testing.assert_equal(np.array(confidence_sets),
                            np.array(smooth_confidence_sets))

  @parameterized.parameters([
      dict(probabilities=np.array([]), tau=0.9,
           temperature=0.01, dispersion=0.01, length=10),
      dict(probabilities=np.zeros((100)), tau=0.9,
           temperature=0.01, dispersion=0.01, length=10),
      dict(probabilities=np.zeros((100, 10)), tau=-0.1,
           temperature=0.01, dispersion=0.01, length=10),
      dict(probabilities=np.zeros((100, 10)), tau=0.9,
           temperature=0, dispersion=0.01, length=10),
      dict(probabilities=np.zeros((100, 10)), tau=0.9,
           temperature=0.01, dispersion=0, length=10),
      dict(probabilities=np.zeros((100, 10)), tau=0.9,
           temperature=-0.1, dispersion=0.01, length=10),
      dict(probabilities=np.zeros((100, 10)), tau=0.9,
           temperature=0.01, dispersion=-0.1, length=10),
      dict(probabilities=np.zeros((100, 10)), tau=0.9,
           temperature=0.01, dispersion=0.01, length=9),
  ])
  def test_smooth_predict_aps_errors(
      self, probabilities, tau, temperature, dispersion, length):
    with self.assertRaises(ValueError):
      sos = self._get_smooth_order_stats(length)
      scp.smooth_predict_aps_with_checks(
          jnp.array(probabilities), tau, sos, None, temperature, dispersion)

  @parameterized.parameters([
      dict(num_examples=1000, num_classes=10, alpha=0.9)
  ])
  def test_smooth_calibrate_aps(self, num_examples, num_classes, alpha):
    labels = cptutils.get_labels(num_examples, num_classes)
    probabilities = cptutils.get_probabilities(labels, 0)

    rng = None
    k_reg = None
    lambda_reg = None
    # If we want to have equality below, temperature and dispersion need
    # to be as low as possible to get results based on hard sorting and
    # hard thresholding.
    dispersion = 0.00001

    tau = cp.calibrate_raps(
        probabilities, labels, alpha=alpha,
        k_reg=k_reg, lambda_reg=lambda_reg, rng=rng)

    probabilities_sos = self._get_smooth_order_stats(num_classes)
    scores_sos = self._get_smooth_order_stats(num_examples)
    smooth_quantile_fn = functools.partial(
        scp.smooth_conformal_quantile, sos=scores_sos, dispersion=dispersion)
    tau_ = scp.smooth_calibrate_aps(
        probabilities, labels, alpha=alpha,
        sos=probabilities_sos, dispersion=dispersion,
        smooth_quantile_fn=smooth_quantile_fn, rng=rng)
    self.assertAlmostEqual(tau, tau_, places=2)

  @parameterized.parameters([
      dict(probabilities=np.array([]),
           labels=np.array([], dtype=int), alpha=0.1,
           length=10, dispersion=0.01),
      dict(probabilities=np.zeros((100)),
           labels=np.ones((100), dtype=int), alpha=0.1,
           length=10, dispersion=0.01),
      dict(probabilities=np.zeros((100, 10)),
           labels=np.ones((100), dtype=int) * 99, alpha=0.1,
           length=10, dispersion=0.01),
      dict(probabilities=np.zeros((100, 10)),
           labels=np.ones((100)) * 0.5, alpha=0.1,
           length=10, dispersion=0.01),
      dict(probabilities=np.zeros((100, 10)),
           labels=np.ones((100), dtype=int), alpha=-0.1,
           length=10, dispersion=0.01),
      dict(probabilities=np.zeros((100, 10)),
           labels=np.ones((100), dtype=int), alpha=1.1,
           length=10, dispersion=0.01),
      dict(probabilities=np.zeros((100, 10)),
           labels=np.ones((100), dtype=int), alpha=0.1,
           length=9, dispersion=0.01),
      dict(probabilities=np.zeros((100, 10)),
           labels=np.ones((100), dtype=int), alpha=0.1,
           length=10, dispersion=-1),
  ])
  def test_smooth_calibrate_aps_errors(
      self, probabilities, labels, alpha, length, dispersion):
    probabilities_sos = self._get_smooth_order_stats(length)
    scores_sos = self._get_smooth_order_stats(100)
    smooth_quantile_fn = functools.partial(
        scp.smooth_conformal_quantile_with_checks,
        sos=scores_sos, dispersion=0.1)
    with self.assertRaises(ValueError):
      scp.smooth_calibrate_aps_with_checks(
          jnp.array(probabilities), jnp.array(labels), alpha,
          sos=probabilities_sos, dispersion=dispersion,
          smooth_quantile_fn=smooth_quantile_fn, rng=None)

  @parameterized.parameters([
      dict(num_examples=10000, num_classes=10, tau=0.9)
  ])
  def test_predict_threshold(self, num_examples, num_classes, tau):
    labels = cptutils.get_labels(num_examples, num_classes)
    probabilities = cptutils.get_probabilities(labels, 0)
    confidence_sets = cp.predict_threshold(probabilities, tau)
    temperature = 0.00001
    smooth_confidence_sets = scp.smooth_predict_threshold(
        probabilities, tau, temperature=temperature)
    smooth_confidence_sets = smooth_confidence_sets.at[
        smooth_confidence_sets > 0.5].set(1)
    smooth_confidence_sets = smooth_confidence_sets.at[
        smooth_confidence_sets <= 0.5].set(0)
    np.testing.assert_equal(np.array(confidence_sets),
                            np.array(smooth_confidence_sets))

  @parameterized.parameters([
      dict(probabilities=np.array([]), tau=0.9,
           temperature=0.01),
      dict(probabilities=np.zeros((100)), tau=0.9,
           temperature=0.01),
      dict(probabilities=np.zeros((100, 10)), tau=-0.1,
           temperature=0.01),
      dict(probabilities=np.zeros((100, 10)), tau=0.9,
           temperature=0),
  ])
  def test_predict_threshdold_errors(self, probabilities, tau, temperature):
    with self.assertRaises(ValueError):
      scp.smooth_predict_threshold_with_checks(
          jnp.array(probabilities), tau, temperature)

  @parameterized.parameters([
      dict(num_examples=1000, num_classes=10, alpha=0.9)
  ])
  def test_smooth_calibrate_threshold(self, num_examples, num_classes, alpha):
    labels = cptutils.get_labels(num_examples, num_classes)
    probabilities = cptutils.get_probabilities(labels, 0)
    tau = cp.calibrate_threshold(probabilities, labels, alpha)

    dispersion = 0.00001
    scores_sos = self._get_smooth_order_stats(num_examples)
    smooth_quantile_fn = functools.partial(
        scp.smooth_conformal_quantile, sos=scores_sos, dispersion=dispersion)
    tau_ = scp.smooth_calibrate_threshold(
        probabilities, labels, alpha,
        smooth_quantile_fn=smooth_quantile_fn)
    self.assertAlmostEqual(tau, tau_, places=2)

  @parameterized.parameters([
      dict(probabilities=np.array([]),
           labels=np.array([], dtype=int), alpha=0.1),
      dict(probabilities=np.zeros((100)),
           labels=np.ones((100), dtype=int), alpha=0.1),
      dict(probabilities=np.zeros((100, 10)),
           labels=np.ones((100), dtype=int) * 99, alpha=0.1),
      dict(probabilities=np.zeros((100, 10)),
           labels=np.ones((100)) * 0.5, alpha=0.1),
      dict(probabilities=np.zeros((100, 10)),
           labels=np.ones((100), dtype=int), alpha=-0.1),
  ])
  def test_smooth_calibrate_threshold_errors(
      self, probabilities, labels, alpha):
    scores_sos = self._get_smooth_order_stats(100)
    smooth_quantile_fn = functools.partial(
        scp.smooth_conformal_quantile_with_checks,
        sos=scores_sos, dispersion=0.1)
    with self.assertRaises(ValueError):
      scp.smooth_calibrate_threshold_with_checks(
          jnp.array(probabilities), jnp.array(labels), alpha,
          smooth_quantile_fn=smooth_quantile_fn)


if __name__ == '__main__':
  absltest.main()
