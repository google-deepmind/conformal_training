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

"""Tests for evaluation utilities."""
import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
import ml_collections as collections
import numpy as np

import conformal_training.colab_utils as cpcolab
import conformal_training.data_utils as cpdatautils
import conformal_training.test_utils as cptutils


class ColabUtilsTest(parameterized.TestCase):

  def _get_model(self, num_examples, num_classes):
    val_examples = num_examples//2
    labels = cptutils.get_labels(num_examples, num_classes)
    logits = cptutils.get_probabilities(labels, dominance=0.5)
    config = collections.ConfigDict()
    config.dataset = 'cifar10'
    config.val_examples = val_examples
    data = cpdatautils.get_data_stats(config)
    data['groups'] = {'groups': cpcolab.get_groups(config.dataset, 'groups')}
    model = {
        'val_logits': logits[:val_examples],
        'val_labels': labels[:val_examples],
        'test_logits': logits[val_examples:],
        'test_labels': labels[val_examples:],
        'data': data,
    }
    return model

  def _check_results(self, results):
    self.assertIn('mean', results.keys())
    self.assertIn('std', results.keys())
    if os.getenv('EVAL_VAL', '0') == '1':
      self.assertIn('val', results['mean'].keys())
    self.assertIn('test', results['mean'].keys())

    # Just test whether some basic metrics are there and not NaN or so.
    metrics_to_check = [
        'size', 'coverage', 'accuracy',
        'class_size_0', 'class_coverage_0',
        'size_0', 'cumulative_size_0',
        'groups_miscoverage',
    ]
    if os.getenv('EVAL_CONFUSION') == '1':
      metrics_to_check += [
          'classification_confusion_0_0', 'coverage_confusion_0_0'
      ]
    for metric in metrics_to_check:
      mean = results['mean']['test'][metric]
      std = results['std']['test'][metric]
      self.assertFalse(np.isnan(mean))
      self.assertFalse(np.isinf(mean))
      self.assertGreaterEqual(mean, 0.)
      self.assertFalse(np.isnan(std))
      self.assertFalse(np.isinf(std))
      self.assertGreaterEqual(std, 0.)
    # Extra check for cumulative size
    self.assertAlmostEqual(results['mean']['test']['cumulative_size_9'], 1)

  def test_evaluate_conformal_prediction(self):
    num_examples = 1000
    num_classes = 10
    model = self._get_model(num_examples, num_classes)
    calibrate_fn, predict_fn = cpcolab.get_threshold_fns(0.05, jit=True)
    rng = jax.random.PRNGKey(0)
    results = cpcolab.evaluate_conformal_prediction(
        model, calibrate_fn, predict_fn, trials=2, rng=rng)
    self._check_results(results)


if __name__ == '__main__':
  absltest.main()
