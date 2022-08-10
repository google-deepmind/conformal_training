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

"""Evaluate experiment."""
import os
import sys

from absl import flags
from absl import logging
import jax

from absl import app
import colab_utils as cbutils

FLAGS = flags.FLAGS
flags.DEFINE_string('experiment_path', './', 'base path for experiments')
flags.DEFINE_string('experiment_dataset', '', 'dataset to evaluate')
flags.DEFINE_string(
    'experiment_method', 'thr', 'conformal predictor to use, thr or apr')
flags.DEFINE_boolean('experiment_logfile', False,
                     'log results to file in experiment_path')


def main(argv):
  del argv

  if FLAGS.experiment_logfile:
    logging.get_absl_handler().use_absl_log_file(
        f'eval_{FLAGS.experiment_method}', FLAGS.experiment_path)
  else:
    logging.get_absl_handler().python_handler.stream = sys.stdout

  if not os.path.exists(FLAGS.experiment_path):
    logging.error('could not find experiment path %s', FLAGS.experiment_path)
    return

  alpha = 0.01
  if FLAGS.experiment_method == 'thr':
    calibrate_fn, predict_fn = cbutils.get_threshold_fns(alpha)
  elif FLAGS.experiment_method == 'aps':
    calibrate_fn, predict_fn = cbutils.get_raps_fns(alpha, 0, 0)
  else:
    raise ValueError('Invalid conformal predictor, choose thr or aps.')

  if FLAGS.experiment_dataset == 'mnist':
    num_classes = 10
    groups = ['singleton', 'groups']
  elif FLAGS.experiment_dataset == 'emnist_byclass':
    num_classes = 52
    groups = ['groups']
  elif FLAGS.experiment_dataset == 'fashion_mnist':
    num_classes = 10
    groups = ['singleton']
  elif FLAGS.experiment_dataset == 'cifar10':
    num_classes = 10
    groups = ['singleton', 'groups']
  elif FLAGS.experiment_dataset == 'cifar100':
    num_classes = 100
    groups = ['groups', 'hierarchy']
  else:
    raise ValueError('Invalid dataset %s.' % FLAGS.experiment_dataset)

  model = cbutils.load_predictions(FLAGS.experiment_path, val_examples=5000)

  for group in groups:
    model['data']['groups'][group] = cbutils.get_groups(
        FLAGS.experiment_dataset, group)

  results = cbutils.evaluate_conformal_prediction(
      model, calibrate_fn, predict_fn, trials=10, rng=jax.random.PRNGKey(0))

  logging.info('Accuracy: %f', results['mean']['test']['accuracy'])
  logging.info('Coverage: %f', results['mean']['test']['coverage'])
  logging.info('Size: %f', results['mean']['test']['size'])

  for k in range(num_classes):
    logging.info(
        'Class size %d: %f', k, results['mean']['test'][f'class_size_{k}'])

  for group in groups:
    k = 0
    key = f'{group}_size_{k}'
    while key in results['mean']['test'].keys():
      logging.info(
          'Group %s size %d: %f', group, k, results['mean']['test'][key])
      k += 1
      key = f'{group}_size_{k}'

    logging.info(
        'Group %s miscoverage 0: %f',
        group, results['mean']['test'][f'{group}_miscoverage_0'])
    logging.info(
        'Group %s miscoverage 1: %f',
        group, results['mean']['test'][f'{group}_miscoverage_1'])

  # Selected coverage confusion combinations:
  logging.info(
      'Coverage confusion 4-6: %f',
      results['mean']['test']['coverage_confusion_4_6'])
  logging.info(
      'Coverage confusion 6-4: %f',
      results['mean']['test']['coverage_confusion_6_4'])


if __name__ == '__main__':
  app.run(main)
