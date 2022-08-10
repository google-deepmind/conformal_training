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

"""Launches experiments."""
import copy
import os
from typing import Tuple, Dict, Any

from absl import flags
from absl import logging
import ml_collections as collections

from absl import app
from config import get_config

# pylint: disable=unused-import
from experiments.run_cifar10 import get_parameters as get_cifar10_parameters
from experiments.run_cifar100 import get_parameters as get_cifar100_parameters
from experiments.run_emnist_byclass import get_parameters as get_emnist_byclass_parameters
from experiments.run_fashion_mnist import get_parameters as get_fashion_mnist_parameters
from experiments.run_mnist import get_parameters as get_mnist_parameters
from experiments.run_wine_quality import get_parameters as get_wine_quality_parameters
from train import train

FLAGS = flags.FLAGS
flags.DEFINE_string('experiment_dataset', 'cifar10', 'dataset to use')
flags.DEFINE_string('experiment_experiment', '',
                    'experiments to run on dataset')
flags.DEFINE_integer('experiment_seeds', 10,
                     'number of seed to run per experiment')
flags.DEFINE_boolean('experiment_debug', False,
                     'debug experiment sweep and seeds')
flags.DEFINE_string('experiment_path', './', 'base path for experiments')


def get_parameters(
    dataset: str,
    experiment: str,
) -> Tuple[collections.ConfigDict, Dict[str, Any]]:
  """Get parameters for given dataset.

  Args:
    dataset: dataset to experiment on
    experiment: experiment to run

  Returns:
    Configuration arguments and hyper-parameter sweep.
  """

  config = get_config()
  config.architecture = 'mlp'
  config.cnn.channels = 32
  config.cnn.layers = 3
  config.cnn.kernels = 3
  config.cnn.activation = 'relu'
  config.mlp.units = 64
  config.mlp.layers = 2
  config.mlp.activation = 'relu'
  config.resnet.version = 18
  config.resnet.channels = 32
  config.resnet.resnet_v2 = True
  config.resnet.init_logits = True
  config.optimizer = 'sgd'
  config.adam.b1 = 0.9
  config.adam.b2 = 0.999
  config.adam.eps = 1e-8
  config.sgd.momentum = 0.9
  config.sgd.nesterov = True
  config.learning_rate = 0.01
  config.learning_rate_schedule = 'step'
  config.step.learning_rate_decay = 0.1
  config.exponential.learning_rate_decay = 0.95
  config.mode = 'normal'
  config.coverage.method = 'threshold_p'
  config.coverage.alpha = 0.01
  config.coverage.target_alpha = 0.01
  config.coverage.temperature = 1.
  config.coverage.dispersion = 0.1
  config.coverage.size_weight = 0.05
  config.coverage.tau = 1
  config.coverage.coverage_loss = 'classification'
  config.coverage.loss_matrix = ()
  config.coverage.cross_entropy_weight = 0.
  config.coverage.size_loss = 'valid'
  config.coverage.size_transform = 'identity'
  config.coverage.size_bound = 3.
  config.coverage.size_bound_weight = 0.9
  config.coverage.loss_transform = 'log'
  config.coverage.size_weights = ()
  config.coverage.rng = False
  config.coverage.calibration_batches = 10
  config.conformal.method = 'threshold_p'
  config.conformal.alpha = 0.01
  config.conformal.target_alpha = 0.01
  config.conformal.temperature = 1.
  config.conformal.dispersion = 0.1
  config.conformal.size_weight = 0.1
  config.conformal.coverage_loss = 'classification'
  config.conformal.loss_matrix = ()
  config.conformal.cross_entropy_weight = 0.
  config.conformal.size_loss = 'valid'
  config.conformal.size_transform = 'identity'
  config.conformal.size_bound = 3.
  config.conformal.size_bound_weight = 0.9
  config.conformal.loss_transform = 'log'
  config.conformal.size_weights = ()
  config.conformal.fraction = 0.5
  config.conformal.rng = False
  config.weight_decay = 0.0005
  config.batch_size = 500
  config.test_batch_size = 100
  config.epochs = 150
  config.finetune.enabled = False
  config.finetune.model_state = False
  config.finetune.experiment_id = None
  config.finetune.work_unit_id = None
  config.finetune.layers = None
  config.finetune.reinitialize = False
  config.dataset = dataset
  config.seed = 0
  config.checkpoint_frequency = 10
  config.resampling = 0
  config.whitening = True
  config.cifar_augmentation = 'standard+autoaugment+cutout'
  config.val_examples = 5000
  config.checkpoint_dtl = 155
  config.jit = True

  experiment = experiment.split('.')
  sub_experiment = experiment[1] if len(experiment) > 1 else None
  experiment = experiment[0]
  get_parameters_key = 'get_%s_parameters' % dataset
  if get_parameters_key not in globals().keys():
    raise ValueError('Experiment definitions could not be loaded.')
  config, parameter_sweep = globals()[get_parameters_key](
      experiment, sub_experiment, config)

  return config, parameter_sweep


def main(argv):
  del argv

  supported_datasets = (
      'wine_quality',
      'mnist',
      'emnist_byclass',
      'fashion_mnist',
      'cifar10',
      'cifar100',
  )
  if FLAGS.experiment_dataset not in supported_datasets:
    raise ValueError('Invalid dataset selected.')
  if FLAGS.experiment_seeds <= 0:
    raise ValueError('Invalid number of seeds.')

  logging.info(
      'starting dataset=%s experiment=%s seeds=%d',
      FLAGS.experiment_dataset, FLAGS.experiment_experiment,
      FLAGS.experiment_seeds)
  config, parameter_sweep = get_parameters(
      FLAGS.experiment_dataset, FLAGS.experiment_experiment)
  config.path = os.path.join(
      FLAGS.experiment_path,
      '%s_%s' % (FLAGS.experiment_dataset, FLAGS.experiment_experiment))
  config.finetune.path = os.path.join(
      FLAGS.experiment_path, config.finetune.path)
  if FLAGS.experiment_seeds > 1:
    config.resampling = 5
    logging.info('resampling=%d', config.resampling)

  def update_config(config, key, value):
    """Helper to easily update a config value by dot-separated key."""
    if key.count('.') > 1:
      raise ValueError(f'Key {key} not supported.')
    elif key.count('.') == 1:
      key, sub_key = key.split('.')
      config[key][sub_key] = value
    else:
      config[key] = value

  for seed in range(FLAGS.experiment_seeds):
    # A sweep in one parameter is supported, e.g., the loss matrix or weights.
    if parameter_sweep is not None:
      sweep_key = parameter_sweep['key']
      for i, sweep_value in enumerate(parameter_sweep['values']):
        sweep_config = copy.deepcopy(config)
        update_config(sweep_config, sweep_key, sweep_value)
        update_config(sweep_config, 'seed', seed)
        path = config.path + '_value%d_seed%d/' % (i, seed)
        update_config(sweep_config, 'path', path)
        logging.info(
            'running %s=%r seed=%d path=%s', sweep_key, sweep_value, seed, path)
        if not FLAGS.experiment_debug:
          train(sweep_config)
    # Only update config regarding seed and path, no other values are changed.
    else:
      seed_config = copy.deepcopy(config)
      update_config(seed_config, 'seed', seed)
      path = config.path + '_seed%d/' % seed
      update_config(seed_config, 'path', path)
      logging.info('running seed=%d path=%s', seed, path)
      if not FLAGS.experiment_debug:
        train(seed_config)


if __name__ == '__main__':
  app.run(main)
