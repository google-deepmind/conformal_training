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

"""Experiment definitions for CIFAR10."""
from typing import Tuple, Dict, Any, Optional

import ml_collections as collections

import conformal_training.experiments.experiment_utils as cpeutils


def get_parameters(
    experiment: str,
    sub_experiment: str,
    config: collections.ConfigDict,
) -> Tuple[collections.ConfigDict, Optional[Dict[str, Any]]]:
  """Get parameters for CIFAR10 experiments.

  Args:
    experiment: experiment to run
    sub_experiment: sub experiment, e.g., parameter to tune
    config: experiment configuration

  Returns:
    Training configuration and parameter sweeps
  """
  config.architecture = 'resnet'
  config.resnet.version = 34
  config.resnet.channels = 4
  config.cifar_augmentation = 'standard'

  parameter_sweep = None
  groups = (0, 0, 1, 1, 1, 1, 1, 1, 0, 0)

  # Training from scratch:
  if experiment == 'models':
    config.whitening = True
    config.cifar_augmentation = 'standard'
  # Fine-tuning
  else:
    config.epochs = 50
    config.finetune.enabled = True
    config.finetune.path = './cifar10_models_seed0/'
    config.finetune.model_state = False
    config.finetune.layers = 'res_net/~/logits'
    config.finetune.reinitialize = True

    if experiment == 'baseline_trials':
      config.mode = 'normal'
    elif experiment == 'conformal':
      config.mode = 'conformal'
      config.conformal.coverage_loss = 'none'
      config.conformal.loss_transform = 'log'
      config.conformal.size_transform = 'identity'
      config.conformal.rng = False

      if sub_experiment == 'finetune_threshold_logp_trials':
        config.learning_rate = 0.01
        config.batch_size = 500
        config.conformal.temperature = 1.
        config.conformal.size_loss = 'normal'
        config.conformal.method = 'threshold_logp'
        config.conformal.size_weight = 0.05
      elif sub_experiment == 'group_zero':
        config.learning_rate = 0.005
        config.batch_size = 500
        config.conformal.temperature = 0.1
        config.conformal.coverage_loss = 'classification'
        config.conformal.size_loss = 'valid'
        config.conformal.method = 'threshold_logp'
        config.conformal.size_weight = 1.

        parameter_sweep = {
            'key': 'conformal.loss_matrix',
            'values': [
                cpeutils.loss_matrix_group_zero(0.001, 1, groups, 10),
                cpeutils.loss_matrix_group_zero(0.005, 1, groups, 10),
                cpeutils.loss_matrix_group_zero(0.01, 1, groups, 10),
                cpeutils.loss_matrix_group_zero(0.05, 1, groups, 10),
                cpeutils.loss_matrix_group_zero(0.1, 1, groups, 10),
                cpeutils.loss_matrix_group_zero(0.5, 1, groups, 10),
                cpeutils.loss_matrix_group_zero(1, 1, groups, 10),
                cpeutils.loss_matrix_group_zero(2, 1, groups, 10),
            ],
        }
      elif sub_experiment == 'group_one':
        config.learning_rate = 0.005
        config.batch_size = 500
        config.conformal.temperature = 0.1
        config.conformal.coverage_loss = 'classification'
        config.conformal.size_loss = 'valid'
        config.conformal.method = 'threshold_logp'
        config.conformal.size_weight = 1.

        parameter_sweep = {
            'key': 'conformal.loss_matrix',
            'values': [
                cpeutils.loss_matrix_group_one(0.001, 1, groups, 10),
                cpeutils.loss_matrix_group_one(0.005, 1, groups, 10),
                cpeutils.loss_matrix_group_one(0.01, 1, groups, 10),
                cpeutils.loss_matrix_group_one(0.05, 1, groups, 10),
                cpeutils.loss_matrix_group_one(0.1, 1, groups, 10),
                cpeutils.loss_matrix_group_one(0.5, 1, groups, 10),
                cpeutils.loss_matrix_group_one(1, 1, groups, 10),
                cpeutils.loss_matrix_group_one(2, 1, groups, 10),
            ],
        }
      elif sub_experiment == 'group_size_0':
        config.learning_rate = 0.01
        config.batch_size = 500
        config.conformal.temperature = 1.
        config.conformal.size_loss = 'normal'
        config.conformal.method = 'threshold_logp'
        config.conformal.size_weight = 0.05

        parameter_sweep = {
            'key': 'conformal.size_weights',
            'values': [
                cpeutils.size_weights_group(groups, (1.1, 1)),
                cpeutils.size_weights_group(groups, (1.25, 1)),
                cpeutils.size_weights_group(groups, (1.5, 1)),
                cpeutils.size_weights_group(groups, (2, 1)),
                cpeutils.size_weights_group(groups, (3, 1)),
                cpeutils.size_weights_group(groups, (4, 1)),
                cpeutils.size_weights_group(groups, (5, 1)),
            ],
        }
      elif sub_experiment == 'group_size_1':
        config.learning_rate = 0.01
        config.batch_size = 500
        config.conformal.temperature = 1.
        config.conformal.size_loss = 'normal'
        config.conformal.method = 'threshold_logp'
        config.conformal.size_weight = 0.05

        parameter_sweep = {
            'key': 'conformal.size_weights',
            'values': [
                cpeutils.size_weights_group(groups, (1, 1.1)),
                cpeutils.size_weights_group(groups, (1, 1.25)),
                cpeutils.size_weights_group(groups, (1, 1.5)),
                cpeutils.size_weights_group(groups, (1, 2)),
                cpeutils.size_weights_group(groups, (1, 3)),
                cpeutils.size_weights_group(groups, (1, 4)),
                cpeutils.size_weights_group(groups, (1, 5)),
            ],
        }
      elif sub_experiment.find('class_size_') >= 0:
        config.learning_rate = 0.01
        config.batch_size = 500
        config.conformal.temperature = 1.
        config.conformal.size_loss = 'normal'
        config.conformal.method = 'threshold_logp'
        config.conformal.size_weight = 0.05

        selected_class = int(sub_experiment.replace('class_size_', ''))
        parameter_sweep = {
            'key': 'conformal.size_weights',
            'values': [
                cpeutils.size_weights_selected([selected_class], 0, 10),
                cpeutils.size_weights_selected([selected_class], 0.1, 10),
                cpeutils.size_weights_selected([selected_class], 0.5, 10),
                cpeutils.size_weights_selected([selected_class], 1, 10),
                cpeutils.size_weights_selected([selected_class], 2, 10),
                cpeutils.size_weights_selected([selected_class], 5, 10),
                cpeutils.size_weights_selected([selected_class], 10, 10),
            ],
        }

      else:
        raise ValueError('Invalid conformal sub experiment.')
    else:
      raise ValueError('Experiment not implemented.')
  return config, parameter_sweep
