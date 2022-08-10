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

"""Experiment definitions for CIFAR100 experiments."""
from typing import Tuple, Dict, Any, Optional

import ml_collections as collections
import numpy as np

import experiments.experiment_utils as cpeutils


def get_parameters(
    experiment: str,
    sub_experiment: str,
    config: collections.ConfigDict,
) -> Tuple[collections.ConfigDict, Optional[Dict[str, Any]]]:
  """Get parameters for CIFAR100 experiments.

  Args:
    experiment: experiment to run
    sub_experiment: sub experiment, e.g., parameter to tune
    config: experiment configuration

  Returns:
    Training configuration and parameter sweeps
  """
  config.architecture = 'resnet'
  config.resnet.version = 50
  config.resnet.channels = 64  # 256
  config.cifar_augmentation = 'standard+autoaugment+cutout'
  parameter_sweep = None
  groups = (
      1, 1, 0, 1, 1, 0, 1, 1, 0, 0,
      0, 0, 0, 0, 1, 1, 0, 0, 1, 1,
      0, 1, 0, 1, 1, 0, 1, 1, 0, 1,
      1, 1, 1, 1, 1, 0, 1, 0, 1, 0,
      0, 0, 1, 1, 1, 1, 0, 1, 0, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
      1, 0, 1, 1, 1, 1, 1, 1, 0, 0,
      1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
      1, 0, 1, 1, 0, 0, 0, 0, 1, 0,
      0, 1, 1, 1, 0, 1, 1, 1, 0, 1,
  )
  hierarchy = (
      4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
      3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
      6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
      0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
      5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
      16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
      10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
      2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
      16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
      18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
  )

  if experiment == 'models':
    config.learning_rate = 0.05
    config.batch_size = 100
  else:
    config.epochs = 50
    config.finetune.enabled = True
    config.finetune.path = 'cifar100_models_seed0/'
    config.finetune.model_state = False
    config.finetune.layers = 'res_net/~/logits'
    config.finetune.reinitialize = True

    if experiment == 'baseline':
      config.mode = 'normal'
    elif experiment == 'conformal':
      config.mode = 'conformal'
      config.conformal.coverage_loss = 'none'
      config.conformal.loss_transform = 'log'
      config.conformal.size_transform = 'identity'
      config.conformal.rng = False

      if sub_experiment == 'training':
        config.learning_rate = 0.005
        config.batch_size = 100
        config.conformal.temperature = 1.
        config.conformal.size_loss = 'normal'
        config.conformal.method = 'threshold_logp'
        config.conformal.size_weight = 0.005
      elif sub_experiment.find('hierarchy_size_') >= 0:
        config.learning_rate = 0.005
        config.batch_size = 100
        config.conformal.temperature = 1.
        config.conformal.size_loss = 'normal'
        config.conformal.method = 'threshold_logp'
        config.conformal.size_weight = 0.005

        selected_hierarchy = int(sub_experiment.replace('hierarchy_size_', ''))
        def cifar100_size_weights(selected_group, selected_weight, num_groups):
          """Helper to define size weights for hierarchy weight manipulation."""
          weights = np.ones(num_groups)
          weights[selected_group] = selected_weight
          return tuple(weights)

        parameter_sweep = {
            'key': 'conformal.size_weights',
            'values': [
                cpeutils.size_weights_group(
                    hierarchy,
                    cifar100_size_weights(selected_hierarchy, 1.1, 20)),
                cpeutils.size_weights_group(
                    hierarchy,
                    cifar100_size_weights(selected_hierarchy, 1.25, 20)),
                cpeutils.size_weights_group(
                    hierarchy,
                    cifar100_size_weights(selected_hierarchy, 1.5, 20)),
                cpeutils.size_weights_group(
                    hierarchy,
                    cifar100_size_weights(selected_hierarchy, 2, 20)),
                cpeutils.size_weights_group(
                    hierarchy,
                    cifar100_size_weights(selected_hierarchy, 3, 20)),
                cpeutils.size_weights_group(
                    hierarchy,
                    cifar100_size_weights(selected_hierarchy, 4, 20)),
                cpeutils.size_weights_group(
                    hierarchy,
                    cifar100_size_weights(selected_hierarchy, 5, 20)),
            ],
        }
      elif sub_experiment == 'group_zero':
        config.learning_rate = 0.005
        config.batch_size = 100
        config.conformal.temperature = 1.
        config.conformal.coverage_loss = 'classification'
        config.conformal.size_loss = 'valid'
        config.conformal.method = 'threshold_logp'
        config.conformal.size_weight = 0.01

        parameter_sweep = {
            'key': 'conformal.loss_matrix',
            'values': [
                cpeutils.loss_matrix_group_zero(0.01, 1, groups, 100),
                cpeutils.loss_matrix_group_zero(0.05, 1, groups, 100),
                cpeutils.loss_matrix_group_zero(0.1, 1, groups, 100),
                cpeutils.loss_matrix_group_zero(0.5, 1, groups, 100),
                cpeutils.loss_matrix_group_zero(1, 1, groups, 100),
            ],
        }
      elif sub_experiment == 'group_one':
        config.learning_rate = 0.005
        config.batch_size = 100
        config.conformal.temperature = 1.
        config.conformal.coverage_loss = 'classification'
        config.conformal.size_loss = 'valid'
        config.conformal.method = 'threshold_logp'
        config.conformal.size_weight = 0.01

        parameter_sweep = {
            'key': 'conformal.loss_matrix',
            'values': [
                cpeutils.loss_matrix_group_one(0.01, 1, groups, 100),
                cpeutils.loss_matrix_group_one(0.05, 1, groups, 100),
                cpeutils.loss_matrix_group_one(0.1, 1, groups, 100),
                cpeutils.loss_matrix_group_one(0.5, 1, groups, 100),
                cpeutils.loss_matrix_group_one(1, 1, groups, 100),
            ],
        }
      else:
        raise ValueError('Invalid conformal sub experiment.')
    else:
      raise ValueError('Experiment not implemented.')
  return config, parameter_sweep
