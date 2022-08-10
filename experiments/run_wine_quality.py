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

"""Launch definitions for WineQuality."""
from typing import Tuple, Dict, Any, Optional

import ml_collections as collections

import conformal_training.experiments.experiment_utils as cpeutils


def get_parameters(
    experiment: str,
    sub_experiment: str,
    config: collections.ConfigDict,
) -> Tuple[collections.ConfigDict, Optional[Dict[str, Any]]]:
  """Get parameters for Wine Quality experiments.

  Args:
    experiment: experiment to run
    sub_experiment: sub experiment, e.g., parameter to tune
    config: experiment configuration

  Returns:
    Training configuration and parameter sweeps
  """
  config.architecture = 'mlp'
  config.mlp.layers = 2
  config.mlp.units = 256
  config.learning_rate = 0.01
  config.val_examples = 500
  config.epochs = 100
  config.checkpoint_frequency = 10
  parameter_sweep = None

  if experiment == 'models':
    config.learning_rate = 0.1
    config.batch_size = 500
  elif experiment == 'conformal':
    config.mode = 'conformal'
    config.conformal.coverage_loss = 'none'
    config.conformal.loss_transform = 'log'
    config.conformal.size_transform = 'identity'
    config.conformal.rng = False

    if sub_experiment == 'threshold_logp_trials':
      config.learning_rate = 0.005
      config.batch_size = 100
      config.conformal.temperature = 0.5
      config.conformal.method = 'threshold_logp'
      config.conformal.size_weight = 0.05
    elif sub_experiment == 'importance_0':
      config.conformal.coverage_loss = 'classification'
      config.learning_rate = 0.05
      config.batch_size = 500
      config.conformal.temperature = 0.5
      config.conformal.method = 'threshold_p'
      config.conformal.size_weight = 1.
      config.conformal.size_loss = 'valid'

      parameter_sweep = {
          'key': 'conformal.loss_matrix',
          'values': [
              cpeutils.loss_matrix_importance((1, 0.25), 2),
              cpeutils.loss_matrix_importance((1, 0.5), 2),
              cpeutils.loss_matrix_importance((1.25, 1), 2),
              cpeutils.loss_matrix_importance((1.5, 1), 2),
              cpeutils.loss_matrix_importance((2, 1), 2),
              cpeutils.loss_matrix_importance((4, 1), 2),
          ],
      }
    elif sub_experiment == 'importance_1':
      config.conformal.coverage_loss = 'classification'
      config.learning_rate = 0.05
      config.batch_size = 500
      config.conformal.temperature = 0.5
      config.conformal.method = 'threshold_p'
      config.conformal.size_weight = 1.
      config.conformal.size_loss = 'valid'

      parameter_sweep = {
          'key': 'conformal.loss_matrix',
          'values': [
              cpeutils.loss_matrix_importance((0.25, 1), 2),
              cpeutils.loss_matrix_importance((0.5, 1), 2),
              cpeutils.loss_matrix_importance((1, 1.25), 2),
              cpeutils.loss_matrix_importance((1, 1.5), 2),
              cpeutils.loss_matrix_importance((1, 2), 2),
              cpeutils.loss_matrix_importance((1, 4), 2),

          ],
      }
    elif sub_experiment == 'confusion_1_0':
      config.conformal.coverage_loss = 'classification'
      config.learning_rate = 0.05
      config.batch_size = 500
      config.conformal.temperature = 0.5
      config.conformal.method = 'threshold_p'
      config.conformal.size_weight = 1.
      config.conformal.size_loss = 'valid'

      parameter_sweep = {
          'key': 'conformal.loss_matrix',
          'values': [
              cpeutils.loss_matrix_confusion(0, 1, 0, 0.01, 1, 2),
              cpeutils.loss_matrix_confusion(0, 1, 0, 0.05, 1, 2),
              cpeutils.loss_matrix_confusion(0, 1, 0, 0.1, 1, 2),
              cpeutils.loss_matrix_confusion(0, 1, 0, 0.5, 1, 2),
              cpeutils.loss_matrix_confusion(0, 1, 0, 1, 1, 2),
          ],
      }
    elif sub_experiment == 'confusion_0_1':
      config.conformal.coverage_loss = 'classification'
      config.learning_rate = 0.05
      config.batch_size = 500
      config.conformal.temperature = 0.5
      config.conformal.method = 'threshold_p'
      config.conformal.size_weight = 1.
      config.conformal.size_loss = 'valid'

      parameter_sweep = {
          'key': 'conformal.loss_matrix',
          'values': [
              cpeutils.loss_matrix_confusion(0, 1, 0.01, 0, 1, 2),
              cpeutils.loss_matrix_confusion(0, 1, 0.05, 0, 1, 2),
              cpeutils.loss_matrix_confusion(0, 1, 0.1, 0, 1, 2),
              cpeutils.loss_matrix_confusion(0, 1, 0.5, 0, 1, 2),
              cpeutils.loss_matrix_confusion(0, 1, 1, 0, 1, 2),
          ],
      }
    else:
      raise ValueError('Invalid conformal sub experiment.')
  else:
    raise ValueError('Experiment not implemented.')
  return config, parameter_sweep
