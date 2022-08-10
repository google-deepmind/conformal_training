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

"""Launch definitions for EMNIST/byClass."""
from typing import Tuple, Dict, Any, Optional

import ml_collections as collections


def get_parameters(
    experiment: str,
    unused_sub_experiment: str,
    config: collections.ConfigDict,
) -> Tuple[collections.ConfigDict, Optional[Dict[str, Any]]]:
  """Get parameters for MNIST experiments.

  Args:
    experiment: experiment to run
    unused_sub_experiment: sub experiment, e.g., parameter to tune
    config: experiment configuration

  Returns:
    Training configuration and parameter sweeps
  """
  config.epochs = 75
  config.architecture = 'mlp'
  config.mlp.layers = 2
  config.mlp.units = 128
  config.cnn.channels = 32
  # We adjust the number of validation examples to the number of classes.
  config.val_examples = 52 * 100  # 4700 balanced, 5200 byClass.
  # For large batch sizes parts of validation/test sets might be
  # missing otherwise.

  parameter_sweep = None
  if experiment == 'models':
    config.learning_rate = 0.05
    config.batch_size = 100
  elif experiment == 'conformal':
    config.mode = 'conformal'
    config.conformal.coverage_loss = 'none'
    config.conformal.loss_transform = 'log'
    config.conformal.size_transform = 'identity'
    config.conformal.rng = False

    config.learning_rate = 0.01
    config.batch_size = 100
    config.conformal.temperature = 1.
    config.conformal.size_loss = 'valid'
    config.conformal.method = 'threshold_logp'
    config.conformal.size_weight = 0.01
  else:
    raise ValueError('Experiment not implemented.')
  return config, parameter_sweep
