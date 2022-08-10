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

"""Configuration for training."""

import ml_collections as collections


def get_conformal_config() -> collections.ConfigDict:
  """Default configuration for coverage and conformal training."""
  config = collections.ConfigDict()

  config.method = 'threshold_logp'
  config.class_groups = ()
  config.alpha = 0.01
  config.target_alpha = 0.01
  config.temperature = 1.
  config.dispersion = 0.1
  config.size_weight = 1.
  config.coverage_loss = 'none'
  config.loss_matrix = ()
  config.cross_entropy_weight = 0.
  config.size_loss = 'valid'
  config.size_transform = 'identity'
  config.size_bound = 3.
  config.size_bound_weight = 0.9
  config.size_weights = ()
  config.loss_transform = 'log'
  config.rng = False

  return config


def get_config() -> collections.ConfigDict:
  """Default configuration for training.

  Returns:
    Configuration as ConfigDict.
  """
  config = collections.ConfigDict()
  config.architecture = 'mlp'

  config.cnn = collections.ConfigDict()
  config.cnn.channels = 32
  config.cnn.layers = 3
  config.cnn.kernels = 3
  config.cnn.activation = 'relu'

  config.mlp = collections.ConfigDict()
  config.mlp.units = 32
  config.mlp.layers = 0
  config.mlp.activation = 'relu'

  config.resnet = collections.ConfigDict()
  config.resnet.version = 34
  config.resnet.channels = 4
  config.resnet.resnet_v2 = True
  config.resnet.init_logits = True

  config.optimizer = 'sgd'

  config.adam = collections.ConfigDict()
  config.adam.b1 = 0.9
  config.adam.b2 = 0.999
  config.adam.eps = 1e-8

  config.sgd = collections.ConfigDict()
  config.sgd.momentum = 0.9
  config.sgd.nesterov = True

  config.learning_rate_schedule = 'exponential'

  config.step = collections.ConfigDict()
  config.step.learning_rate_decay = 0.1

  config.exponential = collections.ConfigDict()
  config.exponential.learning_rate_decay = 0.5

  config.mode = 'normal'

  config.coverage = get_conformal_config()
  config.coverage.tau = 1.
  config.coverage.calibration_batches = 10

  config.conformal = get_conformal_config()
  config.conformal.fraction = 0.5

  config.learning_rate = 0.01
  config.momentum = 0.9
  config.nesterov = True
  config.weight_decay = 0.0005
  config.batch_size = 500
  config.test_batch_size = 100
  config.epochs = 10

  config.finetune = collections.ConfigDict()
  config.finetune.enabled = False
  config.finetune.model_state = True
  config.finetune.layers = 'batch_norm_1,linear_2'
  config.finetune.reinitialize = True

  # This is the path from which the model-to-be-fine-tuned will be loaded:
  config.finetune.path = './test/'
  config.path = './test/'

  config.seed = 0
  config.checkpoint_frequency = 5
  config.resampling = 0
  config.whitening = True
  config.cifar_augmentation = 'standard'
  config.val_examples = 5000

  config.dataset = 'mnist'

  config.jit = False

  return config
