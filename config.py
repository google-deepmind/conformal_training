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
  # Defines groups of classes for reducing mis-coverage using the
  # classification loss:
  # (see experiments/run_mnist.py for examples)
  config.class_groups = ()
  # Confidence level to use during training:
  config.alpha = 0.01
  # The target alpha to enforce using the coverage loss, mostly
  # relevant for coverage training:
  config.target_alpha = 0.01
  # Temperature for soft thresholding:
  config.temperature = 1.
  # Dispersion for smooth/differentiable sorting:
  config.dispersion = 0.1
  # Weight of the inefficiency loss:
  config.size_weight = 1.
  # Which coverage loss to use, see train_coverage.py for options.
  config.coverage_loss = 'none'
  # Loss matrix to use in the classification loss:
  # (see experiments/run_fashion_mnist.py for examples)
  config.loss_matrix = ()
  # Optional cross-entropy loss in addition to inefficiency/classification
  # loss:
  config.cross_entropy_weight = 0.
  # Which size loss to use, mainly valid or normal:
  config.size_loss = 'valid'
  # Loss transform, usually identity or log:
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

  # Architecture: mlp, cnn or resnet.
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

  # Optimizer: sgd or adam.
  config.optimizer = 'sgd'
  config.adam = collections.ConfigDict()
  config.adam.b1 = 0.9
  config.adam.b2 = 0.999
  config.adam.eps = 1e-8
  config.sgd = collections.ConfigDict()
  config.sgd.momentum = 0.9
  config.sgd.nesterov = True

  # Learning rate schedules:
  config.learning_rate_schedule = 'exponential'
  config.step = collections.ConfigDict()
  config.step.learning_rate_decay = 0.1
  config.exponential = collections.ConfigDict()
  config.exponential.learning_rate_decay = 0.5

  # Training mode: normal, coverage or conformal:
  config.mode = 'normal'

  config.coverage = get_conformal_config()
  # Fixed threshold for coverage training:
  config.coverage.tau = 1.
  # When fine-tuning a model, fix threshold tau based on that many
  # batches of the training set:
  config.coverage.calibration_batches = 10

  config.conformal = get_conformal_config()
  # Fraction of each batch to use for (smooth) calibration.
  config.conformal.fraction = 0.5

  # General learning hyper-parameters:
  config.learning_rate = 0.01
  config.momentum = 0.9
  config.weight_decay = 0.0005
  config.batch_size = 500
  config.test_batch_size = 100
  config.epochs = 10

  config.finetune = collections.ConfigDict()
  config.finetune.enabled = False
  # Also update/fine-tune model state:
  config.finetune.model_state = True
  # Which layers to fine-tune:
  config.finetune.layers = 'batch_norm_1,linear_2'
  # Whether to re-initialize selected layers or not:
  config.finetune.reinitialize = True

  # This is the path from which the model-to-be-fine-tuned will be loaded:
  config.finetune.path = './test/'
  # Path to save checkpoints and final predictions to:
  config.path = './test/'

  config.seed = 0
  config.checkpoint_frequency = 5
  config.resampling = 0
  config.whitening = True
  config.cifar_augmentation = 'standard'
  # How many validation examples to take from the training examples::
  config.val_examples = 5000

  config.dataset = 'mnist'

  config.jit = False

  return config
