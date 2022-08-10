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

"""Utils for open sourcing."""
import os
import pickle
from typing import Any, Tuple, Dict

from absl import logging
import jax.numpy as jnp
import ml_collections as collections


def _dump_pickle(mixed: Any, path: str):
  """Write data to a pickle file."""
  f = open(path, 'wb')
  pickle.dump(mixed, f)
  f.close()
  logging.info('Wrote %s', path)


def _load_pickle(path: str) -> Any:
  """Load data from a pickle file."""
  f = open(path, 'rb')
  mixed = pickle.load(f)
  f.close()
  logging.info('Read %s', path)
  return mixed


class Checkpoint:
  """Checkpoint to save and load models."""

  class State:
    """State holding parameters, model and optimizer state and epoch."""

    def __init__(self):
      """Create a checkpoint state."""
      self.params = None
      """ (FlatMapping) Model parameters. """
      self.model_state = None
      """ (FlatMapping) Model state. """
      self.optimizer_state = None
      """" (List[optax.TraceState] Optimizer state. """
      self.epoch = None
      """ (int) Epoch. """

  def __init__(self, path: str = './'):
    """Create a checkpoint in the provided path.

    Args:
      path: path to checkpoint
    """
    self.state = Checkpoint.State()
    """ (State) State of checkpoint."""
    self.path = path
    """(str) Path to checkpoint."""
    self.params_file = os.path.join(self.path, 'params.pkl')
    """ (str) File to store params. """
    self.model_state_file = os.path.join(self.path, 'model_state.pkl')
    """ (str) File to store params. """
    self.optimizer_state_file = os.path.join(self.path, 'optimizer_state.pkl')
    """ (str) File to store params. """
    self.epoch_file = os.path.join(self.path, 'epoch.pkl')
    """ (str) File to store params. """

  def _exists(self):
    """Check if checkpoint exists.

    Returns:
      true if all checkpoint files were found
    """
    complete_checkpoint = True
    for path in [
        self.params_file, self.model_state_file,
        self.optimizer_state_file, self.epoch_file,
    ]:
      if not os.path.isfile(path):
        complete_checkpoint = False
    return complete_checkpoint

  def restore(self):
    """Restore checkpoint from files."""
    if not self._exists():
      raise ValueError(f'Checkpoint {self.path} not found.')
    self.state.params = _load_pickle(self.params_file)
    self.state.model_state = _load_pickle(self.model_state_file)
    self.state.optimizer_state = _load_pickle(self.optimizer_state_file)
    self.state.epoch = _load_pickle(self.epoch_file)

  def save(self):
    """Save checkpoint to files."""
    os.makedirs(self.path, exist_ok=True)
    _dump_pickle(self.state.params, self.params_file)
    _dump_pickle(self.state.model_state, self.model_state_file)
    _dump_pickle(self.state.optimizer_state, self.optimizer_state_file)
    _dump_pickle(self.state.epoch, self.epoch_file)

  def restore_or_save(self):
    """Restore or save checkpoint."""
    if self._exists():
      self.restore()
    else:
      self.save()


def create_checkpoint(config: collections.ConfigDict) -> Checkpoint:
  """Create a checkpoint.

  Args:
    config: configuration

  Returns:
    Checkpoint.
  """
  return Checkpoint(config.path)


def load_checkpoint(config: collections.ConfigDict) -> Tuple[Checkpoint, str]:
  """Loads the checkpoint using the provided config.path.

  Args:
    config: fine-tuning configuration

  Returns:
    Checkpoint and loaded path
  """
  checkpoint = Checkpoint(config.path)
  checkpoint.restore()
  return checkpoint, config.path


class PickleWriter:
  """Pickle writer to save evaluation."""

  def __init__(self, path: str, name: str):
    """Create a writer.

    Args:
      path: path to directory to write pickle file to
      name: name of pickle file, without extension
    """
    self.path = os.path.join(path, name + '.pkl')
    """ (str) Path to write to."""

  def write(self, values: Any):
    """Write data."""
    _dump_pickle(values, self.path)


def create_writer(config: collections.ConfigDict, key: str) -> Any:
  """Create a writer to save evaluation results.

  Args:
    config: configuration
    key: identifier for writer

  Returns:
    Writer
  """
  return PickleWriter(config.path, key)


class PickleReader:
  """Pickle reader to load evaluation."""

  def __init__(self, path: str, name: str):
    """Create a reader.

    Args:
      path: path to directory to read from
      name: name of pickle file to read, without extension
    """
    self.path = os.path.join(path, name + '.pkl')
    """ (str) Path to write to."""

  def read(self) -> Any:
    """Read pickled data."""
    return _load_pickle(self.path)


def load_predictions(
    path: str, key: str = 'eval', val_examples: int = 0) -> Dict[str, Any]:
  """Load model predictions/logits for specific experiment.

  Args:
    path: path to experiment
    key: evaluation key to load test and val predictions for
    val_examples: number of validation examples used in experiment

  Returns:
    Dictionary containing groups for evaluation and test/val logits/labels
  """
  test_reader = PickleReader(path, f'{key}_test')
  val_reader = PickleReader(path, f'{key}_val')
  eval_test = test_reader.read()

  # Groups are used for evaluation but added optionally later, still
  # need to initialize the dict for everything to work properly.
  model = {
      'data': {'groups': {}},
      'test_logits': eval_test['logits'],
      'test_labels': eval_test['labels'],
      'val_logits': jnp.array([]),
      'val_labels': jnp.array([]),
  }

  test_examples = model['test_labels'].shape[0]
  logging.info('loaded %s: %d test examples', path, test_examples)

  if val_examples > 0:
    eval_val = val_reader.read()
    model['val_logits'] = eval_val['logits']
    model['val_labels'] = eval_val['labels']
    val_examples = model['val_labels'].shape[0]
    logging.info('loaded %s: %d val examples', path, val_examples)

  return model
