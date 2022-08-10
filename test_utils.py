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

"""Utilities to avoid redundancy in tests."""
from typing import Optional

import jax.numpy as jnp
import numpy as np
import scipy.special


def get_labels(num_examples: int, num_classes: int) -> jnp.ndarray:
  """Get random labels.

  Args:
    num_examples: number of examples
    num_classes: number of classes

  Returns:
    Labels
  """
  return jnp.array(np.random.randint(0, num_classes, (num_examples)))


def get_probabilities(
    labels: jnp.ndarray, dominance: float,
    log: Optional[bool] = False) -> jnp.ndarray:
  """Get random probabilities where the logit of the true label dominates.

  Args:
    labels: labels to generate probabilities for
    dominance: float value added to the logit of the true label before
      applying softmax; determines whether probability of true class is the
      largest
    log: return log-probabilities

  Returns:
    Probabilities
  """
  probabilities = np.random.random((labels.shape[0], np.max(labels) + 1))
  probabilities[np.arange(probabilities.shape[0]), labels] += dominance
  probabilities = scipy.special.softmax(probabilities, axis=1)
  if log:
    probabilities = np.log(probabilities)
  return jnp.array(probabilities)
