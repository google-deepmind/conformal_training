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

"""Tests for variational sorting networks."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

import sorting_nets
import variational_sorting_net


class VariationalSortingNetTest(parameterized.TestCase):

  @parameterized.parameters([
      [16, "entropy_reg", "hard", 12],
      [16, "entropy_reg", "entropy_reg", 12],
  ])
  def test_sort(self, length, smoothing_strategy, sorting_strategy, prng_key):

    dispersion = 0.05
    key = jax.random.PRNGKey(prng_key)
    subkey, key = jax.random.split(key)
    x = jax.random.uniform(subkey, shape=(length,))*5

    snets = {
        "batcher-bitonic": sorting_nets.comm_pattern_batcher(
            length, make_parallel=True)
    }

    for sn in snets:
      bs = variational_sorting_net.VariationalSortingNet(
          snets[sn], smoothing_strategy=smoothing_strategy,
          sorting_strategy=sorting_strategy)

      x_hard, _ = bs.sort_tester(x, dispersion=dispersion, key=subkey)

      if sorting_strategy == "hard":
        x_sorted = jnp.sort(x)
        assert jnp.abs(x_hard[-1] - x_sorted[-1]) < 1e-6

  @parameterized.parameters([
      [2],
      [5],
  ])
  def test_jacobian(self, log2_length):
    length = 2 ** log2_length
    snet = sorting_nets.comm_pattern_bitonic(log2_length)
    bs = variational_sorting_net.VariationalSortingNet(
        snet, smoothing_strategy="entropy_reg", sorting_strategy="hard")
    jac_sort = jax.jacrev(bs.sort)
    key = jax.random.PRNGKey(12)
    subkey, key = jax.random.split(key)
    x = jax.random.uniform(subkey, shape=(length,))*5

    jac = jac_sort(x, dispersion=0.1)

    assert jac.shape == (length, length)


if __name__ == "__main__":
  absltest.main()
