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

"""Tests sorting nets."""
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp

from conformal_training import sorting_nets
from conformal_training import variational_sorting_net


class SortingNetsTest(parameterized.TestCase):

  @parameterized.parameters([
      [4],
  ])
  def test_create_comms(self, length):
    comms = sorting_nets.comm_pattern_bitonic(2)
    chex.assert_equal(comms["alg"], "bitonic")

    comms = sorting_nets.comm_pattern_batcher(length, make_parallel=True)
    chex.assert_equal(comms["alg"], "batcher-bitonic")

    comms = sorting_nets.comm_pattern_batcher(length, make_parallel=False)
    chex.assert_equal(comms["alg"], "batcher-bitonic")

  @parameterized.parameters([
      [[[[0, 1]], [[1, 2]], [[0, 2]]], 3]
  ])
  def test_comm_pattern_from_list(self, snet_list, num_stages):
    comms = sorting_nets.comm_pattern_from_list(snet_list)
    chex.assert_equal(comms["alg"], "fixed")
    chex.assert_equal(comms["num_stages"], num_stages)

  @parameterized.parameters([
      [[[[0, 1]], [[2, 3]], [[0, 2]]], 2]
  ])
  def test_parallelize(self, snet_list, final_len):
    snet_par = sorting_nets.parallelize(snet_list)
    chex.assert_equal(len(snet_par), final_len)

    comms = sorting_nets.comm_pattern_from_list(snet_par)
    chex.assert_equal(comms["alg"], "fixed")
    chex.assert_equal(comms["num_wires"], 4)
    chex.assert_equal(comms["num_stages"], 2)
    chex.assert_equal(comms["num_comparators"], 3)

  def test_prune(self):
    snet_list = sorting_nets.SNET_10
    snet_pruned = sorting_nets.prune(snet_list, keep=[9])

    comms = sorting_nets.comm_pattern_from_list(snet_pruned, make_parallel=True)
    chex.assert_equal(comms["alg"], "fixed")
    chex.assert_equal(comms["num_wires"], 10)
    chex.assert_equal(comms["num_stages"], 4)
    chex.assert_equal(comms["num_comparators"], 9)

    k_top = 2
    length = comms["num_wires"]
    keep = list(range(length-1, length -1 - k_top - 1, -1))
    pruned_list = sorting_nets.prune(snet_list, keep=keep)
    comms = sorting_nets.comm_pattern_from_list(pruned_list, make_parallel=True)
    bs = variational_sorting_net.VariationalSortingNet(comms)

    prng_key = jax.random.PRNGKey(1)
    x = jax.random.uniform(prng_key, [length])
    xh, _ = bs.sort_tester(x, dispersion=0.1)
    x_sort = jnp.sort(x)

    chex.assert_equal(xh[-1], x_sort[-1])
    chex.assert_equal(xh[-2], x_sort[-2])

if __name__ == "__main__":
  absltest.main()
