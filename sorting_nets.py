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

"""Module for constructing sorting networks."""
import numpy as np
jnp = np

SNET_10 = [[[0, 1], [3, 4], [5, 6], [8, 9]],
           [[2, 4], [7, 9]],
           [[2, 3], [1, 4], [7, 8], [6, 9]],
           [[0, 3], [5, 8], [4, 9]],
           [[0, 2], [1, 3], [5, 7], [6, 8]],
           [[1, 2], [6, 7], [0, 5], [3, 8]],
           [[1, 6], [2, 7], [4, 8]],
           [[1, 5], [3, 7]],
           [[4, 7], [2, 5], [3, 6]],
           [[4, 6], [3, 5]],
           [[4, 5]]]


def comm_pattern_bitonic(num_bits):
  """Bitonic sort communication pattern on a hypercube of size 2**num_bits.

  Args:
    num_bits: size of the array to be sorted is 2**num_bits
  Returns:
    comms: Catalog
  """

  total_stages = num_bits*(num_bits+1)//2
  edge_list = []
  absolute_substage = 0
  for stage in range(num_bits):
    for substage in range(stage+1):
      i = np.arange(2**(num_bits-stage+substage-1))
      j = np.arange(2**(stage-substage))
      idx1 = jnp.reshape(
          i.reshape((i.shape[0], 1))*2**(stage-substage+1)
          + j.reshape((1, j.shape[0])), (i.shape[0]*j.shape[0]))
      idx2 = idx1 + 2**(stage-substage)
      direction = (idx1 // (2**(stage+1))) % 2
      edges = np.zeros([2**(num_bits-1), 2], dtype=np.int32)
      edges[:, 0] = np.where(direction == 0, idx1, idx2)
      edges[:, 1] = np.where(direction == 0, idx2, idx1)
      edge_list.append(jnp.array(edges))
      absolute_substage += 1

  return {"alg": "bitonic",
          "num_wires": 2**num_bits,
          "num_stages": total_stages,
          "num_comparators": total_stages*(2**(num_bits-1)),
          "edge_list": edge_list}


def comm_pattern_from_list(snet_list, make_parallel=False):
  """A fixed network from a list of comperators.

  Args:
    snet_list: List of stages. stages is also a list of edges
    make_parallel: (Optional) Organize parallel exeecutable comparators
  Returns:
    comms: Catalog. We make sure that edge_list is in sorted form
  """
  if make_parallel:
    snet_list = parallelize(snet_list)
  total_stages = len(snet_list)
  edge_list = []
  max_wire_seen = 0
  comp_count = 0
  for a in snet_list:
    v = np.array(a)
    max_wire_seen = max(max_wire_seen, np.max(v))
    comp_count = comp_count + v.shape[0]
    idx = np.argsort(v[:, 0])
    edge_list.append(jnp.array(v[idx, :]))

  return {"alg": "fixed",
          "num_wires": max_wire_seen+1,
          "num_stages": total_stages,
          "num_comparators": comp_count,
          "edge_list": edge_list}


def prune(snet_list, keep):
  """Prune comparators not used for wires in keep."""
  keep = set(keep)
  pruned_list = [[]]
  for stage in reversed(snet_list):
    if pruned_list[0]:
      pruned_list.insert(0, [])
    for edge in stage:
      if (edge[0] in keep) or (edge[1] in keep):
        keep.update(edge)
        pruned_list[0].append(edge)
  return pruned_list


def parallelize(snet_lst):
  """Organize comparators that can be run in parallel in stages.

  We visit each comparator in the sequence and try to place it
  to the earliest stage by starting from the last stage constructed.

  Args:
    snet_lst: List of sorting network stages (that are lists of edges)
  Returns:
    stage: Rearanged comparators as stages
  """

  stage_sets = [set()]
  stage = [[]]
  for edge_lst in snet_lst:
    for edge in edge_lst:
      placed = False
      place_here = len(stage)-1
      for stage_idx in reversed(range(len(stage))):
        if ((edge[0] not in stage_sets[stage_idx])
            and (edge[1] not in stage_sets[stage_idx])):
          place_here = stage_idx
          placed = True
        else:
          break
      if not placed:
        stage.append([edge])
        stage_sets.append(set(edge))
      else:
        stage[place_here].append(edge)
        stage_sets[place_here].update(edge)
  return stage


def generate_list_bitonic(length, make_parallel=True):
  """Generate a Bitonic sorting network list of arbitrary length.

  Args:
    length: Number of wires
    make_parallel: Flag to organize parallel executable comparators into stages
  Returns:
    snet_list: list of pairwise swaps

  """
  def greatest_power_of_two_less_than(n):
    k = 1
    while k > 0 and k < n:
      k = k * 2
    return k // 2

  def bitonic_sort(lo, n, direction):
    if n > 1:
      m = n // 2
      bitonic_sort(lo, m, not direction)
      bitonic_sort(lo+m, n-m, direction)
      bitonic_merge(lo, n, direction)

  def bitonic_merge(lo, n, direction):
    if n > 1:
      m = greatest_power_of_two_less_than(n)
      for i in range(lo, lo+n-m):
        if direction:
          snet_list.append([[i, i+m]])
        else:
          snet_list.append([[i+m, i]])
      bitonic_merge(lo, m, direction)
      bitonic_merge(lo+m, n-m, direction)

  snet_list = []
  bitonic_sort(0, length, True)
  return parallelize(snet_list) if make_parallel else snet_list


def comm_pattern_batcher(length, make_parallel=True):
  """Batcher bitonic communication pattern for an array with size length."""
  snet_list = generate_list_bitonic(length, make_parallel)
  comms = comm_pattern_from_list(snet_list)
  comms["alg"] = "batcher-bitonic"
  return comms
