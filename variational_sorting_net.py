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

"""Variational sorting networks."""
import functools
import jax
import jax.numpy as jnp


def _swap_prob_hard(x1, x2):
  return jnp.array(jnp.greater(x1, x2), dtype=jnp.float32)

_DELTA_THRESHOLD_EXPECTED = 0.001
_DELTA_THRESHOLD_SAMPLE = 0.001
_EPS = 1e-9


def _swap_prob_entropy_reg(x1, x2, dispersion=1.0):
  """Swapping probability, entropy regularization."""
  d = 2 * jax.nn.relu((x2-x1))/dispersion
  d2 = 2*jax.nn.relu((x1-x2))/dispersion
  return jnp.exp(d2 - jnp.logaddexp(d, d2))


def _swap_prob_entropy_reg_l2(x1, x2, dispersion=1.0):
  """Swapping probability, entropy regularization."""
  d = 2*jnp.square(jax.nn.relu(x2-x1))/dispersion
  d2 = 2*jnp.square(jax.nn.relu(x1-x2))/dispersion
  return jnp.exp(d2 - jnp.logaddexp(d, d2))


def _swap_prob_entropy_reg_lp(x1, x2, dispersion=1.0, norm_p=1.0):
  """Swapping probability, entropy regularization."""
  d = 2*jnp.power(jax.nn.relu(x2-x1), norm_p)/dispersion
  d2 = 2*jnp.power(jax.nn.relu(x1-x2), norm_p)/dispersion
  return jnp.exp(d2 - jnp.logaddexp(d, d2))


def butterfly(lam, x1, x2):
  return lam*x2+(1-lam)*x1, lam*x1+(1-lam)*x2


def forward_step(
    x,
    stage_idx,
    comms,
    dispersion=1.0,
    swap_prob_fun=_swap_prob_entropy_reg,
    hard_swap_prob_fun=_swap_prob_hard,
    key=None):
  """Computes swapping probabilities at stage_idx of the sorting network."""

  idx1 = comms["edge_list"][stage_idx][:, 0]
  idx2 = comms["edge_list"][stage_idx][:, 1]

  x1, x2 = butterfly(hard_swap_prob_fun(x[idx1], x[idx2]), x[idx1], x[idx2])
  if key is None:
    lam = swap_prob_fun(x[idx1], x[idx2], dispersion)
  else:
    subkey = jax.random.split(key, comms["edge_list"][stage_idx].shape[0])
    lam = swap_prob_fun(subkey, x[idx1], x[idx2], dispersion)

  x = x.at[idx1].set(x1, indices_are_sorted=True)
  x = x.at[idx2].set(x2, indices_are_sorted=True)

  return x, lam


def backward_step(u, stage_idx, comms, lam):
  """Executes in parallel stage_idx of the sorting network."""

  idx1 = comms["edge_list"][stage_idx][:, 0]
  idx2 = comms["edge_list"][stage_idx][:, 1]

  if len(u.shape) > 1:
    u1, u2 = butterfly(jnp.reshape(lam, (lam.shape[0], 1)),
                       u[idx1, :], u[idx2, :])
    u = u.at[idx1, :].set(u1, indices_are_sorted=True)
    u = u.at[idx2, :].set(u2, indices_are_sorted=True)
  else:
    u1, u2 = butterfly(lam, u[idx1], u[idx2])
    u = u.at[idx1].set(u1, indices_are_sorted=True)
    u = u.at[idx2].set(u2, indices_are_sorted=True)

  return u


def forward_only_step(
    x, v,
    stage_idx,
    comms,
    dispersion=1.0,
    swap_prob_fun=_swap_prob_entropy_reg,
    hard_swap_prob_fun=_swap_prob_hard,
    key=None):
  """Executes in parallel stage_idx of the sorting network."""

  idx1 = comms["edge_list"][stage_idx][:, 0]
  idx2 = comms["edge_list"][stage_idx][:, 1]

  x1, x2 = butterfly(hard_swap_prob_fun(x[idx1], x[idx2]), x[idx1], x[idx2])
  if key is None:
    lam = swap_prob_fun(x[idx1], x[idx2], dispersion)
  else:
    subkey = jax.random.split(key, comms["edge_list"][stage_idx].shape[0])
    lam = swap_prob_fun(subkey, x[idx1], x[idx2], dispersion)

  x = x.at[idx1].set(x1, indices_are_sorted=True)
  x = x.at[idx2].set(x2, indices_are_sorted=True)

  if len(v.shape) > 1:
    v1, v2 = butterfly(jnp.reshape(lam, (lam.shape[0], 1)),
                       v[idx1, :], v[idx2, :])
    v = v.at[idx1, :].set(v1, indices_are_sorted=True)
    v = v.at[idx2, :].set(v2, indices_are_sorted=True)
  else:
    v1, v2 = butterfly(lam, v[idx1], v[idx2])
    v = v.at[idx1].set(v1, indices_are_sorted=True)
    v = v.at[idx2].set(v2, indices_are_sorted=True)
  return x, v, lam


def costfun(target_vec, initial_vec, norm_p=None):
  """Computes pairwise p-norm between entries of a vector.

  Given two vectors y, x, this function computes

  Args:
    target_vec: y vector (corresponds to the columns)
    initial_vec: x vector (corresponds to the rows)
    norm_p: norm parameter (Default=1, Euclidian (square distance)=2)
  Returns:
    costmat: a matrix C with entries C_ij = |y_i - x_j|^p
  """
  dist = (jnp.reshape(target_vec, (target_vec.shape[0], 1))
          - jnp.reshape(initial_vec, (1, initial_vec.shape[0])))

  if norm_p is None or norm_p == 1:
    return jnp.abs(dist)
  elif norm_p == 2:
    return jnp.square(dist)
  else:
    return jnp.power(jnp.abs(dist), norm_p)


def permutation_entropy(perm):
  """Entropy of a soft permutation matrix.

  Args:
    perm : Soft permutation with marginals equal to the ones vector.
  Returns:
    entropy: H_n[P] = -sum_{ij} P_{ij} log P_{ij} + n log(n)
  """
  length = perm.shape[0]
  neg_entr = jnp.where(jnp.greater(perm, _EPS) * jnp.less(perm, 1.0-_EPS),
                       perm*jnp.log(perm), 0.0)

  entropy = -jnp.sum(neg_entr) + length*jnp.log(length)
  return entropy


def permutation_elbo(perm, x, dispersion, norm_p=None, target_vec=None):
  if target_vec is None:
    target_vec = jnp.sort(x)
  cost_matrix = costfun(target_vec, x, norm_p=norm_p) / dispersion

  fidelity = - jnp.trace(cost_matrix.T.dot(perm))
  entropy = permutation_entropy(perm)
  elbo = fidelity + entropy
  return elbo, fidelity, entropy


class VariationalSortingNet(object):
  """Class for efficient and differentiable order statistics."""

  def __init__(
      self, comms,
      smoothing_strategy="entropy_reg",
      sorting_strategy="hard",
      sorting_dispersion=0.001,
      norm_p=1):
    """Generate a sorting network that sort the input vector and values.

    Args:
      comms: Communication pattern (obtained via sorting_nets.comms* functions)
      smoothing_strategy: How to sort the values.
                          (default="entropy_reg")
      sorting_strategy: How to sort the keys. {hard, entropy_reg}
                        (default="hard")
      sorting_dispersion: Dispersion parameter to sort the input vector x.
                          (default=0.001)
                          Only used when sorting_strategy is not hard
      norm_p: norm to use for the cost function (default=1)
    """
    assert smoothing_strategy in ["entropy_reg"]
    assert sorting_strategy in ["hard", "entropy_reg"]
    assert norm_p > 0

    if norm_p == 1 or norm_p is None:
      norm_choice = 1
    elif norm_p == 2:
      norm_choice = 2
    else:
      norm_choice = 0

    self.comms = comms
    if smoothing_strategy == "entropy_reg":
      funcs = [functools.partial(_swap_prob_entropy_reg_lp, norm_p=norm_p),
               _swap_prob_entropy_reg,
               _swap_prob_entropy_reg_l2]
      swap_prob_fun = funcs[norm_choice]
      self._is_sampler = False

    if sorting_strategy == "hard":
      hard_swap_prob_fun = _swap_prob_hard
    elif sorting_strategy == "entropy_reg":
      hard_swap_prob_fun = functools.partial(
          _swap_prob_entropy_reg, dispersion=sorting_dispersion)

    if self._is_sampler:
      self.stage_fwd_only = functools.partial(
          forward_only_step, swap_prob_fun=swap_prob_fun,
          hard_swap_prob_fun=hard_swap_prob_fun)
      self.stage_fwd = functools.partial(
          forward_step, swap_prob_fun=swap_prob_fun,
          hard_swap_prob_fun=hard_swap_prob_fun)
    else:
      self.stage_fwd_only = functools.partial(
          forward_only_step, swap_prob_fun=swap_prob_fun,
          hard_swap_prob_fun=hard_swap_prob_fun, key=None)
      self.stage_fwd = functools.partial(
          forward_step, swap_prob_fun=swap_prob_fun,
          hard_swap_prob_fun=hard_swap_prob_fun, key=None)

  def forward_only(
      self, x, v, u=None, dispersion=1.,
      lower=0, upper=None, key=None):
    r"""Evaluate order statistics u^\top P(x) v by forward only propagation.

      This function should be preferred over forward_backward when implementing
      cost functions for large models.
    Args:
      x : Input vector that determines the soft permutation P that approximately
          brings x into sorted ordeer
      v : Values to be smoothly sorted
      u : (Optional) test vector, default = identity
      dispersion : Smoothing parameter
      lower : Index of the first stage of the sorting network to start the sort
      upper : Final stage to finish the sort
      key: (optional) Random seed to use for the forward sampling algorithm
    Returns:
      x_sorted : hard sorted vectors
      orderstats : Result of u^\top P(x) v
    """
    assert self.comms["num_wires"] == x.shape[0]

    if upper is None:
      upper = self.comms["num_stages"]

    if not self._is_sampler:
      for i in range(lower, upper):
        x, v, _ = self.stage_fwd_only(x, v, i,
                                      self.comms, dispersion=dispersion)
    else:
      subkey = jax.random.split(key, upper-lower)
      for i in range(lower, upper):
        x, v, _ = self.stage_fwd_only(x, v, i,
                                      self.comms,
                                      dispersion=dispersion,
                                      key=subkey[i])

    if u is None:
      return x, v
    else:
      return x, u.T.dot(v)

  def forward_backward(
      self, x, u,
      v=None, dispersion=1.,
      lower=0, upper=None, key=None):
    r"""Evaluate order statistics u^\top P(x) v by forward-backward.

      This function should be avoided when implementing cost functions for
      large models, as it stores swapping probabilities. Use forward_only to be
      preferred.
    Args:
      x : Input vector that determines the soft permutation P that approximately
          brings x into sorted order
      u : Test vector to be transformed by transpose(P(x))
      v : (Optional) Values to be sorted, default = identity
      dispersion : Smoothing parameter
      lower : Index of the first stage of the sorting network to start the sort
      upper : Final stage to finish the sort
      key: (optional) Random seed to use for the forward sampling algorithm
    Returns:
      x_sorted : hard sorted vectors
      orderstats : Result of u^\top P(x) v
      lambdas : Structure containing the swap probabilities
    """
    assert self.comms["num_wires"] == x.shape[0]

    if upper is None:
      upper = self.comms["num_stages"]

    # forward pass
    lambdas = []

    if not self._is_sampler:
      for i in range(lower, upper):
        x, lam = self.stage_fwd(x, i, self.comms, dispersion=dispersion)
        lambdas.append(lam)

    else:
      subkey = jax.random.split(key, upper-lower)
      for i in range(lower, upper):
        x, lam = self.stage_fwd(x, i,
                                self.comms,
                                dispersion=dispersion,
                                key=subkey[i])
        lambdas.append(lam)
    # Backward pass.
    for i in reversed(range(lower, upper)):
      u = backward_step(u, i, self.comms, lambdas[i-lower])

    if v is None:
      return x, u.T, lambdas
    else:
      return x, u.T.dot(v), lambdas

  def sort(self, x, dispersion, key=None):
    """Smooth sort."""
    _, x_ss = self.forward_only(x, x, dispersion=dispersion, key=key)
    return x_ss

  def sort_tester(self, x, dispersion, key=None):
    """Smooth sort."""
    xh, x_ss = self.forward_only(x, x, dispersion=dispersion, key=key)
    return xh, x_ss

  def ismax(self, x, dispersion, key=None):
    r"""Probabilities that maximum of x is x[i] for i=0..len(x)-1."""
    length = self.comms["num_wires"]
    u = jax.nn.one_hot(length-1, length)
    _, res, _ = self.forward_backward(x, u=u, dispersion=dispersion, key=key)
    return res

  def max(self, x, dispersion, key=None):
    length = self.comms["num_wires"]
    u = jax.nn.one_hot(length-1, length)
    _, x_ss = self.forward_only(x, x, u=u, dispersion=dispersion, key=key)
    return x_ss

  def ismin(self, x, dispersion, key=None):
    r"""Probabilities that minimum of x is x[i] for i=0..len(x)-1."""
    length = self.comms["num_wires"]
    u = jax.nn.one_hot(0, length)
    _, res, _ = self.forward_backward(x, u=u, dispersion=dispersion, key=key)
    return res

  def min(self, x, dispersion, key=None):
    length = self.comms["num_wires"]
    u = jax.nn.one_hot(0, length)
    _, x_ss = self.forward_only(x, x, u=u, dispersion=dispersion, key=key)
    return x_ss

  def isquantile(self, x, dispersion, alpha=0.5, tau=0.5, key=None):
    r"""Probabilities that the alpha quantile of x is x[i] for i=0..len(x)-1."""
    length = self.comms["num_wires"]
    idx1 = jnp.floor(alpha * (length-1))
    idx2 = jnp.ceil(alpha * (length-1))
    u = tau * jax.nn.one_hot(idx2, length)
    u += (1 - tau) * jax.nn.one_hot(idx1, length)
    _, res, _ = self.forward_backward(x, u=u, dispersion=dispersion, key=key)
    return res

  def quantile(self, x, dispersion, alpha=0.5, tau=0.5, key=None):
    """Retrieves the smoothed alpha quantile."""
    length = self.comms["num_wires"]
    idx1 = jnp.floor(alpha * (length-1))
    idx2 = jnp.ceil(alpha * (length-1))
    u = tau * jax.nn.one_hot(idx2, length)
    u += (1 - tau) * jax.nn.one_hot(idx1, length)
    _, x_ss = self.forward_only(x, x, u=u, dispersion=dispersion, key=key)
    return x_ss

  def ismedian(self, x, dispersion, tau=0.5, key=None):
    r"""Probabilities that the median of x is x[i] for i=0..len(x)-1.

    Args:
      x : jnp.array to be sorted
      dispersion: Smoothing parameter
      tau: an arbitrary parameter in [0, 1] for resolving ties
      key: seed (used only if self.is_sampler is true)
    Returns:
      result: median of x
    """
    return self.isquantile(x, dispersion=dispersion, alpha=0.5,
                           tau=tau, key=key)

  def median(self, x, dispersion, tau=0.5, key=None):
    """Retrieves the smoothed median."""
    return self.quantile(x, dispersion, alpha=0.5, tau=tau, key=key)

  def istopk(self, x, dispersion, topk, key=None):
    """Smooth discrete distribution with a mode highest k entries."""
    length = self.comms["num_wires"]
    u = jnp.sum(jax.nn.one_hot(range(length-1, length-topk-1, -1), length),
                axis=0)
    _, res, _ = self.forward_backward(x, u=u, dispersion=dispersion, key=key)
    return res

  def sortperm(self, x, dispersion, key=None, full_output=False):
    """Smoothed sorting permutation of x."""
    length = self.comms["num_wires"]
    u = jnp.eye(length)
    if full_output:
      xh, res, lambdas = self.forward_backward(x, u=u, dispersion=dispersion,
                                               key=key)
      return xh, res, lambdas
    else:
      _, res, _ = self.forward_backward(x, u=u, dispersion=dispersion, key=key)
      return res

  def subperm(self, x, dispersion, idx, from_top=False, key=None):
    """Retrieves a subset of the sorting permutation.

    Args:
      x : jnp.array to be sorted
      dispersion: Smoothing parameter
      idx: Indices of columns in an arbitrary order
      from_top: Flag to interpret idx (default=False).
          When from_top == True, the maximum is retrieved with idx = [0]
          When from_top == False, the maximum is retrieved with idx = [length-1]
      key: seed (used only if self.is_sampler is true)
    Returns:
      res: Result of running the order statistics.
    """
    length = self.comms["num_wires"]
    if from_top:
      u = jnp.flipud(jnp.eye(length)[:, idx])
    else:
      u = jnp.eye(length)[:, idx]
    _, res, _ = self.forward_backward(x, u=u, dispersion=dispersion, key=key)
    return res

  def log_likelihood_max(self, x, v, dispersion, output_log_scale=True):
    if output_log_scale:
      return jnp.log(_EPS + self.ismax(x, dispersion).dot(v))
    else:
      return self.ismax(x, dispersion).dot(v)

  def log_likelihood_order(self, x, order, dispersion):
    target_perm = jax.nn.one_hot(order, len(order), dtype=jnp.float32).T
    inner = jnp.diag(self.sortperm(x, dispersion).dot(target_perm))
    return jnp.sum(jnp.log(_EPS + inner))
