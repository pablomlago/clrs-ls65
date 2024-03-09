# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import haiku as hk
import jax.numpy as jnp
from jax import random

def sample_msgs(msgs, adj_mat, num_samples_per_node, seed=None):
    """Processor inference step.

    Args:
      msgs: Messages in layer.
      adj_mat: Graph adjacency matrix.
      num_samples_per_node: Number of messages to sample for each node.

    Returns:
      Message samples.
    """

    b, n, _, h = msgs.shape

    if seed is not None:
      key = random.PRNGKey(seed)  # PRNG key for reproducibility
      key, subkey = random.split(key)
    else:
       subkey = hk.next_rng_key()
    # TODO: Ensure that the messages come from nodes in the neighbourhood

    # Generate random indices
    # random_indices shape will be [B, N, 2], with values in range [0, N)
    random_indices = random.randint(subkey, (b, num_samples_per_node, n), minval=0, maxval=n)
    # You need to create a meshgrid for B and N dimensions to use in advanced indexing
    sampled_msgs = jnp.take_along_axis(msgs, jnp.expand_dims(random_indices, axis=-1), axis=1)

    return sampled_msgs

def sample_nodes(batch_size, num_nodes, adj_mat, num_samples, seed=None):
    """Processor inference step.

    Args:
      node_hiddens: Hiddens for the nodes
      adj_mat: Graph adjacency matrix.
      num_samples_per_node: Number of nodes to sample

    Returns:
      Sampled nodes
    """
    if seed is not None:
      key = random.PRNGKey(seed)  # PRNG key for reproducibility
      key, subkey = random.split(key)
    else:
       subkey = hk.next_rng_key()
    # TODO: Ensure that the messages come from nodes in the neighbourhood

    # Generate random indices
    # random_indices shape will be [B, num_samples], with values in range [0, N)
    mask_1 = random.randint(subkey, (batch_size, num_samples), minval=0, maxval=num_nodes)

    # Change the seed again, otherwise the same nodes as in mask_1 will be sampled
    if seed is not None:
      key = random.PRNGKey(seed)  # PRNG key for reproducibility
      key, subkey = random.split(key)
    else:
      subkey = hk.next_rng_key()
    # TODO: Ensure that the messages come from nodes in the neighbourhood

    mask_2 = random.randint(subkey, (batch_size, num_samples), minval=0, maxval=num_nodes)

    return mask_1, mask_2