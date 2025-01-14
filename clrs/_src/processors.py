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

"""JAX implementation of baseline processor networks."""

import abc
from typing import Any, Callable, List, Optional, Tuple
import functools

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax import random
from jax import lax
import numpy as np

from clrs._src.utils import sample_msgs, sample_nodes

@chex.dataclass
class AsynchronyInformation:
  l2_loss: chex.Array
  l3_cocycle_loss: chex.Array
  l3_multimorphism_loss: chex.Array
  l2_node_update_aggregated: Optional[chex.Array]
  l2_node_update_partial: Optional[chex.Array]
  l3_cocycle_args_update_aggregated: Optional[chex.Array]
  l3_cocycle_args_update_aggregated_partial: Optional[chex.Array]
  l3_multimorphism_msgs_aggregated: Optional[chex.Array]
  l3_multimorphism_msgs_partial: Optional[chex.Array]

def aggregate_asynchrony_information(
    aggregated_asynchrony_information: Optional[AsynchronyInformation], 
    new_asynchrony_information: AsynchronyInformation,
  ) -> AsynchronyInformation:
  # If the aggregated information so far is None, simply return the new information
  if aggregated_asynchrony_information is None:
    return new_asynchrony_information
  # Otherse aggregate information
  return AsynchronyInformation(
    l2_loss=aggregated_asynchrony_information.l2_loss + new_asynchrony_information.l2_loss,
    l3_cocycle_loss=aggregated_asynchrony_information.l3_cocycle_loss + new_asynchrony_information.l3_cocycle_loss,
    l3_multimorphism_loss=aggregated_asynchrony_information.l3_multimorphism_loss + new_asynchrony_information.l3_multimorphism_loss,
    # Embeddings to create L2 asynchrony visualizations
    l2_node_update_aggregated=new_asynchrony_information.l2_node_update_aggregated,
    l2_node_update_partial=new_asynchrony_information.l2_node_update_partial,
    # Embeddings to create L3 cocycle visualisations
    l3_cocycle_args_update_aggregated=new_asynchrony_information.l3_cocycle_args_update_aggregated,
    l3_cocycle_args_update_aggregated_partial=new_asynchrony_information.l3_cocycle_args_update_aggregated_partial,
    # Embeddings to create L3 multimorphism visualisations
    l3_multimorphism_msgs_aggregated=new_asynchrony_information.l3_multimorphism_msgs_aggregated,
    l3_multimorphism_msgs_partial=new_asynchrony_information.l3_multimorphism_msgs_partial,
  )

_Array = chex.Array
_Fn = Callable[..., Any]
BIG_NUMBER = 1e6
PROCESSOR_TAG = 'clrs_processor'

class TropicalLinear(hk.Module):
  """Linear module."""

  def __init__(
      self,
      output_size: int,
      w_init: Optional[hk.initializers.Initializer] = None,
      name: Optional[str] = None,
  ):
    """Constructs a Tropical Linear module.

    Args:
      output_size: Output dimensionality.
      w_init: Optional initializer for weights. By default, uses random values
        from truncated normal, with stddev ``1 / sqrt(fan_in)``. See
        https://arxiv.org/abs/1502.03167v3.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.input_size = None
    self.output_size = output_size
    self.w_init = w_init

  def __call__(
      self,
      inputs: jax.Array,
      *,
      precision: Optional[lax.Precision] = None,
  ) -> jax.Array:
    """Computes a tropical linear transform of the input."""
    if not inputs.shape:
      raise ValueError("Input must not be scalar.")

    input_size = self.input_size = inputs.shape[-1]
    output_size = self.output_size
    dtype = inputs.dtype

    w_init = self.w_init
    if w_init is None:
      stddev = 1. / np.sqrt(self.input_size)
      w_init = hk.initializers.TruncatedNormal(stddev=stddev)
    w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)

    inputs_expanded = jnp.expand_dims(inputs, axis=-1)
    w_expanded = jnp.expand_dims(w, axis=(0,1))

    out = jax.scipy.special.logsumexp(inputs_expanded  + w_expanded, axis=2) 

    return out

########
# Function to compute asynchrony losses
########
def compute_asynchrony_losses(
    num_messages_sample: int, 
    num_nodes_sample: int,
    hidden: _Array, 
    node_args: _Array,
    msgs: _Array, 
    adj_mat: _Array,
    node_fts: _Array,
    edge_fts: _Array,
    graph_fts: _Array,
    msg_reduction_fn: _Fn, 
    arg_reduction_fn: _Fn, 
    node_update_scan_fn: _Fn, 
    arg_update_scan_fn: _Fn,
    msg_generation_fn: _Fn
  ):

  # Extract useful variables
  batch_size, num_nodes, _ = hidden.shape

  #######
  ## Computation of the L2 asynchrony loss: node udpate needs to be a left action
  #######

  # Ensure that there are enough messages, second dimension of hidden corresponds to the number of nodes
  num_messages_sample = min(num_messages_sample, num_nodes)  
  # Randomly sample incoming messages for each node. Resulting shape is [B, num_messages_sample, N, H]
  sampled_msgs = sample_msgs(msgs, adj_mat, num_messages_sample)
  # Apply the reduction over the messages, resulting shape is [B,N,H]
  sampled_msgs_reduced = msg_reduction_fn(sampled_msgs, axis=1)
  # Node update for the aggregated sampled messages (first component of the loss)
  node_update_aggregated, _ = node_update_scan_fn(hidden, sampled_msgs_reduced)
  # Argument update for the aggregated sampled messages (first component of the cocycle loss)
  args_update_aggregated, _ = arg_update_scan_fn(hidden, sampled_msgs_reduced)
  # The shape of sample messages is [B, num_samples_per_node, N, H], and the arguments
  # in jax.lax.scan are passed through the leading dimension
  sampled_msgs = jnp.transpose(sampled_msgs, (1, 0, 2, 3))
  # Iterative updates for the sampled messages
  node_update_partial, node_update_partial_steps = hk.scan(node_update_scan_fn, hidden, sampled_msgs, num_messages_sample)
  # Iterative argument generations for the sampled messages
  args_update_partial_steps, _ = jax.vmap(arg_update_scan_fn, in_axes=0, out_axes=0)(
    jnp.concatenate(
      [
        jnp.expand_dims(hidden, axis=0), # s
        node_update_partial_steps[:-1, ...] # \phi(s,m)
      ],
      axis=0
    ), 
    sampled_msgs
  )
  # Regularisation term penalising the violation of the associativity in the node update
  l2_loss = jnp.mean((node_update_aggregated - node_update_partial)**2)

  #######
  ## Computation of the L3 cocycle loss: assuming argument function is the same as node update
  #######

  # The shape of node_update_partial_steps is [num_messages_sample, B, N, H], so need to
  # reduce over leading dimension
  args_update_aggregated_partial = arg_reduction_fn(args_update_partial_steps, axis=0)
  # Cocycle loss computation
  l3_cocycle_loss = jnp.mean((args_update_aggregated - args_update_aggregated_partial)**2)

  #######
  ## Computation of the L3 multimorphism loss
  #######

  # Ensure that there are enough nodes to sample, second dimension of hidden corresponds to the number of nodes
  num_nodes_sample = min(num_nodes_sample, num_nodes)  
  # Subsample a set of nodes to ensure that the loss computation does not result in a huge overhead
  mask_receivers, mask_senders = sample_nodes(batch_size, num_nodes, adj_mat, num_nodes_sample)

  # Subsample hiddens for senders and receivers - [B,num_nodes_sample,H]
  args_receivers_sample = jnp.take_along_axis(node_args, jnp.expand_dims(mask_receivers, axis=-1), axis=1)
  args_senders_sample = jnp.take_along_axis(node_args, jnp.expand_dims(mask_senders, axis=-1), axis=1)
  
  # Subsample node features for senders and receivers - [B,num_nodes_sample, H]
  node_fts_receivers_sample = jnp.take_along_axis(node_fts, jnp.expand_dims(mask_receivers, axis=-1), axis=1)
  node_fts_senders_sample = jnp.take_along_axis(node_fts, jnp.expand_dims(mask_senders, axis=-1), axis=1)
  
  # Subsample edge features - [B, num_nodes_sample (senders), num_nodes_sample (receivers), H]
  edge_fts_sample = jnp.take_along_axis(edge_fts, jnp.expand_dims(mask_senders, axis=(2,3)), axis=1)
  edge_fts_sample = jnp.take_along_axis(edge_fts_sample, jnp.expand_dims(mask_receivers, axis=(1,3)), axis=2)

  # Subsample in the partial node updates to obtain the partial arguments generated
  # The shape is [num_messages_sample, b, num_nodes_sample, h]
  args_update_partial_steps_receivers = jnp.take_along_axis(args_update_partial_steps, jnp.expand_dims(mask_receivers, axis=(0,-1)), axis=2)
  args_update_partial_steps_senders = jnp.take_along_axis(args_update_partial_steps, jnp.expand_dims(mask_senders, axis=(0,-1)), axis=2)

  # Partial message generation function
  msg_generation_fn_partial = functools.partial(
    msg_generation_fn,
    node_fts_receivers=node_fts_receivers_sample,
    node_fts_senders=node_fts_senders_sample,
    edge_fts=edge_fts_sample,
    graph_fts=graph_fts
  )
  # Partial message generation function (receivers)
  msg_generation_fn_partial_receivers = functools.partial(
    msg_generation_fn_partial,
    args_senders=args_senders_sample,
  )
  
  # Partial message generation function (senders)
  msg_generation_fn_partial_senders = functools.partial(
    lambda args_senders, args_receivers: msg_generation_fn_partial(args_receivers, args_senders),
    args_receivers=args_receivers_sample,
  )
  
  # Function to evaluate linearity in each component, returns two tensors of shape [B, num_nodes_sample, num_nodes_sample, H]
  def compute_aggregated_partial_msgs(msg_generation_fn_partial, args_update_partial_steps):
    msgs_aggregated = msg_generation_fn_partial(
      arg_reduction_fn(args_update_partial_steps, axis=0),
    )
    msgs_partial = msg_reduction_fn(
      jax.vmap(msg_generation_fn_partial, in_axes=0, out_axes=0)(
        args_update_partial_steps
      ), 
      axis=0
    )
    return msgs_aggregated, msgs_partial

  # Verify linearity in the second dimension (receivers)
  msgs_aggregated_receivers, msgs_partial_receivers = compute_aggregated_partial_msgs(
    msg_generation_fn_partial_receivers, args_update_partial_steps_receivers,
  )
  # Verify linearity in the first dimension (senders)
  msgs_aggregated_senders, msgs_partial_senders = compute_aggregated_partial_msgs(
    msg_generation_fn_partial_senders, args_update_partial_steps_senders,
  )
  
  # Aggregate violations of the linearity in each dimension
  l3_multimorphism_loss = jnp.mean(
    (
      msgs_aggregated_receivers-msgs_partial_receivers
    )**2 +
    (
      msgs_aggregated_senders-msgs_partial_senders  
    )**2
  )/2.

  # Return the three loss components as a dictionary
  return AsynchronyInformation(
    l2_loss=l2_loss, 
    l3_cocycle_loss=l3_cocycle_loss, 
    l3_multimorphism_loss=l3_multimorphism_loss,
    # Expand across first dimension to concatenate over steps
    l2_node_update_aggregated=node_update_aggregated,
    l2_node_update_partial=node_update_partial,
    l3_cocycle_args_update_aggregated=args_update_aggregated,
    l3_cocycle_args_update_aggregated_partial=args_update_aggregated_partial,
    l3_multimorphism_msgs_aggregated=msgs_aggregated_receivers+msgs_aggregated_senders,
    l3_multimorphism_msgs_partial=msgs_partial_receivers+msgs_partial_senders,
  )

def apply_msg_reduction(msg_reduction_fn: _Fn, msgs: _Array, adj_mat: _Array):
  # Computes m_{i}^{(t)} as a reduction over f_{m}(z_{i}^{(t)}, z_{j}^{(t)}, e_{ij}^{(t)}, g^{(t)}), 
  # e.g. m_{i}^{(t)}=max_{1 \leq j \leq n}f_{m}(z_{i}^{(t)}, z_{j}^{(t)}, e_{ij}^{(t)}, g^{(t)})
  if msg_reduction_fn == jnp.mean:
    # msgs: [B, N, N, self.mid_size], adj_mat: [B, N, N, 1]
    # If (i,j) \not \in adj_mat, then msgs[i,j,:]=0,  therefore,
    # when performing the sum over the axis=1, the reduction is 
    # applied in a one-hop neighborhood basis 
    msgs = jnp.sum(msgs * jnp.expand_dims(adj_mat, -1), axis=1)
    msgs = msgs / jnp.sum(adj_mat, axis=-1, keepdims=True)
  elif msg_reduction_fn == jnp.max:
    maxarg = jnp.where(jnp.expand_dims(adj_mat, -1),
                        msgs,
                        -BIG_NUMBER)
    msgs = jnp.max(maxarg, axis=1)
  else:
    msgs = msg_reduction_fn(msgs * jnp.expand_dims(adj_mat, -1), axis=1)

  return msgs

class Processor(hk.Module):
  """Processor abstract base class."""

  def __init__(self, name: str):
    if not name.endswith(PROCESSOR_TAG):
      name = name + '_' + PROCESSOR_TAG
    super().__init__(name=name)

  @abc.abstractmethod
  def __call__(
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **kwargs,
  ) -> Tuple[_Array, Optional[_Array]]:
    """Processor inference step.

    Args:
      node_fts: Node features.
      edge_fts: Edge features.
      graph_fts: Graph features.
      adj_mat: Graph adjacency matrix.
      hidden: Hidden features.
      **kwargs: Extra kwargs.

    Returns:
      Output of processor inference step as a 2-tuple of (node, edge)
      embeddings. The edge embeddings can be None.
    """
    pass

  @property
  def inf_bias(self):
    return False

  @property
  def inf_bias_edge(self):
    return False


class GAT(Processor):
  """Graph Attention Network (Velickovic et al., ICLR 2018)."""

  def __init__(
      self,
      out_size: int,
      nb_heads: int,
      activation: Optional[_Fn] = jax.nn.relu,
      residual: bool = True,
      use_ln: bool = False,
      name: str = 'gat_aggr',
  ):
    super().__init__(name=name)
    self.out_size = out_size
    self.nb_heads = nb_heads
    if out_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the width!')
    self.head_size = out_size // nb_heads
    self.activation = activation
    self.residual = residual
    self.use_ln = use_ln

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """GAT inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    z = jnp.concatenate([node_fts, hidden], axis=-1)
    m = hk.Linear(self.out_size)
    skip = hk.Linear(self.out_size)

    bias_mat = (adj_mat - 1.0) * 1e9
    bias_mat = jnp.tile(bias_mat[..., None],
                        (1, 1, 1, self.nb_heads))     # [B, N, N, H]
    bias_mat = jnp.transpose(bias_mat, (0, 3, 1, 2))  # [B, H, N, N]

    a_1 = hk.Linear(self.nb_heads)
    a_2 = hk.Linear(self.nb_heads)
    a_e = hk.Linear(self.nb_heads)
    a_g = hk.Linear(self.nb_heads)

    values = m(z)                                      # [B, N, H*F]
    values = jnp.reshape(
        values,
        values.shape[:-1] + (self.nb_heads, self.head_size))  # [B, N, H, F]
    values = jnp.transpose(values, (0, 2, 1, 3))              # [B, H, N, F]

    att_1 = jnp.expand_dims(a_1(z), axis=-1)
    att_2 = jnp.expand_dims(a_2(z), axis=-1)
    att_e = a_e(edge_fts)
    att_g = jnp.expand_dims(a_g(graph_fts), axis=-1)

    logits = (
        jnp.transpose(att_1, (0, 2, 1, 3)) +  # + [B, H, N, 1]
        jnp.transpose(att_2, (0, 2, 3, 1)) +  # + [B, H, 1, N]
        jnp.transpose(att_e, (0, 3, 1, 2)) +  # + [B, H, N, N]
        jnp.expand_dims(att_g, axis=-1)       # + [B, H, 1, 1]
    )                                         # = [B, H, N, N]
    coefs = jax.nn.softmax(jax.nn.leaky_relu(logits) + bias_mat, axis=-1)
    ret = jnp.matmul(coefs, values)  # [B, H, N, F]
    ret = jnp.transpose(ret, (0, 2, 1, 3))  # [B, N, H, F]
    ret = jnp.reshape(ret, ret.shape[:-2] + (self.out_size,))  # [B, N, H*F]

    if self.residual:
      ret += skip(z)

    if self.activation is not None:
      ret = self.activation(ret)

    if self.use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    return ret, None  # pytype: disable=bad-return-type  # numpy-scalars


class GATFull(GAT):
  """Graph Attention Network with full adjacency matrix."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class GATv2(Processor):
  """Graph Attention Network v2 (Brody et al., ICLR 2022)."""

  def __init__(
      self,
      out_size: int,
      nb_heads: int,
      mid_size: Optional[int] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      residual: bool = True,
      use_ln: bool = False,
      name: str = 'gatv2_aggr',
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.nb_heads = nb_heads
    if out_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the width!')
    self.head_size = out_size // nb_heads
    if self.mid_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the message!')
    self.mid_head_size = self.mid_size // nb_heads
    self.activation = activation
    self.residual = residual
    self.use_ln = use_ln

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """GATv2 inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    z = jnp.concatenate([node_fts, hidden], axis=-1)
    m = hk.Linear(self.out_size)
    skip = hk.Linear(self.out_size)

    bias_mat = (adj_mat - 1.0) * 1e9
    bias_mat = jnp.tile(bias_mat[..., None],
                        (1, 1, 1, self.nb_heads))     # [B, N, N, H]
    bias_mat = jnp.transpose(bias_mat, (0, 3, 1, 2))  # [B, H, N, N]

    w_1 = hk.Linear(self.mid_size)
    w_2 = hk.Linear(self.mid_size)
    w_e = hk.Linear(self.mid_size)
    w_g = hk.Linear(self.mid_size)

    a_heads = []
    for _ in range(self.nb_heads):
      a_heads.append(hk.Linear(1))

    values = m(z)                                      # [B, N, H*F]
    values = jnp.reshape(
        values,
        values.shape[:-1] + (self.nb_heads, self.head_size))  # [B, N, H, F]
    values = jnp.transpose(values, (0, 2, 1, 3))              # [B, H, N, F]

    pre_att_1 = w_1(z)
    pre_att_2 = w_2(z)
    pre_att_e = w_e(edge_fts)
    pre_att_g = w_g(graph_fts)

    pre_att = (
        jnp.expand_dims(pre_att_1, axis=1) +     # + [B, 1, N, H*F]
        jnp.expand_dims(pre_att_2, axis=2) +     # + [B, N, 1, H*F]
        pre_att_e +                              # + [B, N, N, H*F]
        jnp.expand_dims(pre_att_g, axis=(1, 2))  # + [B, 1, 1, H*F]
    )                                            # = [B, N, N, H*F]

    pre_att = jnp.reshape(
        pre_att,
        pre_att.shape[:-1] + (self.nb_heads, self.mid_head_size)
    )  # [B, N, N, H, F]

    pre_att = jnp.transpose(pre_att, (0, 3, 1, 2, 4))  # [B, H, N, N, F]

    # This part is not very efficient, but we agree to keep it this way to
    # enhance readability, assuming `nb_heads` will not be large.
    logit_heads = []
    for head in range(self.nb_heads):
      logit_heads.append(
          jnp.squeeze(
              a_heads[head](jax.nn.leaky_relu(pre_att[:, head])),
              axis=-1)
      )  # [B, N, N]

    logits = jnp.stack(logit_heads, axis=1)  # [B, H, N, N]

    coefs = jax.nn.softmax(logits + bias_mat, axis=-1)
    ret = jnp.matmul(coefs, values)  # [B, H, N, F]
    ret = jnp.transpose(ret, (0, 2, 1, 3))  # [B, N, H, F]
    ret = jnp.reshape(ret, ret.shape[:-2] + (self.out_size,))  # [B, N, H*F]

    if self.residual:
      ret += skip(z)

    if self.activation is not None:
      ret = self.activation(ret)

    if self.use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    return ret, None  # pytype: disable=bad-return-type  # numpy-scalars


class GATv2FullD2(GATv2):
  """Graph Attention Network v2 with full adjacency matrix and D2 symmetry."""

  def d2_forward(self,
                 node_fts: List[_Array],
                 edge_fts: List[_Array],
                 graph_fts: List[_Array],
                 adj_mat: _Array,
                 hidden: _Array,
                 **unused_kwargs) -> List[_Array]:
    num_d2_actions = 4

    d2_inverses = [
        0, 1, 2, 3  # All members of D_2 are self-inverses!
    ]

    d2_multiply = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ]

    assert len(node_fts) == num_d2_actions
    assert len(edge_fts) == num_d2_actions
    assert len(graph_fts) == num_d2_actions

    ret_nodes = []
    adj_mat = jnp.ones_like(adj_mat)

    for g in range(num_d2_actions):
      emb_values = []
      for h in range(num_d2_actions):
        gh = d2_multiply[d2_inverses[g]][h]
        node_features = jnp.concatenate(
            (node_fts[g], node_fts[gh]),
            axis=-1)
        edge_features = jnp.concatenate(
            (edge_fts[g], edge_fts[gh]),
            axis=-1)
        graph_features = jnp.concatenate(
            (graph_fts[g], graph_fts[gh]),
            axis=-1)
        cell_embedding = super().__call__(
            node_fts=node_features,
            edge_fts=edge_features,
            graph_fts=graph_features,
            adj_mat=adj_mat,
            hidden=hidden
        )
        emb_values.append(cell_embedding[0])
      ret_nodes.append(
          jnp.mean(jnp.stack(emb_values, axis=0), axis=0)
      )

    return ret_nodes


class GATv2Full(GATv2):
  """Graph Attention Network v2 with full adjacency matrix."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


def get_triplet_msgs(z, edge_fts, graph_fts, nb_triplet_fts):
  """Triplet messages, as done by Dudzik and Velickovic (2022)."""
  t_1 = hk.Linear(nb_triplet_fts)
  t_2 = hk.Linear(nb_triplet_fts)
  t_3 = hk.Linear(nb_triplet_fts)
  t_e_1 = hk.Linear(nb_triplet_fts)
  t_e_2 = hk.Linear(nb_triplet_fts)
  t_e_3 = hk.Linear(nb_triplet_fts)
  t_g = hk.Linear(nb_triplet_fts)

  tri_1 = t_1(z)
  tri_2 = t_2(z)
  tri_3 = t_3(z)
  tri_e_1 = t_e_1(edge_fts)
  tri_e_2 = t_e_2(edge_fts)
  tri_e_3 = t_e_3(edge_fts)
  tri_g = t_g(graph_fts)

  return (
      jnp.expand_dims(tri_1, axis=(2, 3))    +  #   (B, N, 1, 1, H)
      jnp.expand_dims(tri_2, axis=(1, 3))    +  # + (B, 1, N, 1, H)
      jnp.expand_dims(tri_3, axis=(1, 2))    +  # + (B, 1, 1, N, H)
      jnp.expand_dims(tri_e_1, axis=3)       +  # + (B, N, N, 1, H)
      jnp.expand_dims(tri_e_2, axis=2)       +  # + (B, N, 1, N, H)
      jnp.expand_dims(tri_e_3, axis=1)       +  # + (B, 1, N, N, H)
      jnp.expand_dims(tri_g, axis=(1, 2, 3))    # + (B, 1, 1, 1, H)
  )                                             # = (B, N, N, N, H)


class PGN(Processor):
  """Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      mid_act: Optional[_Fn] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      reduction: _Fn = jnp.max,
      msgs_mlp_sizes: Optional[List[int]] = None,
      use_ln: bool = False,
      use_triplets: bool = False,
      nb_triplet_fts: int = 8,
      gated: bool = False,
      force_linear: bool = False,
      name: str = 'mpnn_aggr',
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.mid_act = mid_act
    self.activation = activation
    self.reduction = reduction
    self._msgs_mlp_sizes = msgs_mlp_sizes
    self.use_ln = use_ln
    self.use_triplets = use_triplets
    self.nb_triplet_fts = nb_triplet_fts
    self._force_linear = force_linear
    self.gated = gated

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MPNN inference step."""

    # b: batch size, n: number of nodes, h: hidden features

    # node_fts: [B, N, H] 
    # edge_fts: [B, N, N, H]
    # graph_fts: [B, H]
    # adj_mat: [B, N, N]
    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    # hidden: [B, N, H] (Note that it is possible for this H to be different)

    # Concatenate along last axis
    # z^{(t)} = x_{i}^{(t)} || h_{i}^{(t-1)}
    z = jnp.concatenate([node_fts, hidden], axis=-1) # [B, N, 2*H]
    m_1 = hk.Linear(self.mid_size)
    m_2 = hk.Linear(self.mid_size)
    m_e = hk.Linear(self.mid_size)
    m_g = hk.Linear(self.mid_size)

    o1 = hk.Linear(self.out_size)
    o2 = hk.Linear(self.out_size)

    if self.gated:
      gate1 = hk.Linear(self.out_size)
      gate2 = hk.Linear(self.out_size)
      gate3 = hk.Linear(self.out_size, b_init=hk.initializers.Constant(-3))

    msg_1 = m_1(z) # [B, N, self.mid_size]
    msg_2 = m_2(z) # [B, N, self.mid_size]
    msg_e = m_e(edge_fts) # [B, N, N, self.mid_size]
    msg_g = m_g(graph_fts) # [B, self.mid_size]

    tri_msgs = None

    if self.use_triplets and not self._force_linear:
      # Triplet messages, as done by Dudzik and Velickovic (2022)
      triplets = get_triplet_msgs(z, edge_fts, graph_fts, self.nb_triplet_fts)

      o3 = hk.Linear(self.out_size)
      tri_msgs = o3(jnp.max(triplets, axis=1))  # [B, N, N, H]

      if self.activation is not None:
        tri_msgs = self.activation(tri_msgs)

    # The partial messages are aggregated by summing them
    msgs = (
        jnp.expand_dims(msg_1, axis=1) + # [B, 1, N, self.mid_size]
        jnp.expand_dims(msg_2, axis=2) + # [B, N, 1, self.mid_size]
        msg_e + # [B, N, N, self.mid_size]
        jnp.expand_dims(msg_g, axis=(1, 2))) # [B, 1, 1, self.mid_size]

    if self._msgs_mlp_sizes is not None:
      # self._msgs_mlp_sizes is a tuple, e.g. [64, 128, 16], with the output dimesions
      # for a series of Linear layers, followed by their respective activations
      if not self._force_linear:
        msgs = hk.nets.MLP(self._msgs_mlp_sizes)(jax.nn.relu(msgs))
      else:
        msgs = hk.nets.MLP(self._msgs_mlp_sizes, activation=lambda x: x)(msgs)

    # Additional activation
    if self.mid_act is not None and not self._force_linear:
      msgs = self.mid_act(msgs)

    # Computation of h_{i}^{(t)}=f_{r}(z_{i}^{(t)}, m_{i}^{(t)})
    def compute_node_update(hidden, msgs, z):
      h_1 = o1(z)
      h_2 = o2(msgs)
      # h_{i}^{(t)} = Linear(z_{i}^{(t)}) + Linear(m_{i}^{(t)})
      ret = h_1 + h_2

      if self.activation is not None and not self._force_linear:
        ret = self.activation(ret)

      # Include LayerNorm if needed
      if self.use_ln:
        ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ret = ln(ret)

      if self.gated:
        gate = jax.nn.sigmoid(gate3(jax.nn.relu(gate1(z) + gate2(msgs))))
        ret = ret * gate + hidden * (1-gate)

      return ret
    
    # Format compatible with jax.lax.scan. This method is created to prevent
    # redundant concatenations of the hidden state with the node features,
    # in comparison with the method compute_node_update which receives z directly
    def compute_node_update_scan(hidden, msgs):
      z = jnp.concatenate([node_fts, hidden], axis=-1) # [B, N, 2*H]
      # The hidden state of the node is propagated within the carry. The second
      # return is None as it is not used, and it is only required by hk.scan
      return compute_node_update(hidden, msgs, z), None

    # At this point, messages contains the operation f_{m}(z_{i}^{(t)}, z_{j}^{(t)}, e_{ij}^{(t)}, g^{(t)}),
    # where f_{m} is the message function (MLP with non-linearities)
      
    ###############
    ## The following code includes the operations needed for the computation of the
    ## regularized loss.
    ###############

    # Number of messages to sample TODO: Pass through command line
    num_samples_per_node = 2
    # Randomly sample incoming messages for each node. sample_msgs has resulting shape [B, 2, N, H]
    sampled_msgs = sample_msgs(msgs, adj_mat, num_samples_per_node)

    # Computes m_{i}^{(t)} as a reduction over f_{m}(z_{i}^{(t)}, z_{j}^{(t)}, e_{ij}^{(t)}, g^{(t)}), 
    # e.g. m_{i}^{(t)}=max_{1 \leq j \leq n}f_{m}(z_{i}^{(t)}, z_{j}^{(t)}, e_{ij}^{(t)}, g^{(t)})
    if self.reduction == jnp.mean:
    # sampled_msgs: [B, num_samples_per_node, N, H], sampled_msgs_red: [B, N, H]
      # Note that there is no need to multiply by the adjacency matrix
      # as the incoming messages will be sampled from nodes in the 
      # neighbourhood of each node
      sampled_msgs_reduced = jnp.sum(sampled_msgs, axis=1)
      sampled_msgs_reduced = sampled_msgs_reduced / num_samples_per_node
    elif self.reduction == jnp.max:
      sampled_msgs_reduced = jnp.max(sampled_msgs, axis=1)
    else:
      sampled_msgs_reduced = self.reduction(sampled_msgs, axis=1)

    # Node update for the aggregated sampled messages (first component of the loss)
    ret_loss_1 = compute_node_update(hidden, sampled_msgs_reduced, z)
    # The shape of sample messages is [B, num_samples_per_node, N, H], and the arguments
    # in jax.lax.scan are passed through the leading dimension
    sampled_msgs = jnp.transpose(sampled_msgs, (1, 0, 2, 3))
    # The first update is different to prevent a redundant concatenation of the node
    # features.
    next_hidden = compute_node_update(hidden, sampled_msgs[0], z)
    # Iterative updates for the sampled messages
    ret_loss_2, _ = hk.scan(compute_node_update_scan, next_hidden, sampled_msgs[1:], num_samples_per_node-1)
    # Regularization term penalising the violation of the associativity in the node update
    mse_loss = jnp.mean((ret_loss_1 - ret_loss_2)**2)

    ###############
    ## End of code for computing the regularized loss.
    ###############

    # Computes m_{i}^{(t)} as a reduction over f_{m}(z_{i}^{(t)}, z_{j}^{(t)}, e_{ij}^{(t)}, g^{(t)}), 
    # e.g. m_{i}^{(t)}=max_{1 \leq j \leq n}f_{m}(z_{i}^{(t)}, z_{j}^{(t)}, e_{ij}^{(t)}, g^{(t)})
    if self.reduction == jnp.mean:
      # msgs: [B, N, N, self.mid_size], adj_mat: [B, N, N, 1]
      # If (i,j) \not \in adj_mat, then msgs[i,j,:]=0,  therefore,
      # when performing the sum over the axis=1, the reduction is 
      # applied in a one-hop neighborhood basis 
      msgs = jnp.sum(msgs * jnp.expand_dims(adj_mat, -1), axis=1)
      msgs = msgs / jnp.sum(adj_mat, axis=-1, keepdims=True)
    elif self.reduction == jnp.max:
      maxarg = jnp.where(jnp.expand_dims(adj_mat, -1),
                         msgs,
                         -BIG_NUMBER)
      msgs = jnp.max(maxarg, axis=1)
    elif self.reduction == jax.nn.softmax:
      maxarg = jnp.where(jnp.expand_dims(adj_mat, -1),
                         msgs,
                         -BIG_NUMBER)
      inv_temp = 1  # hk.get_parameter("temp", (), init=jnp.ones)
      soft_coeffs = jax.nn.softmax(inv_temp * maxarg, axis=1)
      msgs = jnp.sum(msgs * soft_coeffs, axis=1)
    else:
      msgs = self.reduction(msgs * jnp.expand_dims(adj_mat, -1), axis=1)

    # Computation of h_{i}^{(t)}=f_{r}(z_{i}^{(t)}, m_{i}^{(t)})
    ret = compute_node_update(hidden, msgs, z)

    return ret, tri_msgs, mse_loss  # pytype: disable=bad-return-type  # numpy-scalars

class PGN_L1(Processor):
  """Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      mid_act: Optional[_Fn] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      reduction: _Fn = jnp.max,
      arg_reduction: _Fn = jnp.max,
      msgs_mlp_sizes: Optional[List[int]] = None,
      use_ln: bool = False,
      use_triplets: bool = False,
      nb_triplet_fts: int = 8,
      gated: bool = False,
      name: str = 'mpnn_l1',
      num_messages_sample: int = 2,
      num_nodes_sample: int = 8,
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.mid_act = mid_act
    # Same as in Asynchronous Algorithmic Alignment with Cocycles
    self.activation = activation
    self.reduction = reduction
    self.arg_reduction = arg_reduction
    self._msgs_mlp_sizes = msgs_mlp_sizes
    self.use_ln = use_ln
    self.use_triplets = use_triplets
    self.nb_triplet_fts = nb_triplet_fts
    self.gated = gated
    self.num_messages_sample = num_messages_sample
    self.num_nodes_sample = num_nodes_sample

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      node_args: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MPNN inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    m_1 = hk.Linear(self.mid_size)
    m_2 = hk.Linear(self.mid_size)
    m_e = hk.Linear(self.mid_size)
    m_g = hk.Linear(self.mid_size)

    o1 = hk.Linear(self.out_size)
    o2 = hk.Linear(self.out_size)

    if self.use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    ###############
    ## Functions to compute message passing steps
    ###############
    
    # Definition of msg_generation_fn
    def compute_messages(args_receivers, args_senders, node_fts_receivers, node_fts_senders, edge_fts, graph_fts):
      z_1 = jnp.concatenate([node_fts_receivers, args_receivers], axis=-1)
      z_2 = jnp.concatenate([node_fts_senders, args_senders], axis=-1)
      msg_1 = m_1(z_1)
      msg_2 = m_2(z_2)
      msg_e = m_e(edge_fts)
      msg_g = m_g(graph_fts)

      # The message generator function is linear (psi)
      msgs = (
          jnp.expand_dims(msg_1, axis=1) + jnp.expand_dims(msg_2, axis=2) +
          msg_e + jnp.expand_dims(msg_g, axis=(1, 2)))
      
      return msgs
    
    def compute_node_update_scan(hidden, msgs):
      # The node update is linear
      h_1 = o1(hidden)
      h_2 = o2(msgs)

      ret = h_1 + h_2

      # ReLU activation
      if self.activation is not None:
        ret = self.activation(ret)

      # Include LayerNorm if needed
      if self.use_ln:
        ret = ln(ret)

      # The hidden state of the node is propagated within the carry. The second
      # return is None as it is not used, and it is only required by hk.scan
      return ret, ret

    ###############
    ## End of functions to compute message passing steps
    ###############

    # Compute messages
    msgs = compute_messages(hidden, hidden, node_fts, node_fts, edge_fts, graph_fts)

    # Call to code to compute losses
    asynchrony_losses = compute_asynchrony_losses(
      self.num_messages_sample, 
      self.num_nodes_sample, 
      hidden,
      hidden,
      msgs, 
      adj_mat, 
      node_fts, 
      edge_fts, 
      graph_fts, 
      self.reduction, 
      self.arg_reduction, 
      compute_node_update_scan, 
      compute_node_update_scan,
      compute_messages
    )

    ###############
    ## End of code for computing the regularized loss.
    ###############

    # Computes m_{i}^{(t)} as a reduction over f_{m}(z_{i}^{(t)}, z_{j}^{(t)}, e_{ij}^{(t)}, g^{(t)}), 
    # e.g. m_{i}^{(t)}=max_{1 \leq j \leq n}f_{m}(z_{i}^{(t)}, z_{j}^{(t)}, e_{ij}^{(t)}, g^{(t)})
    msgs = apply_msg_reduction(self.reduction, msgs, adj_mat)

    # Computation of h_{i}^{(t)}=f_{r}(z_{i}^{(t)}, m_{i}^{(t)})
    ret, _ = compute_node_update_scan(hidden, msgs)

    return ret, ret, None, asynchrony_losses  # pytype: disable=bad-return-type  # numpy-scalars
  
class PGN_L2(Processor):
  """Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      mid_act: Optional[_Fn] = None,
      activation: Optional[_Fn] = None,
      reduction: _Fn = jnp.max,
      arg_reduction: _Fn = jnp.max,
      msgs_mlp_sizes: Optional[List[int]] = None,
      use_ln: bool = False,
      use_triplets: bool = False,
      nb_triplet_fts: int = 8,
      gated: bool = False,
      name: str = 'mpnn_l2',
      num_messages_sample: int = 2,
      num_nodes_sample: int = 8,
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.mid_act = mid_act
    # Same as in Asynchronous Algorithmic Alignment with Cocycles
    self.activation = activation
    self.reduction = reduction
    self.arg_reduction = arg_reduction
    self._msgs_mlp_sizes = msgs_mlp_sizes
    self.use_ln = use_ln
    self.use_triplets = use_triplets
    self.nb_triplet_fts = nb_triplet_fts
    self.gated = gated
    self.num_messages_sample = num_messages_sample
    self.num_nodes_sample = num_nodes_sample

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      node_args: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MPNN inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    m_1 = hk.Linear(self.mid_size)
    m_2 = hk.Linear(self.mid_size)
    m_e = hk.Linear(self.mid_size)
    m_g = hk.Linear(self.mid_size)

    ###############
    ## Functions to compute message passing steps
    ###############
    
    # Definition of msg_generation_fn
    def compute_messages(args_receivers, args_senders, node_fts_receivers, node_fts_senders, edge_fts, graph_fts):

      z_1 = jnp.concatenate([node_fts_receivers, args_receivers], axis=-1)
      z_2 = jnp.concatenate([node_fts_senders, args_senders], axis=-1)
      msg_1 = m_1(z_1)
      msg_2 = m_2(z_2)
      msg_e = m_e(edge_fts)
      msg_g = m_g(graph_fts)

      # The message generator function is linear (psi)
      msgs = (
          jnp.expand_dims(msg_1, axis=1) + jnp.expand_dims(msg_2, axis=2) +
          msg_e + jnp.expand_dims(msg_g, axis=(1, 2)))
      
      return msgs
    
    def compute_node_update_scan(hidden, msgs):
      # The node update is the maximum
      ret = jnp.maximum(hidden, msgs)
      # The hidden state of the node is propagated within the carry. The second
      # return is None as it is not used, and it is only required by hk.scan
      return ret, ret

    ###############
    ## End of functions to compute message passing steps
    ###############
    
    # Compute messages that are propagated
    msgs = compute_messages(hidden, hidden, node_fts, node_fts, edge_fts, graph_fts)

    # Compute asynchrony losses
    asynchrony_losses = compute_asynchrony_losses(
      self.num_messages_sample, 
      self.num_nodes_sample, 
      hidden, 
      hidden,
      msgs, 
      adj_mat, 
      node_fts, 
      edge_fts, 
      graph_fts, 
      self.reduction,
      self.arg_reduction,
      compute_node_update_scan, 
      compute_node_update_scan,
      compute_messages
    )

    ###############
    ## End of code for computing the regularized loss.
    ###############

    # Computes m_{i}^{(t)} as a reduction over f_{m}(z_{i}^{(t)}, z_{j}^{(t)}, e_{ij}^{(t)}, g^{(t)}), 
    # e.g. m_{i}^{(t)}=max_{1 \leq j \leq n}f_{m}(z_{i}^{(t)}, z_{j}^{(t)}, e_{ij}^{(t)}, g^{(t)})
    msgs = apply_msg_reduction(self.reduction, msgs, adj_mat)

    # The node update is the maximum
    ret, _ = compute_node_update_scan(hidden, msgs)

    return ret, ret, None, asynchrony_losses # pytype: disable=bad-return-type  # numpy-scalars

class PGN_L3(Processor):
  """Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      mid_act: Optional[_Fn] = None,
      activation: Optional[_Fn] = None,
      reduction: _Fn = jnp.max,
      arg_reduction: _Fn = jnp.max,
      msgs_mlp_sizes: Optional[List[int]] = None,
      use_ln: bool = False,
      use_triplets: bool = False,
      nb_triplet_fts: int = 8,
      gated: bool = False,
      name: str = 'mpnn_l3',
      num_messages_sample: int = 2,
      num_nodes_sample: int = 8,
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.mid_act = mid_act
    # Same as in Asynchronous Algorithmic Alignment with Cocycles
    self.activation = activation
    self.reduction = reduction
    self.arg_reduction = arg_reduction
    self._msgs_mlp_sizes = msgs_mlp_sizes
    self.use_ln = use_ln
    self.use_triplets = use_triplets
    self.nb_triplet_fts = nb_triplet_fts
    self.gated = gated
    self.num_messages_sample = num_messages_sample
    self.num_nodes_sample = num_nodes_sample

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      node_args: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MPNN inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    m_1 = TropicalLinear(self.mid_size)
    m_2 = TropicalLinear(self.mid_size)

    m_1_input = hk.Linear(self.mid_size)
    m_2_input = hk.Linear(self.mid_size)
    m_e = hk.Linear(self.mid_size)
    m_g = hk.Linear(self.mid_size)

    ###############
    ## The following code includes the operations needed to compute losses
    ###############

    # Definition of msg_generation_fn
    def compute_messages(args_receivers, args_senders, node_fts_receivers, node_fts_senders, edge_fts, graph_fts):

      msg_1_input = m_1_input(node_fts_receivers)
      msg_2_input = m_2_input(node_fts_senders)
      msg_e = m_e(edge_fts)
      msg_g = m_g(graph_fts)

      msg_1 = m_1(args_receivers)
      msg_2 = m_2(args_senders)

      # The message generator function is linear (psi)
      msgs = (
          jnp.expand_dims(msg_1, axis=1) + jnp.expand_dims(msg_2, axis=2) +
          jnp.expand_dims(msg_1_input, axis=1) + jnp.expand_dims(msg_2_input, axis=2) +
          msg_e + jnp.expand_dims(msg_g, axis=(1, 2))
      )

      return msgs

    # Definition of node_update_scan_fn
    def compute_node_update_scan(hidden, msgs):
      # The node update is the maximum
      ret = jnp.maximum(hidden, msgs)
      # The hidden state of the node is propagated within the carry, and it is also accumulated
      # during hk.scan
      return ret, ret
    
    ###############
    ## End of operations to compute losses
    ###############

    # Compute messages
    msgs = compute_messages(hidden, hidden, node_fts, node_fts, edge_fts, graph_fts)

    # Call to code to compute losses
    asynchrony_losses = compute_asynchrony_losses(
      self.num_messages_sample, 
      self.num_nodes_sample,
      hidden, 
      hidden, 
      msgs, 
      adj_mat, 
      node_fts, 
      edge_fts, 
      graph_fts, 
      self.reduction, 
      self.arg_reduction, 
      compute_node_update_scan, 
      compute_node_update_scan, 
      compute_messages
    )

    # Computes m_{i}^{(t)} as a reduction over f_{m}(z_{i}^{(t)}, z_{j}^{(t)}, e_{ij}^{(t)}, g^{(t)}), 
    # e.g. m_{i}^{(t)}=max_{1 \leq j \leq n}f_{m}(z_{i}^{(t)}, z_{j}^{(t)}, e_{ij}^{(t)}, g^{(t)})
    msgs = apply_msg_reduction(self.reduction, msgs, adj_mat)

    ret, _ = compute_node_update_scan(hidden, msgs)

    return ret, ret, None, asynchrony_losses # pytype: disable=bad-return-type  # numpy-scalars
  
class PGN_L2_L3(Processor):
  """Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      mid_act: Optional[_Fn] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      reduction: _Fn = jnp.max,
      arg_reduction: _Fn = jnp.max,
      msgs_mlp_sizes: Optional[List[int]] = None,
      use_ln: bool = False,
      use_triplets: bool = False,
      nb_triplet_fts: int = 8,
      gated: bool = False,
      name: str = 'mpnn_l2_l3',
      num_messages_sample: int = 2,
      num_nodes_sample: int = 8,
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.mid_act = mid_act
    # Same as in Asynchronous Algorithmic Alignment with Cocycles
    self.activation = activation
    self.reduction = reduction
    self.arg_reduction = arg_reduction
    self._msgs_mlp_sizes = msgs_mlp_sizes
    self.use_ln = use_ln
    self.use_triplets = use_triplets
    self.nb_triplet_fts = nb_triplet_fts
    self.gated = gated
    self.num_messages_sample = num_messages_sample
    self.num_nodes_sample = num_nodes_sample

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      node_args: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MPNN inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    m_1 = hk.Linear(self.mid_size)
    m_2 = hk.Linear(self.mid_size)

    m_1_input = hk.Linear(self.mid_size)
    m_2_input = hk.Linear(self.mid_size)
    m_e = hk.Linear(self.mid_size)
    m_g = hk.Linear(self.mid_size)

    o1 = hk.Linear(self.out_size)
    o2 = hk.Linear(self.out_size)

    if self.use_ln:
      # Use LayerNorm to prevent great losses observed when using sum aggregation
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    ###############
    ## The following code includes the operations needed to compute losses
    ###############

    # Definition of msg_generation_fn
    def compute_messages(args_receivers, args_senders, node_fts_receivers, node_fts_senders, edge_fts, graph_fts):

      msg_1_input = m_1_input(node_fts_receivers)
      msg_2_input = m_2_input(node_fts_senders)
      msg_e = m_e(edge_fts)
      msg_g = m_g(graph_fts)

      msg_1 = m_1(args_receivers)
      msg_2 = m_2(args_senders)

      # The message generator function is linear (psi)
      msgs = (
          jnp.expand_dims(msg_1, axis=1) + jnp.expand_dims(msg_2, axis=2) +
          jnp.expand_dims(msg_1_input, axis=1) + jnp.expand_dims(msg_2_input, axis=2) +
          msg_e + jnp.expand_dims(msg_g, axis=(1, 2))
      )

      return msgs

    # Definition of node_update_scan_fn
    def compute_node_update_scan(hidden, msgs):
      # The node update is the maximum
      ret = jnp.maximum(hidden, msgs)
      # The hidden state of the node is propagated within the carry. The second
      # return is None as it is not used, and it is only required by hk.scan
      return ret, ret
    
    # Definition of arg_update_scan_fn
    def compute_arg_update_scan(hidden, msgs):
      # The arg update is linear
      h_1 = o1(hidden)
      h_2 = o2(msgs)

      ret = h_1 + h_2

      # ReLU activation
      if self.activation is not None:
        ret = self.activation(ret)

      if self.use_ln:
        ret = ln(ret)
      
      # The arguments generated at the node are propagated within the carry.
      return ret, ret
    
    ###############
    ## End of operations to compute losses
    ###############

    # Compute messages
    msgs = compute_messages(node_args, node_args, node_fts, node_fts, edge_fts, graph_fts)

    # Call to code to compute losses
    asynchrony_losses = compute_asynchrony_losses(
      self.num_messages_sample, 
      self.num_nodes_sample, 
      hidden, 
      node_args, 
      msgs, 
      adj_mat, 
      node_fts, 
      edge_fts, 
      graph_fts, 
      self.reduction, 
      self.arg_reduction, 
      compute_node_update_scan, 
      compute_arg_update_scan, 
      compute_messages
    )

    # Computes m_{i}^{(t)} as a reduction over f_{m}(z_{i}^{(t)}, z_{j}^{(t)}, e_{ij}^{(t)}, g^{(t)}), 
    # e.g. m_{i}^{(t)}=max_{1 \leq j \leq n}f_{m}(z_{i}^{(t)}, z_{j}^{(t)}, e_{ij}^{(t)}, g^{(t)})
    msgs = apply_msg_reduction(self.reduction, msgs, adj_mat)

    ret, _ = compute_node_update_scan(hidden, msgs)
    args, _ = compute_arg_update_scan(hidden, msgs)

    return ret, args, None, asynchrony_losses # pytype: disable=bad-return-type  # numpy-scalars
  
class PGN_L1_L3(Processor):
  """Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      mid_act: Optional[_Fn] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      reduction: _Fn = jnp.max,
      arg_reduction: _Fn = jnp.max,
      msgs_mlp_sizes: Optional[List[int]] = None,
      use_ln: bool = False,
      use_triplets: bool = False,
      nb_triplet_fts: int = 8,
      gated: bool = False,
      name: str = 'mpnn_l1_l3',
      num_messages_sample: int = 2,
      num_nodes_sample: int = 8,
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.mid_act = mid_act
    # Same as in Asynchronous Algorithmic Alignment with Cocycles
    self.activation = activation
    self.reduction = reduction
    self.arg_reduction = arg_reduction
    self._msgs_mlp_sizes = msgs_mlp_sizes
    self.use_ln = use_ln
    self.use_triplets = use_triplets
    self.nb_triplet_fts = nb_triplet_fts
    self.gated = gated
    self.num_messages_sample = num_messages_sample
    self.num_nodes_sample = num_nodes_sample

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      node_args: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MPNN inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    m_1 = hk.Linear(self.mid_size)
    m_2 = hk.Linear(self.mid_size)

    m_1_input = hk.Linear(self.mid_size)
    m_2_input = hk.Linear(self.mid_size)
    m_e = hk.Linear(self.mid_size)
    m_g = hk.Linear(self.mid_size)

    o1 = hk.Linear(self.out_size)
    o2 = hk.Linear(self.out_size)

    o3 = hk.Linear(self.out_size)
    o4 = hk.Linear(self.out_size)

    if self.use_ln:
      # Use LayerNorm to prevent great losses observed when using sum aggregation
      ln_1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ln_2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    ###############
    ## The following code includes the operations needed to compute losses
    ###############

    # Definition of msg_generation_fn
    def compute_messages(args_receivers, args_senders, node_fts_receivers, node_fts_senders, edge_fts, graph_fts):

      msg_1_input = m_1_input(node_fts_receivers)
      msg_2_input = m_2_input(node_fts_senders)
      msg_e = m_e(edge_fts)
      msg_g = m_g(graph_fts)

      msg_1 = m_1(args_receivers)
      msg_2 = m_2(args_senders)

      # The message generator function is linear (psi)
      msgs = (
          jnp.expand_dims(msg_1, axis=1) + jnp.expand_dims(msg_2, axis=2) +
          jnp.expand_dims(msg_1_input, axis=1) + jnp.expand_dims(msg_2_input, axis=2) +
          msg_e + jnp.expand_dims(msg_g, axis=(1, 2))
      )

      return msgs

    def compute_node_update_scan(hidden, msgs):
      # The node update is linear
      h_1 = o1(hidden)
      h_2 = o2(msgs)

      ret = h_1 + h_2

      # ReLU activation
      if self.activation is not None:
        ret = self.activation(ret)

      # Include LayerNorm if needed
      if self.use_ln:
        ret = ln_1(ret)

      # The hidden state of the node is propagated within the carry. The second
      # return is None as it is not used, and it is only required by hk.scan
      return ret, ret
    
    # Definition of arg_update_scan_fn
    def compute_arg_update_scan(hidden, msgs):
      # The arg update is linear
      h_1 = o3(hidden)
      h_2 = o4(msgs)

      ret = h_1 + h_2

      # ReLU activation
      if self.activation is not None:
        ret = self.activation(ret)

      if self.use_ln:
        ret = ln_2(ret)
      
      # The arguments generated at the node are propagated within the carry.
      return ret, ret
    
    ###############
    ## End of operations to compute losses
    ###############

    # Compute messages
    msgs = compute_messages(node_args, node_args, node_fts, node_fts, edge_fts, graph_fts)

    # Call to code to compute losses
    asynchrony_losses = compute_asynchrony_losses(
      self.num_messages_sample, 
      self.num_nodes_sample, 
      hidden, 
      node_args, 
      msgs, 
      adj_mat, 
      node_fts, 
      edge_fts, 
      graph_fts, 
      self.reduction, 
      self.arg_reduction, 
      compute_node_update_scan, 
      compute_arg_update_scan, 
      compute_messages
    )

    # Computes m_{i}^{(t)} as a reduction over f_{m}(z_{i}^{(t)}, z_{j}^{(t)}, e_{ij}^{(t)}, g^{(t)}), 
    # e.g. m_{i}^{(t)}=max_{1 \leq j \leq n}f_{m}(z_{i}^{(t)}, z_{j}^{(t)}, e_{ij}^{(t)}, g^{(t)})
    msgs = apply_msg_reduction(self.reduction, msgs, adj_mat)

    ret, _ = compute_node_update_scan(hidden, msgs)
    args, _ = compute_arg_update_scan(hidden, msgs)

    return ret, args, None, asynchrony_losses # pytype: disable=bad-return-type  # numpy-scalars

class LinearPGN(PGN):
  """PGN Network without nonlinearities"""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      reduction: _Fn = jnp.max,
      msgs_mlp_sizes: Optional[List[int]] = None,
      use_ln: bool = False,
      use_triplets: bool = False,
      nb_triplet_fts: int = 8,
      gated: bool = False,
      name: str = 'linear_pgn',
  ):
    super().__init__(out_size=out_size,
                     mid_size=mid_size,
                     mid_act=None,
                     activation=None,
                     reduction=reduction,
                     msgs_mlp_sizes=msgs_mlp_sizes,
                     use_ln=use_ln,
                     use_triplets=use_triplets,
                     nb_triplet_fts=nb_triplet_fts,
                     gated=gated,
                     force_linear=True,
                     name=name)

class DeepSets(PGN):
  """Deep Sets (Zaheer et al., NeurIPS 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    assert adj_mat.ndim == 3
    adj_mat = jnp.ones_like(adj_mat) * jnp.eye(adj_mat.shape[-1])
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class MPNN(PGN):
  """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    # The adjacency matrix has all its entries set to 1: the graph is fully-connected
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)
  

class MPNN_L1(PGN_L1):
  """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, node_args: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden, node_args)
  
class MPNN_L2(PGN_L2):
  """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, node_args: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden, node_args)
  
class MPNN_L3(PGN_L3):
  """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, node_args: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden, node_args)
  
class MPNN_L2_L3(PGN_L2_L3):
  """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, node_args: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden, node_args)
  
class MPNN_L1_L3(PGN_L1_L3):
  """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, node_args: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden, node_args)

class PGNMask(PGN):
  """Masked Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

  @property
  def inf_bias(self):
    return True

  @property
  def inf_bias_edge(self):
    return True


class MemNetMasked(Processor):
  """Implementation of End-to-End Memory Networks.

  Inspired by the description in https://arxiv.org/abs/1503.08895.
  """

  def __init__(
      self,
      vocab_size: int,
      sentence_size: int,
      linear_output_size: int,
      embedding_size: int = 16,
      memory_size: Optional[int] = 128,
      num_hops: int = 1,
      nonlin: Callable[[Any], Any] = jax.nn.relu,
      apply_embeddings: bool = True,
      init_func: hk.initializers.Initializer = jnp.zeros,
      use_ln: bool = False,
      name: str = 'memnet') -> None:
    """Constructor.

    Args:
      vocab_size: the number of words in the dictionary (each story, query and
        answer come contain symbols coming from this dictionary).
      sentence_size: the dimensionality of each memory.
      linear_output_size: the dimensionality of the output of the last layer
        of the model.
      embedding_size: the dimensionality of the latent space to where all
        memories are projected.
      memory_size: the number of memories provided.
      num_hops: the number of layers in the model.
      nonlin: non-linear transformation applied at the end of each layer.
      apply_embeddings: flag whether to aply embeddings.
      init_func: initialization function for the biases.
      use_ln: whether to use layer normalisation in the model.
      name: the name of the model.
    """
    super().__init__(name=name)
    self._vocab_size = vocab_size
    self._embedding_size = embedding_size
    self._sentence_size = sentence_size
    self._memory_size = memory_size
    self._linear_output_size = linear_output_size
    self._num_hops = num_hops
    self._nonlin = nonlin
    self._apply_embeddings = apply_embeddings
    self._init_func = init_func
    self._use_ln = use_ln
    # Encoding part: i.e. "I" of the paper.
    self._encodings = _position_encoding(sentence_size, embedding_size)

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MemNet inference step."""

    del hidden
    node_and_graph_fts = jnp.concatenate([node_fts, graph_fts[:, None]],
                                         axis=1)
    edge_fts_padded = jnp.pad(edge_fts * adj_mat[..., None],
                              ((0, 0), (0, 1), (0, 1), (0, 0)))
    nxt_hidden = jax.vmap(self._apply, (1), 1)(node_and_graph_fts,
                                               edge_fts_padded)

    # Broadcast hidden state corresponding to graph features across the nodes.
    nxt_hidden = nxt_hidden[:, :-1] + nxt_hidden[:, -1:]
    return nxt_hidden, None  # pytype: disable=bad-return-type  # numpy-scalars

  def _apply(self, queries: _Array, stories: _Array) -> _Array:
    """Apply Memory Network to the queries and stories.

    Args:
      queries: Tensor of shape [batch_size, sentence_size].
      stories: Tensor of shape [batch_size, memory_size, sentence_size].

    Returns:
      Tensor of shape [batch_size, vocab_size].
    """
    if self._apply_embeddings:
      query_biases = hk.get_parameter(
          'query_biases',
          shape=[self._vocab_size - 1, self._embedding_size],
          init=self._init_func)
      stories_biases = hk.get_parameter(
          'stories_biases',
          shape=[self._vocab_size - 1, self._embedding_size],
          init=self._init_func)
      memory_biases = hk.get_parameter(
          'memory_contents',
          shape=[self._memory_size, self._embedding_size],
          init=self._init_func)
      output_biases = hk.get_parameter(
          'output_biases',
          shape=[self._vocab_size - 1, self._embedding_size],
          init=self._init_func)

      nil_word_slot = jnp.zeros([1, self._embedding_size])

    # This is "A" in the paper.
    if self._apply_embeddings:
      stories_biases = jnp.concatenate([stories_biases, nil_word_slot], axis=0)
      memory_embeddings = jnp.take(
          stories_biases, stories.reshape([-1]).astype(jnp.int32),
          axis=0).reshape(list(stories.shape) + [self._embedding_size])
      memory_embeddings = jnp.pad(
          memory_embeddings,
          ((0, 0), (0, self._memory_size - jnp.shape(memory_embeddings)[1]),
           (0, 0), (0, 0)))
      memory = jnp.sum(memory_embeddings * self._encodings, 2) + memory_biases
    else:
      memory = stories

    # This is "B" in the paper. Also, when there are no queries (only
    # sentences), then there these lines are substituted by
    # query_embeddings = 0.1.
    if self._apply_embeddings:
      query_biases = jnp.concatenate([query_biases, nil_word_slot], axis=0)
      query_embeddings = jnp.take(
          query_biases, queries.reshape([-1]).astype(jnp.int32),
          axis=0).reshape(list(queries.shape) + [self._embedding_size])
      # This is "u" in the paper.
      query_input_embedding = jnp.sum(query_embeddings * self._encodings, 1)
    else:
      query_input_embedding = queries

    # This is "C" in the paper.
    if self._apply_embeddings:
      output_biases = jnp.concatenate([output_biases, nil_word_slot], axis=0)
      output_embeddings = jnp.take(
          output_biases, stories.reshape([-1]).astype(jnp.int32),
          axis=0).reshape(list(stories.shape) + [self._embedding_size])
      output_embeddings = jnp.pad(
          output_embeddings,
          ((0, 0), (0, self._memory_size - jnp.shape(output_embeddings)[1]),
           (0, 0), (0, 0)))
      output = jnp.sum(output_embeddings * self._encodings, 2)
    else:
      output = stories

    intermediate_linear = hk.Linear(self._embedding_size, with_bias=False)

    # Output_linear is "H".
    output_linear = hk.Linear(self._linear_output_size, with_bias=False)

    for hop_number in range(self._num_hops):
      query_input_embedding_transposed = jnp.transpose(
          jnp.expand_dims(query_input_embedding, -1), [0, 2, 1])

      # Calculate probabilities.
      probs = jax.nn.softmax(
          jnp.sum(memory * query_input_embedding_transposed, 2))

      # Calculate output of the layer by multiplying by C.
      transposed_probs = jnp.transpose(jnp.expand_dims(probs, -1), [0, 2, 1])
      transposed_output_embeddings = jnp.transpose(output, [0, 2, 1])

      # This is "o" in the paper.
      layer_output = jnp.sum(transposed_output_embeddings * transposed_probs, 2)

      # Finally the answer
      if hop_number == self._num_hops - 1:
        # Please note that in the TF version we apply the final linear layer
        # in all hops and this results in shape mismatches.
        output_layer = output_linear(query_input_embedding + layer_output)
      else:
        output_layer = intermediate_linear(query_input_embedding + layer_output)

      query_input_embedding = output_layer
      if self._nonlin:
        output_layer = self._nonlin(output_layer)

    # This linear here is "W".
    ret = hk.Linear(self._vocab_size, with_bias=False)(output_layer)

    if self._use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    return ret


class MemNetFull(MemNetMasked):
  """Memory Networks with full adjacency matrix."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


ProcessorFactory = Callable[[int], Processor]


def get_processor_factory(kind: str,
                          use_ln: bool,
                          nb_triplet_fts: int,
                          nb_heads: Optional[int] = None,
                          reduction: Optional[_Fn] = jnp.max,
                          num_messages_sample: int = 2,
                          num_nodes_sample: int = 8) -> ProcessorFactory:
  """Returns a processor factory.

  Args:
    kind: One of the available types of processor.
    use_ln: Whether the processor passes the output through a layernorm layer.
    nb_triplet_fts: How many triplet features to compute.
    nb_heads: Number of attention heads for GAT processors.
  Returns:
    A callable that takes an `out_size` parameter (equal to the hidden
    dimension of the network) and returns a processor instance.
  """
  def _factory(out_size: int):
    if kind == 'deepsets':
      processor = DeepSets(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=0
      )
    elif kind == 'gat':
      processor = GAT(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln,
      )
    elif kind == 'gat_full':
      processor = GATFull(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'gatv2':
      processor = GATv2(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'gatv2_full':
      processor = GATv2Full(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'memnet_full':
      processor = MemNetFull(
          vocab_size=out_size,
          sentence_size=out_size,
          linear_output_size=out_size,
      )
    elif kind == 'memnet_masked':
      processor = MemNetMasked(
          vocab_size=out_size,
          sentence_size=out_size,
          linear_output_size=out_size,
      )
    elif kind == 'mpnn':
      processor = MPNN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=0,
      )
    elif kind == 'pgn':
      processor = PGN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=0,
          reduction=reduction
      )
    elif kind == 'pgnlin':
      processor = LinearPGN(
        out_size=out_size,
        msgs_mlp_sizes=[out_size, out_size],
        use_ln=use_ln,
        use_triplets=False,
        nb_triplet_fts=0,
        reduction=reduction
        )
    elif kind == 'pgn_mask':
      processor = PGNMask(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=0,
          reduction=reduction
      )
    elif kind == 'triplet_mpnn':
      processor = MPNN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          reduction=reduction
      )
    elif kind == 'triplet_pgn':
      processor = PGN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          reduction=reduction
      )
    elif kind == 'triplet_pgn_mask':
      processor = PGNMask(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
      )
    elif kind == 'gpgn':
      processor = PGN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    elif kind == 'gpgn_mask':
      processor = PGNMask(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    elif kind == 'gmpnn':
      processor = MPNN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    elif kind == 'triplet_gpgn':
      processor = PGN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    elif kind == 'triplet_gpgn_mask':
      processor = PGNMask(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    elif kind == 'triplet_gmpnn':
      processor = MPNN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
          reduction=reduction
      )
    elif kind == 'mpnn_l1':
      processor = MPNN_L1(
          out_size=out_size,
          activation=jax.nn.relu,
          reduction=jnp.sum,
          arg_reduction=jnp.sum,
          msgs_mlp_sizes=None,
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=nb_triplet_fts,
          gated=False,
          num_messages_sample=num_messages_sample,
          num_nodes_sample=num_nodes_sample,
      )
    elif kind == 'mpnn_l1_max':
      processor = MPNN_L1(
          out_size=out_size,
          activation=jax.nn.relu,
          reduction=jnp.max,
          arg_reduction=jnp.max,
          msgs_mlp_sizes=None,
          use_ln=False,
          use_triplets=False,
          nb_triplet_fts=nb_triplet_fts,
          gated=False,
          num_messages_sample=num_messages_sample,
          num_nodes_sample=num_nodes_sample,
      )
    elif kind == 'mpnn_l2':
      processor = MPNN_L2(
          out_size=out_size,
          reduction=jnp.max,
          arg_reduction=jnp.max,
          msgs_mlp_sizes=None,
          use_ln=False,
          use_triplets=False,
          nb_triplet_fts=nb_triplet_fts,
          gated=False,
      )
    elif kind == 'mpnn_l3':
      processor = MPNN_L3(
          out_size=out_size,
          reduction=jnp.max,
          arg_reduction=jnp.max,
          msgs_mlp_sizes=None,
          use_ln=False,
          use_triplets=False,
          nb_triplet_fts=nb_triplet_fts,
          gated=False,
      )
    elif kind == 'mpnn_l2_l3':
      processor = MPNN_L2_L3(
          out_size=out_size,
          activation=jax.nn.relu,
          reduction=jnp.max,
          arg_reduction=jnp.sum,
          msgs_mlp_sizes=None,
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=nb_triplet_fts,
          gated=False,
          num_messages_sample=num_messages_sample,
          num_nodes_sample=num_nodes_sample,
      )
    elif kind == 'mpnn_l2_l3_max':
      processor = MPNN_L2_L3(
          out_size=out_size,
          activation=jax.nn.relu,
          reduction=jnp.max,
          arg_reduction=jnp.max,
          msgs_mlp_sizes=None,
          use_ln=False,
          use_triplets=False,
          nb_triplet_fts=nb_triplet_fts,
          gated=False,
          num_messages_sample=num_messages_sample,
          num_nodes_sample=num_nodes_sample,
      )
    elif kind == 'mpnn_l1_l3':
      processor = MPNN_L1_L3(
          out_size=out_size,
          activation=jax.nn.relu,
          reduction=jnp.sum,
          arg_reduction=jnp.sum,
          msgs_mlp_sizes=None,
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=nb_triplet_fts,
          gated=False,
          num_messages_sample=num_messages_sample,
          num_nodes_sample=num_nodes_sample,
      )
    elif kind == 'mpnn_l1_l3_max':
      processor = MPNN_L1_L3(
          out_size=out_size,
          activation=jax.nn.relu,
          reduction=jnp.max,
          arg_reduction=jnp.max,
          msgs_mlp_sizes=None,
          use_ln=False,
          use_triplets=False,
          nb_triplet_fts=nb_triplet_fts,
          gated=False,
          num_messages_sample=num_messages_sample,
          num_nodes_sample=num_nodes_sample,
      )
    else:
      raise ValueError('Unexpected processor kind ' + kind)

    return processor

  return _factory


def _position_encoding(sentence_size: int, embedding_size: int) -> np.ndarray:
  """Position Encoding described in section 4.1 [1]."""
  encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
  ls = sentence_size + 1
  le = embedding_size + 1
  for i in range(1, le):
    for j in range(1, ls):
      encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
  encoding = 1 + 4 * encoding / embedding_size / sentence_size
  return np.transpose(encoding)
