# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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

"""Run training of one or more algorithmic tasks from CLRS."""

import functools
import os
import shutil
from typing import Any, Dict, List

from absl import app
from absl import flags
from absl import logging
import clrs
import jax
import numpy as np
import requests
import tensorflow as tf


flags.DEFINE_list('algorithms', ['bfs'], 'Which algorithms to run.')
flags.DEFINE_list('train_lengths', ['4', '7', '11', '13', '16'],
                  'Which training sizes to use. A size of -1 means '
                  'use the benchmark dataset.')
flags.DEFINE_integer('length_needle', -8,
                     'Length of needle for training and validation '
                     '(not testing) in string matching algorithms. '
                     'A negative value randomizes the length for each sample '
                     'between 1 and the opposite of the value. '
                     'A value of 0 means use always 1/4 of the length of '
                     'the haystack (the default sampler behavior).')
flags.DEFINE_integer('seed', 42, 'Random seed to set')

flags.DEFINE_boolean('random_pos', True,
                     'Randomize the pos input common to all algos.')
flags.DEFINE_boolean('enforce_permutations', True,
                     'Whether to enforce permutation-type node pointers.')
flags.DEFINE_boolean('enforce_pred_as_input', True,
                     'Whether to change pred_h hints into pred inputs.')
flags.DEFINE_integer('batch_size', 32, 'Batch size used for training.')
flags.DEFINE_boolean('chunked_training', False,
                     'Whether to use chunking for training.')
flags.DEFINE_integer('chunk_length', 16,
                     'Time chunk length used for training (if '
                     '`chunked_training` is True.')
flags.DEFINE_integer('train_steps', 10000, 'Number of training iterations.')
flags.DEFINE_integer('eval_every', 50, 'Evaluation frequency (in steps).')
flags.DEFINE_integer('test_every', 500, 'Evaluation frequency (in steps).')

flags.DEFINE_integer('hidden_size', 128,
                     'Number of hidden units of the model.')
flags.DEFINE_integer('nb_heads', 1, 'Number of heads for GAT processors')
flags.DEFINE_integer('nb_msg_passing_steps', 1,
                     'Number of message passing steps to run per hint.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate to use.')
flags.DEFINE_float('grad_clip_max_norm', 1.0,
                   'Gradient clipping by norm. 0.0 disables grad clipping')
flags.DEFINE_float('dropout_prob', 0.0, 'Dropout rate to use.')
flags.DEFINE_float('hint_teacher_forcing', 0.0,
                   'Probability that ground-truth teacher hints are encoded '
                   'during training instead of predicted hints. Only '
                   'pertinent in encoded_decoded modes.')
flags.DEFINE_enum('hint_mode', 'encoded_decoded',
                  ['encoded_decoded', 'decoded_only', 'none'],
                  'How should hints be used? Note, each mode defines a '
                  'separate task, with various difficulties. `encoded_decoded` '
                  'requires the model to explicitly materialise hint sequences '
                  'and therefore is hardest, but also most aligned to the '
                  'underlying algorithmic rule. Hence, `encoded_decoded` '
                  'should be treated as the default mode for our benchmark. '
                  'In `decoded_only`, hints are only used for defining '
                  'reconstruction losses. Often, this will perform well, but '
                  'note that we currently do not make any efforts to '
                  'counterbalance the various hint losses. Hence, for certain '
                  'tasks, the best performance will now be achievable with no '
                  'hint usage at all (`none`).')
flags.DEFINE_enum('hint_repred_mode', 'soft', ['soft', 'hard', 'hard_on_eval'],
                  'How to process predicted hints when fed back as inputs.'
                  'In soft mode, we use softmaxes for categoricals, pointers '
                  'and mask_one, and sigmoids for masks. '
                  'In hard mode, we use argmax instead of softmax, and hard '
                  'thresholding of masks. '
                  'In hard_on_eval mode, soft mode is '
                  'used for training and hard mode is used for evaluation.')
flags.DEFINE_boolean('use_ln', True,
                     'Whether to use layer normalisation in the processor.')
flags.DEFINE_boolean('use_lstm', False,
                     'Whether to insert an LSTM after message passing.')
flags.DEFINE_integer('nb_triplet_fts', 8,
                     'How many triplet features to compute?')

flags.DEFINE_enum('encoder_init', 'xavier_on_scalars',
                  ['default', 'xavier_on_scalars'],
                  'Initialiser to use for the encoders.')
flags.DEFINE_enum('processor_type', 'triplet_mpnn',
                  ['deepsets', 'mpnn', 'pgn', 'pgnlin', 'pgn_mask',
                   'triplet_mpnn', 'triplet_pgn', 'triplet_pgn_mask',
                   'gat', 'gatv2', 'gat_full', 'gatv2_full',
                   'gpgn', 'gpgn_mask', 'gmpnn',
                   'triplet_gpgn', 'triplet_gpgn_mask', 'triplet_gmpnn', 
                   'mpnn_l1', 'mpnn_l1_max', 'mpnn_l1_residual', 'mpnn_l1_regularised', 
                   'mpnn_l1_regularised_max', 'mpnn_l2', 'mpnn_l3'],
                  'Processor type to use as the network P.')

flags.DEFINE_string('checkpoint_path', '/tmp/CLRS30_v1.0.0',
                    'Path in which checkpoints are saved.')
flags.DEFINE_string('dataset_path', '/tmp/CLRS30_v1.0.0',
                    'Path in which dataset is stored.')
flags.DEFINE_boolean('freeze_processor', False,
                     'Whether to freeze the processor of the model.')
####
# Latent representation flags
####
flags.DEFINE_boolean('test', False,
                     'Skip training and restore best model')
flags.DEFINE_string('sample_strat', None,
                     'Sample augmentation strategy for Bellman Ford')
flags.DEFINE_enum('noise_injection_strategy', 'Noisefree',
                  ['Noisefree', 'Uniform', 'Directional', 'Project', 'Discard', 'Corrupt'],
                  'Type of destructive noise to apply during message passing.')
flags.DEFINE_float('decay', 1.0,
                     'Perform exponential decay inside nets.')
flags.DEFINE_boolean('softmax_reduction', False, 'Use softmax reduction in processor instead of max, for training.')
####
# Asynchrony flags
####
flags.DEFINE_float('regularisation_weight', 0.0,
                   'Weight given to regularisation loss')
flags.DEFINE_boolean('bound_regularisation_loss', False,
                     'Whether to bound the regularisation loss to not grow too much in the early stages of training.')
flags.DEFINE_float('max_proportion_regularisation', 0.2,
                   'Regularisation loss cannot be higher than this proportion of the quality loss.')

flags.DEFINE_integer('num_messages_sample', 2,
                   'Number of messages to sample for each node to compute asynchrony losses.')
flags.DEFINE_integer('num_nodes_sample', 2,
                   'Number of nodes to sample to compute asynchrony losses.')


FLAGS = flags.FLAGS


PRED_AS_INPUT_ALGOS = [
    'binary_search',
    'minimum',
    'find_maximum_subarray',
    'find_maximum_subarray_kadane',
    'matrix_chain_order',
    'lcs_length',
    'optimal_bst',
    'activity_selector',
    'task_scheduling',
    'naive_string_matcher',
    'kmp_matcher',
    'jarvis_march']


def unpack(v):
  try:
    return v.item()  # DeviceArray
  except (AttributeError, ValueError):
    return v


def _iterate_sampler(sampler, batch_size):
  while True:
    yield sampler.next(batch_size)


def _maybe_download_dataset(dataset_path):
  """Download CLRS30_v1.0.0 dataset if needed."""
  dataset_folder = os.path.join(dataset_path, clrs.get_clrs_folder())
  if os.path.isdir(dataset_folder):
    logging.info('Dataset found at %s. Skipping download.', dataset_folder)
    return dataset_folder
  logging.info('Dataset not found in %s. Downloading...', dataset_folder)

  clrs_url = clrs.get_dataset_gcp_url()
  request = requests.get(clrs_url, allow_redirects=True)
  clrs_file = os.path.join(dataset_path, os.path.basename(clrs_url))
  os.makedirs(dataset_folder)
  open(clrs_file, 'wb').write(request.content)
  shutil.unpack_archive(clrs_file, extract_dir=dataset_folder)
  os.remove(clrs_file)
  return dataset_folder


def make_sampler(length: int,
                 rng: Any,
                 algorithm: str,
                 split: str,
                 batch_size: int,
                 multiplier: int,
                 randomize_pos: bool,
                 enforce_pred_as_input: bool,
                 enforce_permutations: bool,
                 chunked: bool,
                 chunk_length: int,
                 sampler_kwargs: Dict[str, Any]):
  """Create a sampler with given options.

  Args:
    length: Size of samples (i.e., number of nodes in the graph).
      A length of -1 will mean that the benchmark
      dataset (for the given split) is used. Positive sizes will instantiate
      samplers of the corresponding size.
    rng: Numpy random state.
    algorithm: The name of the algorithm to sample from.
    split: 'train', 'val' or 'test'.
    batch_size: Samples per batch.
    multiplier: Integer multiplier for the number of samples in the dataset,
      only used for positive sizes. Negative multiplier means infinite samples.
    randomize_pos: Whether to randomize the `pos` input.
    enforce_pred_as_input: Whether to convert fixed pred_h hints to inputs.
    enforce_permutations: Whether to enforce permutation pointers.
    chunked: Whether to chunk the dataset.
    chunk_length: Unroll length of chunks, if `chunked` is True.
    sampler_kwargs: Extra args passed to the sampler.
  Returns:
    A sampler (iterator), the number of samples in the iterator (negative
    if infinite samples), and the spec.
  """
  if length < 0:  # load from file
    dataset_folder = _maybe_download_dataset(FLAGS.dataset_path)
    sampler, num_samples, spec = clrs.create_dataset(folder=dataset_folder,
                                                     algorithm=algorithm,
                                                     batch_size=batch_size,
                                                     split=split)
    sampler = sampler.as_numpy_iterator()
  else:
    num_samples = clrs.CLRS30[split]['num_samples'] * multiplier
    sampler, spec = clrs.build_sampler(
        algorithm,
        seed=rng.randint(2**31),
        num_samples=num_samples,
        length=length,
        **sampler_kwargs,
        )
    sampler = _iterate_sampler(sampler, batch_size)

  if randomize_pos:
    sampler = clrs.process_random_pos(sampler, rng)
  if enforce_pred_as_input and algorithm in PRED_AS_INPUT_ALGOS:
    spec, sampler = clrs.process_pred_as_input(spec, sampler)
  spec, sampler = clrs.process_permutations(spec, sampler, enforce_permutations)
  if chunked:
    sampler = clrs.chunkify(sampler, chunk_length)
  return sampler, num_samples, spec


def make_multi_sampler(sizes, rng, **kwargs):
  """Create a sampler with cycling sample sizes."""
  ss = []
  tot_samples = 0
  for length in sizes:
    sampler, num_samples, spec = make_sampler(length, rng, **kwargs)
    ss.append(sampler)
    tot_samples += num_samples

  def cycle_samplers():
    while True:
      for s in ss:
        yield next(s)
  return cycle_samplers(), tot_samples, spec


def _concat(dps, axis):
  return jax.tree_util.tree_map(lambda *x: np.concatenate(x, axis), *dps)


def collect_and_eval(sampler, predict_fn, sample_count, rng_key, extras):
  """Collect batches of output and hint preds and evaluate them."""
  processed_samples = 0
  preds = []
  outputs = []
  while processed_samples < sample_count:
    feedback = next(sampler)
    batch_size = feedback.outputs[0].data.shape[0]
    outputs.append(feedback.outputs)
    new_rng_key, rng_key = jax.random.split(rng_key)
    cur_preds, _, trajs = predict_fn(new_rng_key, feedback.features)
    preds.append(cur_preds)
    processed_samples += batch_size
  outputs = _concat(outputs, axis=0)
  preds = _concat(preds, axis=0)
  out = clrs.evaluate(outputs, preds)
  if extras:
    out.update(extras)
  return {k: unpack(v) for k, v in out.items()}

def fill(trajs):
  n = max([len(x) for x in trajs])
  for x in trajs:
    while len(x) < n:
      x.append(x[-1])  # Duplicate last trajectory to align lengths
  return trajs

def fill_trajectories(trajs):
  n = max([x.shape[0] for x in trajs])
  trajs = [jax.numpy.concatenate([x, jax.numpy.repeat(x[-1:, ...], n - x.shape[0], axis=0)], axis=0) if n - x.shape[0] > 0 else x for x in trajs]
  return trajs

def dump_trajectories(sampler, predict_fn, sample_count, rng_key):
  """Dump trajectories of datapoints"""
  processed_samples = 0
  trajs = []
  inputs = []
  preds = []
  outputs = []
  lengths = []
  hints = []
  while processed_samples < sample_count:
    feedback = next(sampler)
    batch_size = feedback.outputs[0].data.shape[0]
    outputs.append(feedback.outputs)
    new_rng_key, rng_key = jax.random.split(rng_key)
    cur_preds, cur_hints, cur_trajs = predict_fn(new_rng_key, feedback.features,
                                                 return_hints=True)
    preds.append(cur_preds)
    hints.append(cur_hints)
    trajs.append(cur_trajs)
    inputs.append(feedback)
    lengths.append(feedback.features[2])
    processed_samples += batch_size
  preds = _concat(preds, axis=0)
  outputs = _concat(outputs, axis=0)
  trajs = _concat(fill_trajectories(trajs), axis=1)
  # Reduce over the node dimension
  trajs = jax.numpy.max(trajs, axis=2)
  # The dimensions are T x N x D
  trajs = trajs.transpose(1,0,2)
  hints = _concat(fill(hints), axis=0)
  inputs = _concat(inputs, axis=0)
  lengths = jax.numpy.array(lengths).flatten().astype(int)
  out = clrs.evaluate_each(outputs, preds)
  # graph_fts = jax.numpy.asarray([d['node'] for d in trajs]).transpose(1, 2, 0, 3)
  # graph_fts = jax.numpy.asarray([d['graph'] for d in trajs]).transpose(1, 0, 2)
  graph_fts = np.zeros_like(lengths)
  return lengths, trajs, out, inputs, preds, hints


def create_samplers(rng, train_lengths: List[int]):
  """Create all the samplers."""
  train_samplers = []
  val_samplers = []
  val_sample_counts = []
  test_samplers = []
  test_sample_counts = []
  specil_samplers = []
  specil_sample_counts = []
  spec_list = []

  for algo_idx, algorithm in enumerate(FLAGS.algorithms):
    # Make full dataset pipeline run on CPU (including prefetching).
    with tf.device('/cpu:0'):

      if algorithm in ['naive_string_matcher', 'kmp_matcher']:
        # Fixed haystack + needle; variability will be in needle
        # Still, for chunked training, we maintain as many samplers
        # as train lengths, since, for each length there is a separate state,
        # and we must keep the 1:1 relationship between states and samplers.
        max_length = max(train_lengths)
        if max_length > 0:  # if < 0, we are using the benchmark data
          max_length = (max_length * 5) // 4
        train_lengths = [max_length]
        if FLAGS.chunked_training:
          train_lengths = train_lengths * len(train_lengths)

      logging.info('Creating samplers for algo %s', algorithm)

      p = tuple([0.1 + 0.1 * i for i in range(9)])
      if p and algorithm in ['articulation_points', 'bridges',
                             'mst_kruskal', 'bipartite_matching']:
        # Choose a lower connection probability for the above algorithms,
        # otherwise trajectories are very long
        p = tuple(np.array(p) / 2)
      length_needle = FLAGS.length_needle
      sampler_kwargs = dict(p=p, length_needle=length_needle)
      if length_needle == 0:
        sampler_kwargs.pop('length_needle')

      common_sampler_args = dict(
          algorithm=FLAGS.algorithms[algo_idx],
          rng=rng,
          enforce_pred_as_input=FLAGS.enforce_pred_as_input,
          enforce_permutations=FLAGS.enforce_permutations,
          chunk_length=FLAGS.chunk_length,
          )

      train_args = dict(sizes=train_lengths,
                        split='train',
                        batch_size=FLAGS.batch_size,
                        multiplier=-1,
                        randomize_pos=FLAGS.random_pos,
                        chunked=FLAGS.chunked_training,
                        sampler_kwargs=sampler_kwargs,
                        **common_sampler_args)
      train_sampler, _, spec = make_multi_sampler(**train_args)

      mult = clrs.CLRS_30_ALGS_SETTINGS[algorithm]['num_samples_multiplier']
      val_args = dict(sizes=[np.amax(train_lengths)],
                      split='val',
                      batch_size=32,
                      multiplier=2 * mult,
                      randomize_pos=FLAGS.random_pos,
                      chunked=False,
                      sampler_kwargs=sampler_kwargs,
                      **common_sampler_args)
      val_sampler, val_samples, spec = make_multi_sampler(**val_args)

      test_args = dict(sizes=[-1],
                       split='test',
                       batch_size=32,
                       multiplier=2 * mult,
                       randomize_pos=False,
                       chunked=False,
                       sampler_kwargs={},
                       **common_sampler_args)
      test_sampler, test_samples, spec = make_multi_sampler(**test_args)


      specil_args = dict(sizes=[64],
                       split='test',
                       batch_size=32,
                       multiplier=2**9  * mult,
                       randomize_pos=False,
                       chunked=False,
                       sampler_kwargs=dict(specil=FLAGS.sample_strat, force_otf=True),
                       **common_sampler_args)
      specil_sampler, specil_samples, spec = make_multi_sampler(**specil_args)


    spec_list.append(spec)
    train_samplers.append(train_sampler)
    val_samplers.append(val_sampler)
    val_sample_counts.append(val_samples)
    test_samplers.append(test_sampler)
    test_sample_counts.append(test_samples)
    specil_samplers.append(specil_sampler)
    specil_sample_counts.append(specil_samples)

  return (train_samplers,
          val_samplers, val_sample_counts,
          test_samplers, test_sample_counts,
          specil_samplers, specil_sample_counts,
          spec_list)


def main(unused_argv):
  if FLAGS.hint_mode == 'encoded_decoded':
    encode_hints = True
    decode_hints = True
  elif FLAGS.hint_mode == 'decoded_only':
    encode_hints = False
    decode_hints = True
  elif FLAGS.hint_mode == 'none':
    encode_hints = False
    decode_hints = False
  else:
    raise ValueError('Hint mode not in {encoded_decoded, decoded_only, none}.')

  print(FLAGS.seed)
  print(FLAGS.algorithms)
  print(FLAGS.decay)
  print(FLAGS.softmax_reduction)
  print(FLAGS.train_steps)


  train_lengths = [int(x) for x in FLAGS.train_lengths]

  rng = np.random.RandomState(FLAGS.seed)
  rng_key = jax.random.PRNGKey(rng.randint(2**31))

  # Create samplers
  (train_samplers,
   val_samplers, val_sample_counts,
   test_samplers, test_sample_counts,
   special_samplers, special_sample_counts,
   spec_list) = create_samplers(rng, train_lengths)

  processor_factory = clrs.get_processor_factory(
      FLAGS.processor_type,
      use_ln=FLAGS.use_ln,
      nb_triplet_fts=FLAGS.nb_triplet_fts,
      nb_heads=FLAGS.nb_heads,
      num_messages_sample=FLAGS.num_messages_sample,
      num_nodes_sample=FLAGS.num_nodes_sample,
  )
  model_params = dict(
      processor_factory=processor_factory,
      hidden_dim=FLAGS.hidden_size,
      encode_hints=encode_hints,
      decode_hints=decode_hints,
      encoder_init=FLAGS.encoder_init,
      use_lstm=FLAGS.use_lstm,
      learning_rate=FLAGS.learning_rate,
      grad_clip_max_norm=FLAGS.grad_clip_max_norm,
      checkpoint_path=FLAGS.checkpoint_path,
      freeze_processor=FLAGS.freeze_processor,
      dropout_prob=FLAGS.dropout_prob,
      hint_teacher_forcing=FLAGS.hint_teacher_forcing,
      hint_repred_mode=FLAGS.hint_repred_mode,
      nb_msg_passing_steps=FLAGS.nb_msg_passing_steps,
      noise_mode=FLAGS.noise_injection_strategy,
      decay=FLAGS.decay,
      regularisation_weight=FLAGS.regularisation_weight,
      bound_regularisation_loss=FLAGS.bound_regularisation_loss,
      max_proportion_regularisation=FLAGS.max_proportion_regularisation,
      )

  eval_model = clrs.models.BaselineModel(
      spec=spec_list,
      dummy_trajectory=[next(t) for t in val_samplers],
      **model_params
  )
  if FLAGS.chunked_training:
    train_model = clrs.models.BaselineModelChunked(
        spec=spec_list,
        dummy_trajectory=[next(t) for t in train_samplers],
        **model_params
        )
  else:
    train_model = eval_model

  if FLAGS.softmax_reduction:
    # Modify the train model to softmax.
    # WARN Incompatible with chunked training
    processor_factory_softmax = clrs.get_processor_factory(
      FLAGS.processor_type,
      use_ln=FLAGS.use_ln,
      nb_triplet_fts=FLAGS.nb_triplet_fts,
      nb_heads=FLAGS.nb_heads,
      reduction=jax.nn.softmax
      )
    model_params['processor_factory'] = processor_factory_softmax
    train_model = clrs.models.BaselineModel(
      spec=spec_list,
      dummy_trajectory=[next(t) for t in val_samplers],
      **model_params
      )

  if FLAGS.test:
    with open(f"{FLAGS.checkpoint_path}/best.pkl", 'rb') as file:
      import pickle
      best = pickle.load(file)
      eval_model.params = best['params']
      eval_model.opt_state = best['opt_state']
  else:
    # Training loop.
    best_score = -1.0
    current_train_items = [0] * len(FLAGS.algorithms)
    step = 0
    next_eval = 0
    # Make sure scores improve on first step, but not overcome best score
    # until all algos have had at least one evaluation.
    val_scores = [-99999.9] * len(FLAGS.algorithms)
    length_idx = 0

    while step < FLAGS.train_steps:
      feedback_list = [next(t) for t in train_samplers]

      # Initialize model.
      if step == 0:
        all_features = [f.features for f in feedback_list]
        if FLAGS.chunked_training:
          # We need to initialize the model with samples of all lengths for
          # all algorithms. Also, we need to make sure that the order of these
          # sample sizes is the same as the order of the actual training sizes.
          all_length_features = [all_features] + [
              [next(t).features for t in train_samplers]
              for _ in range(len(train_lengths))]
          train_model.init(all_length_features[:-1], FLAGS.seed + 1)
        else:
          train_model.init(all_features, FLAGS.seed + 1)

      # Training step.
      for algo_idx in range(len(train_samplers)):
        feedback = feedback_list[algo_idx]
        rng_key, new_rng_key = jax.random.split(rng_key)
        if FLAGS.chunked_training:
          # In chunked training, we must indicate which training length we are
          # using, so the model uses the correct state.
          length_and_algo_idx = (length_idx, algo_idx)
        else:
          # In non-chunked training, all training lengths can be treated equally,
          # since there is no state to maintain between batches.
          length_and_algo_idx = algo_idx
        cur_loss = train_model.feedback(rng_key, feedback, length_and_algo_idx)
        rng_key = new_rng_key

      if FLAGS.chunked_training:
        examples_in_chunk = np.sum(feedback.features.is_last).item()
      else:
        examples_in_chunk = len(feedback.features.lengths)
      current_train_items[algo_idx] += examples_in_chunk
      logging.info('Algo %s step %i current loss %f, current_train_items %i.',
                   FLAGS.algorithms[algo_idx], step,
                   cur_loss, current_train_items[algo_idx])

      # Periodically evaluate model
      if step >= next_eval:
        eval_model.params = train_model.params
        for algo_idx in range(len(train_samplers)):
          common_extras = {'examples_seen': current_train_items[algo_idx],
                           'step': step,
                           'algorithm': FLAGS.algorithms[algo_idx]}

          # Validation info.
          new_rng_key, rng_key = jax.random.split(rng_key)
          val_stats = collect_and_eval(
              val_samplers[algo_idx],
              functools.partial(eval_model.predict, algorithm_index=algo_idx),
              val_sample_counts[algo_idx],
              new_rng_key,
              extras=common_extras)
          logging.info('(val) algo %s step %d: %s',
                       FLAGS.algorithms[algo_idx], step, val_stats)
          val_scores[algo_idx] = val_stats['score']

        next_eval += FLAGS.eval_every

        # logging.info(f"Inv T: {train_model.params['net/linear_pgn_clrs_processor']['temp']:.3f}")

        # If best total score, update best checkpoint.
        # Also save a best checkpoint on the first step.
        msg = (f'best avg val score was '
               f'{best_score/len(FLAGS.algorithms):.3f}, '
               f'current avg val score is {np.mean(val_scores):.3f}, '
               f'val scores are: ')
        msg += ', '.join(
            ['%s: %.3f' % (x, y) for (x, y) in zip(FLAGS.algorithms, val_scores)])
        if (sum(val_scores) >  best_score) or step == 0:
          best_score = sum(val_scores)
          logging.info('Checkpointing best model, %s', msg)
          train_model.save_model('best.pkl')
        else:
          logging.info('Not saving new best model, %s', msg)

      step += 1
      length_idx = (length_idx + 1) % len(train_lengths)

    logging.info('Restoring best model from checkpoint...')
    eval_model.restore_model('best.pkl', only_load_processor=False)

    for algo_idx in range(len(train_samplers)):
      common_extras = {'examples_seen': current_train_items[algo_idx],
                       'step': step,
                       'algorithm': FLAGS.algorithms[algo_idx]}

      new_rng_key, rng_key = jax.random.split(rng_key)
      test_stats = collect_and_eval(
          test_samplers[algo_idx],
          functools.partial(eval_model.predict, algorithm_index=algo_idx),
          test_sample_counts[algo_idx],
          new_rng_key,
          extras=common_extras)
      logging.info('(test) algo %s : %s', FLAGS.algorithms[algo_idx], test_stats)


  if not FLAGS.test:
    return

  specil_model = clrs.models.BaselineModel(
      spec=spec_list,
      dummy_trajectory=[next(t) for t in test_samplers],
      **model_params
  )

  specil_model.params = eval_model.params

  for algo_idx in range(len(special_samplers)):
    new_rng_key, rng_key = jax.random.split(rng_key)
    lengths, trajs, stats, feedback, preds, hints = dump_trajectories(
        special_samplers[algo_idx],
        functools.partial(specil_model.predict, algorithm_index=algo_idx, return_all_features=True),
        special_sample_counts[algo_idx],
        new_rng_key)
    logging.info('(test) algo %s : %s', FLAGS.algorithms[algo_idx], {k:100*np.mean(v) for k,v in stats.items()})
    logging.info(f'var: {100*np.std(stats["pi"])}')
    for i in range(1,20):
      logging.info(f'{i}: count={np.sum(lengths == i)} {100*np.mean(stats["pi"][lengths == i]):.2f} +/- '
                   f'{100*np.std(stats["pi"][lengths == i]):.2f}%')
    trajs_dump = {'trajs': trajs, 'score': stats['pi'], 'lengths': lengths,
                  'inputs': feedback.features.inputs,
                  'outputs': feedback.outputs,
                  'hints': hints,
                  }

  np.savez('trajs.npz', **trajs_dump)

  with open('trajs.pkl', 'wb') as file:
    pickle.dump({
      'trajs': trajs,
      'score': stats,
      'feedback': feedback,
      'preds': preds
      }, file, protocol=-1)


  logging.info('Done!')


if __name__ == '__main__':
  app.run(main)
