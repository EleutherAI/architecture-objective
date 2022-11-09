# Copyright 2022 The T5 Authors.
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

"""Preprocessors for T5 Tasks."""
# TODO(adarob): Move some of the more general preprocessors to seqio.

import collections
import functools
import math
import re
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, Union
import uuid

from absl import logging
import babel
import gin
import seqio
import tensorflow.compat.v2 as tf

from t5.data import preprocessors, utils

# We disable no-value-for-parameter since the seqio.map_over_dataset leads to
# a false positive when seeds are provided.
# pylint:disable=no-value-for-parameter
AUTOTUNE = tf.data.experimental.AUTOTUNE

FeatureType = Mapping[str, tf.Tensor]

rekey = seqio.preprocessors.rekey
tokenize = seqio.preprocessors.tokenize



def mlm_lm_processor(dataset,
                    sequence_length,
                    output_features,
                    mean_noise_span_length=3.0,
                    noise_density=0.15,
                    input_feature_key='inputs',
                    merge_examples_to_reduce_padding=True,
                    reserved_for_packing=None,
                    passthrough_feature_keys: Optional[Sequence[str]] = None):
  """Final pretraining objective used in Raffel et al., 2019.

  Args:
    dataset: A tf.data.Dataset with dictionaries containing the key
      `input_feature_key`.
    sequence_length: dict mapping of feature key to int length for that feature.
    output_features: mapping of keys to features.
    mean_noise_span_length: the mean number of tokens per masked span per
      example.
    noise_density: what fraction of the tokens to mask.
    input_feature_key: which feature to use from the dataset as the input text
      tokens.
    merge_examples_to_reduce_padding: if True, combines multiple input examples
      to reduce padding.
    reserved_for_packing: if specified, reduces the desired inputs length by the
      specified amount to enable multiple examples to be packed together
      downstream.
    passthrough_feature_keys: a sequence of feature names that should be passed
      through to the output of this preprocessor. eg: ["tokens"]. Only
      supported if `merge_examples_to_reduce_padding` is set to False.

  Returns:
    a dataset
  """
  inputs_length = sequence_length[input_feature_key]
  if reserved_for_packing:
    inputs_length -= reserved_for_packing

  input_length, targets_length = random_spans_helper(
      extra_tokens_per_span_inputs=1,
      extra_tokens_per_span_targets=1,
      inputs_length=inputs_length,
      mean_noise_span_length=mean_noise_span_length,
      noise_density=noise_density)

  if sequence_length['targets'] < targets_length:
    raise ValueError(
        f'Expected targets length for span corruption ({targets_length}) is '
        f'greater than configured targets length '
        f"({sequence_length['targets']})")

  ds = dataset
  ds = select_random_chunk(
      ds,
      output_features=output_features,
      feature_key='targets',
      max_length=65536,
      passthrough_feature_keys=passthrough_feature_keys)
  if merge_examples_to_reduce_padding:
    if passthrough_feature_keys:
      raise ValueError('passthrough_feature_keys not supported with '
                       'merge_examples_to_reduce_padding=True. '
                       f'Got: {passthrough_feature_keys}')
    ds = reduce_concat_tokens(ds, feature_key='targets', batch_size=128)
  ds = split_tokens(
      ds,
      feature_key='targets',
      min_tokens_per_segment=None,
      max_tokens_per_segment=input_length,
      passthrough_feature_keys=passthrough_feature_keys)
  ds = denoise(
      ds,
      output_features,
      inputs_fn=noise_span_to_unique_sentinel,
      targets_fn=nonnoise_span_to_unique_sentinel,
      noise_density=noise_density,
      noise_mask_fn=functools.partial(
          random_spans_noise_mask,
          mean_noise_span_length=mean_noise_span_length),
      input_feature_key=input_feature_key,
      passthrough_feature_keys=passthrough_feature_keys)
  return ds


# TODO(adarob): Add a test.
def iid_denoising(dataset, sequence_length, output_features):
  """Baseline pretraining objective used in Raffel et al., 2019."""
  ds = dataset
  ds = select_random_chunk(ds, output_features=output_features,
                           feature_key='targets', max_length=65536)
  ds = reduce_concat_tokens(ds, feature_key='targets', batch_size=128)
  ds = split_tokens_to_inputs_length(ds, output_features=output_features,
                                     sequence_length=sequence_length)
  ds = denoise(
      ds,
      output_features,
      inputs_fn=noise_span_to_unique_sentinel,
      targets_fn=nonnoise_span_to_unique_sentinel,
      noise_density=0.15,
      noise_mask_fn=iid_noise_mask
  )
  return ds


def prefix_lm(dataset, sequence_length, output_features):
  """Prefix language modeling objective used in Raffel et al. 2019."""
  ds = dataset
  ds = select_random_chunk(ds, output_features=output_features,
                           feature_key='targets', max_length=65536)
  ds = split_tokens_to_inputs_length(ds, output_features=output_features,
                                     sequence_length=sequence_length)
  ds = denoise(
      ds,
      output_features,
      inputs_fn=drop_nonnoise_tokens,
      targets_fn=drop_noise_tokens,
      noise_density=0.5,
      noise_mask_fn=random_prefix_noise_mask,
  )
  return ds


def full_lm(dataset, sequence_length, output_features):
  """Full language modeling objective with EOS only at document boundaries."""
  ds = dataset
  ds = select_random_chunk(ds, output_features=output_features,
                           feature_key='targets', max_length=65536)
  ds = seqio.preprocessors.append_eos(ds, output_features)
  ds = reduce_concat_tokens(ds, feature_key='targets', batch_size=128)
  # Don't use `split_tokens_to_targets_length` since we've alrady added EOS.
  ds = split_tokens(ds, max_tokens_per_segment=sequence_length['targets'])
  return ds


def single_example_select_random_chunk(
    features: FeatureType,
    seed: tf.Tensor,
    output_features: Mapping[str, seqio.Feature],
    max_length: Optional[int] = None,
    feature_key: str = 'targets',
    additional_feature_keys: Optional[Sequence[str]] = None,
    passthrough_feature_keys: Optional[Sequence[str]] = None,
    sequence_length: Optional[Mapping[str, int]] = None,
    uniform_random_start: bool = False,
    min_length: Optional[int] = None) -> FeatureType:
  """Token-preprocessor to extract one span of at most `max_length` tokens.

  If the token sequence is longer than `max_length`, then we return a random
  subsequence.  Otherwise, we return the full sequence.

  This is generally followed by split_tokens.

  Args:
    features: Single example with `feature_key` containing a tokenized sequence.
    seed: Random seed to be used.
    output_features: Mapping of keys to features.
    max_length: Typically specified in gin configs, takes priority over
      sequence_length.
    feature_key: Which feature to use from the dataset.
    additional_feature_keys: Additional features to use. The same chunk will be
      selected from these features as from the one specified in feature_key,
      so they should all have the same length.
    passthrough_feature_keys: Additional keys to pass through unchanged.
    sequence_length: Used if max_length is not specified. Typically passed in
      by the data pipeline. feature_key will be used to select the length.
    uniform_random_start: If True, will select a starting point in
      [-max_length + 1, n_tokens). If False, will select one of a set of chunks
      offset by max_length. Both of these starting points try to ensure each
      token has an equal probability of being included.
    min_length: If specified, lengths of chunks will be selected uniformly at
      random from [min_length, max_length]. Note that chunks can end up shorter
      than min_length if at the beginning or end of the sequence.

  Returns:
    The features of the selected chunk.
  """
  if passthrough_feature_keys:
    chunk_keys = set([feature_key] + (additional_feature_keys or []))
    overlap_keys = chunk_keys & set(passthrough_feature_keys)
    if overlap_keys:
      raise ValueError(
          f'chunk keys {overlap_keys} also included in passthrough keys')

  if max_length is None and sequence_length is not None:
    max_length = sequence_length[feature_key]
    if output_features[feature_key].add_eos:
      # Leave room to insert an EOS token.
      max_length -= 1
  if max_length is None:
    raise ValueError('Must specify max_length or sequence_length.')

  seeds = tf.unstack(tf.random.experimental.stateless_split(seed))
  tokens = features[feature_key]
  n_tokens = tf.shape(tokens)[0]
  if min_length is not None:
    length = tf.random.stateless_uniform([],
                                         minval=min_length,
                                         maxval=max_length,
                                         dtype=tf.int32,
                                         seed=seeds[0])
  else:
    length = max_length
  if uniform_random_start:
    start = tf.random.stateless_uniform(
        [],
        minval=-length + 1,  # pylint:disable=invalid-unary-operand-type
        maxval=n_tokens,
        dtype=tf.int32,
        seed=seeds[1])
    end = tf.minimum(start + length, n_tokens)
    start = tf.maximum(start, 0)
  else:
    num_segments = tf.cast(
        tf.math.ceil(
            tf.cast(n_tokens, tf.float32) / tf.cast(length, tf.float32)),
        tf.int32)
    start = length * tf.random.stateless_uniform(
        [], maxval=num_segments, dtype=tf.int32, seed=seeds[1])
    end = tf.minimum(start + length, n_tokens)
  chunk = {feature_key: tokens[start:end]}
  if additional_feature_keys is not None:
    for k in additional_feature_keys:
      with tf.control_dependencies([
          tf.assert_equal(
              tf.shape(tokens)[0],
              tf.shape(features[k])[0],
              message=(f'Additional feature {k} is not the same size as '
                       f'{feature_key} along axis 0 in select_random_chunk().'))
      ]):
        chunk[k] = features[k][start:end]
  if passthrough_feature_keys is not None:
    for k in passthrough_feature_keys:
      chunk[k] = features[k]
  return chunk


@gin.configurable
def select_random_chunk(dataset: tf.data.Dataset,
                        output_features: Mapping[str, seqio.Feature],
                        max_length: Optional[int] = None,
                        feature_key: str = 'targets',
                        additional_feature_keys: Optional[Sequence[str]] = None,
                        passthrough_feature_keys: Optional[
                            Sequence[str]] = None,
                        sequence_length: Optional[Mapping[str, int]] = None,
                        uniform_random_start: bool = False,
                        min_length: Optional[int] = None,
                        **unused_kwargs) -> tf.data.Dataset:
  """SeqIO wrapper for single_example_select_random_chunk()."""

  @seqio.map_over_dataset(num_seeds=1)
  def _my_fn(x, seed):
    return single_example_select_random_chunk(
        x,
        seed,
        output_features=output_features,
        max_length=max_length,
        feature_key=feature_key,
        additional_feature_keys=additional_feature_keys,
        passthrough_feature_keys=passthrough_feature_keys,
        sequence_length=sequence_length,
        uniform_random_start=uniform_random_start,
        min_length=min_length)

  # Filter empty examples.
  dataset = dataset.filter(lambda x: tf.not_equal(tf.size(x[feature_key]), 0))
  return _my_fn(dataset)


@gin.configurable
def reduce_concat_tokens(dataset,
                         feature_key='targets',
                         batch_size=128,
                         **unused_kwargs):
  """Token-preprocessor to concatenate multiple unrelated documents.

  If we want to generate examples of exactly the right length,
  (to avoid wasting space on padding), then we use this function, folowed by
  split_tokens.

  Args:
    dataset: a tf.data.Dataset with dictionaries containing the key feature_key.
    feature_key: an string
    batch_size: an integer - how many documents to concatenate into one

  Returns:
    a dataset
  """
  dataset = dataset.map(
      lambda x: {feature_key: x[feature_key]}, num_parallel_calls=AUTOTUNE)
  dataset = dataset.padded_batch(batch_size, padded_shapes={feature_key: [-1]})
  def _my_fn(x):
    tokens = tf.reshape(x[feature_key], [-1])
    # strip padding
    tokens = tf.boolean_mask(tokens, tf.cast(tokens, tf.bool))
    return {feature_key: tokens}

  return dataset.map(_my_fn, num_parallel_calls=AUTOTUNE)


@seqio.map_over_dataset
def trim_tokens_at_front(x,
                         sequence_length,
                         keys_to_trim=None,
                         **unused_kwargs):
  """Token-preprocessor to trim sequence at the beginning.

  Args:
    x: an example with dictionaries containing keys_to_trim.
    sequence_length: a dict of ints.
    keys_to_trim: a list of feature keys.

  Returns:
    A preprocessed example.
  """

  for key in (keys_to_trim or sequence_length.keys()):
    if key in x:
      # trim tokens, leaving room for EOS which gets added later
      x[key] = x[key][-(sequence_length[key] - 1):]
  return x


def trivia_qa_truncate_inputs(dataset, output_features, sequence_length):
  """Token preprocessor for the trivia QA dataset to truncate inputs.

  This function takes a dataset containing "targets" and "inputs". It searches
  for the "targets" in the "inputs" and truncates the "inputs" to
  `sequence_length` while ensuring that the "targets" are present in the
  "inputs". The function will randomly select a subset of "inputs".
  If "targets" are not found in the "inputs", then the example is
  is dropped from the dataset.

  E.g.
  Input dataset
  {
    "inputs": [0, 3, 5, 7, 9, 11, 13, 15, 17, 18]
    "targets": [5, 7, 9]
  }

  Output dataset (assuming sequence_length['inputs'] = 4)
  {
    "inputs": [3, 5, 7, 9]
    "targets": [5, 7, 9]
  }

  or

  {
     "inputs": [5, 7, 9, 11]
     "targets": [5, 7, 9]
  }
  Args:
    dataset: a tf.data.Dataset with dictionaries containing the "inputs" and
      "targets".
    output_features: unused by this function.
    sequence_length: a dict, with keys as "inputs" and "targets" indicating the
      maximum number of tokens in each of the sequences.

  Returns:
    a dataset

  """

  del output_features

  @seqio.map_over_dataset(num_seeds=1)
  def my_fn(features, seed):
    """Function to map original dataset to the new dataset."""
    inputs = features['inputs']
    targets = features['targets']
    ans_len = tf.shape(targets)[0]
    max_input_tokens = sequence_length['inputs']

    def truncate_inputs():
      """Helper function to truncate the inputs."""

      def answer_in_context(context, answer):
        """Helper function that checks if the answer is present in the context.

        Args:
          context: Tensor, tokenized representation of the context
          answer: Tensor, tokenized representation of the answer

        Returns:
          result: boolean, indicates if the answer was present in the context.
          pos_mask: boolean mask, a mask for every possible start position of
            the answer in the context. Indicates whether the answer starts at
            the particular position.
        """
        conv_inp = tf.reshape(tf.cast(context, tf.float32), [1, -1, 1])
        ans_len = tf.shape(answer)[0]
        filters = tf.eye(ans_len, dtype=tf.float32)

        # Assume context len is N and answer len is M.
        # Use a convolution to create a matrix of (N-M) x M elements where
        # each row of the matrix is a sequence of len M. This matrix contains
        # all possible contiguous sequences of length M from the context.
        # Every row of this matrix is compared with the answer to check if the
        # answer exists in the context.
        strided = tf.nn.conv1d(conv_inp,
                               tf.reshape(filters, [ans_len, 1, ans_len]), 1,
                               'VALID')
        strided = tf.cast(strided[0], answer.dtype)
        pos_mask = tf.reduce_all(
            tf.equal(strided, tf.reshape(answer, [1, -1])), 1)
        result = tf.reduce_any(pos_mask)
        return result, pos_mask

      def slice_inputs(inputs, answer_len, pos_mask, seed=None):
        """Helper function to slice inputs while keeping the answer."""
        ans_start_pos = tf.cast(tf.where(pos_mask)[0][0], tf.int32)
        inputs_len = tf.shape(inputs)[0]
        start_range_min = tf.maximum(
            0, ans_start_pos - (max_input_tokens - answer_len))
        start_range_max = tf.minimum(ans_start_pos,
                                     inputs_len - max_input_tokens) + 1

        start_pos = tf.random.stateless_uniform(
            [],
            minval=start_range_min,
            maxval=start_range_max,
            dtype=tf.int32,
            seed=seed)
        return inputs[start_pos:start_pos + max_input_tokens]

      result, pos_mask = answer_in_context(inputs, targets)

      if result:
        return slice_inputs(inputs, ans_len, pos_mask, seed=seed)
      else:
        return tf.constant([], dtype=inputs.dtype)

    if tf.greater(tf.shape(inputs)[0], max_input_tokens):
      inputs = truncate_inputs()
    return {'inputs': inputs, 'targets': features['targets']}

  dataset = my_fn(dataset)
  return dataset.filter(lambda x: tf.size(x['inputs']) > 0)


@gin.configurable()
def unsupervised(dataset,
                 preprocessors=None,
                 output_features=None,
                 sequence_length=None):
  """Configure this to point at unsupervised preprocessors.

   This function creates an extra level of indirection in case we want
   different unsupervised pretraining functions in the future which do not
   fit into the denoise() framework.

   This function should be used as a post-cache preprocessing function.

  Args:
    dataset: A tf.data.Dataset to process.
    preprocessors: a list of token-preprocessor functions. These functions
      should take unused kwargs if output_features or sequence_length is not
      used.
    output_features: dict(str, Feature), output features of the Task to be
      passed to the model.
    sequence_length: dict mapping feature key to int length for that feature.

  Returns:
    A preprocessed tf.data.Dataset.
  """
  if preprocessors is None:
    logging.warning(
        'unsupervised preprocessor got preprocessors=None; no preprocessing '
        'will be applied.'
    )
    return dataset

  kwargs = {}
  if output_features:
    kwargs['output_features'] = output_features
  if sequence_length:
    kwargs['sequence_length'] = sequence_length

  for p in preprocessors:
    dataset = p(dataset, **kwargs)
  return dataset

# ======================== split_tokens and helpers ============================


@gin.configurable
def split_tokens(dataset: tf.data.Dataset,
                 min_tokens_per_segment: Optional[int] = None,
                 max_tokens_per_segment: int = gin.REQUIRED,
                 feature_key: str = 'targets',
                 additional_feature_keys: Optional[Sequence[str]] = None,
                 passthrough_feature_keys: Optional[Sequence[str]] = None,
                 **unused_kwargs) -> tf.data.Dataset:
  """Split examples into multiple examples each.

  The intended use case is to break up long examples for use in unsupervised
  transfer-learning.

  This function is generally preceded by select_random_chunk.

  If min_tokens_per_segment is provided, the segment length is chosen randomly
  per document from a log-uniform distribution.  If min_tokens_per_segment is
  None, then the segment length is max_tokens_per_segment (except for a possibly
  shorter last segment in each document).

  Args:
    dataset: a tf.data.Dataset with dictionaries containing the key feature_key.
    min_tokens_per_segment: an optional integer
    max_tokens_per_segment: an integer, the maximum number of tokens in each
      segment. Only the final segment may be shorter.
    feature_key: a string, the feature to split
    additional_feature_keys: Additional features to split. The same chunk size
      will be used, so they should be the same size as feature_key.
    passthrough_feature_keys: Features to pass through without any splitting.

  Returns:
    a dataset
  """
  if passthrough_feature_keys:
    split_keys = set([feature_key] + (additional_feature_keys or []))
    overlap_keys = split_keys & set(passthrough_feature_keys)
    if overlap_keys:
      raise ValueError(
          f'split keys {overlap_keys} also included in passthrough keys')

  @seqio.map_over_dataset(num_seeds=1)
  def _split_tokens(x, seed):
    """Split one token sequence into multiple sequences."""
    tokens = x[feature_key]
    n_tokens = tf.shape(tokens)[0]
    if min_tokens_per_segment is None:
      length = max_tokens_per_segment
    else:
      # pick a length - log-uniformly distributed
      length = tf.cast(
          tf.exp(
              tf.random.stateless_uniform(
                  [],
                  minval=math.log(min_tokens_per_segment),
                  maxval=math.log(max_tokens_per_segment),
                  seed=seed
              )
          ),
          tf.int32)

    # Pad to a multiple of length, then use tf.reshape to split up the tokens
    # into num_segments segments each of the given length.
    num_segments = tf.cast(
        tf.math.ceil(
            tf.cast(n_tokens, tf.float32) / tf.cast(length, tf.float32))
        ,
        tf.int32)
    padding = num_segments * length - tf.shape(tokens)[0]
    feature_keys_to_split = [feature_key]
    orig_lengths = {}
    outputs = {}
    if additional_feature_keys is not None:
      feature_keys_to_split.extend(additional_feature_keys)
    for k in feature_keys_to_split:
      with tf.control_dependencies([
          tf.assert_equal(
              tf.shape(tokens)[0],
              tf.shape(x[k])[0],
              message=(f'Additional feature {k} is not the same size as '
                       f'{feature_key} along axis 0 in split_tokens().')
          )
      ]):
        shape = tf.shape(x[k])[1:]
        shape_list = x[k].shape[1:]
        padded = tf.pad(
            x[k],
            tf.concat([[[0, padding]],
                       tf.zeros([len(shape_list), 2], dtype=tf.int32)],
                      axis=0))
        orig_lengths[k] = tf.concat(
            [tf.repeat(length, num_segments - 1), [length - padding]], axis=0)
        outputs[k] = tf.reshape(
            padded, tf.concat([[-1, length], shape], axis=0))
    return outputs, orig_lengths

  def _strip_padding(inputs, orig_lengths):
    output = {}
    for k, v in inputs.items():
      output[k] = v[:orig_lengths[k]]
    return output

  # Filter empty examples.
  dataset = dataset.filter(lambda x: tf.not_equal(tf.size(x[feature_key]), 0))

  if passthrough_feature_keys:
    # Extract passthrough keys into a separate dataset.
    def _extract_passthrough_fields(inputs):
      return {
          k: v for k, v in inputs.items() if k in passthrough_feature_keys
      }
    passthrough_ds = dataset.map(
        _extract_passthrough_fields, num_parallel_calls=AUTOTUNE)

  dataset = _split_tokens(dataset)

  if passthrough_feature_keys:
    # Get number of segments from each example in original dataset.
    def _extract_num_segments(inputs, orig_lengths):
      del orig_lengths
      return tf.shape(inputs[feature_key], out_type=tf.int64)[0]
    num_segments_ds = dataset.map(
        _extract_num_segments, num_parallel_calls=AUTOTUNE)

    # Construct a dataset where the passthrough fields are repeated once for
    # each segment.
    def _repeat_passthrough_fields(inputs, num_segments):
      return tf.data.Dataset.from_tensors(inputs).repeat(num_segments)
    passthrough_ds = tf.data.Dataset.zip(
        (passthrough_ds, num_segments_ds)).flat_map(_repeat_passthrough_fields)

  dataset = dataset.unbatch()
  dataset = dataset.map(_strip_padding, num_parallel_calls=AUTOTUNE)

  if passthrough_feature_keys:
    # Add the passthrough fields back to the original dataset.
    def _merge_passthrough_fields(inputs, passthrough_inputs):
      outputs = {}
      outputs.update(inputs)
      outputs.update(passthrough_inputs)
      return outputs
    dataset = tf.data.Dataset.zip((dataset, passthrough_ds)).map(
        _merge_passthrough_fields, num_parallel_calls=AUTOTUNE)

  return dataset


@gin.configurable
def split_tokens_to_inputs_length(dataset, sequence_length,
                                  output_features, **kwargs):
  max_tokens = sequence_length['inputs']
  if output_features['inputs'].add_eos:
    # Leave room to insert an EOS token.
    max_tokens -= 1

  return split_tokens(dataset, max_tokens_per_segment=max_tokens, **kwargs)


@gin.configurable
def split_tokens_to_targets_length(dataset, sequence_length,
                                   output_features, **kwargs):
  max_tokens = sequence_length['targets']
  if output_features['targets'].add_eos:
    # Leave room to insert an EOS token.
    max_tokens -= 1

  return split_tokens(dataset, max_tokens_per_segment=max_tokens, **kwargs)


@gin.configurable
def split_tokens_to_random_length(dataset, sequence_length,
                                  output_features, **kwargs):
  max_tokens = sequence_length['inputs']
  if output_features['inputs'].add_eos:
    # Leave room to insert an EOS token.
    max_tokens -= 1

  return split_tokens(dataset,
                      min_tokens_per_segment=8,
                      max_tokens_per_segment=max_tokens,
                      **kwargs)


@gin.configurable
def concatenate_and_split_to_fixed_length(dataset,
                                          sequence_length,
                                          output_features,
                                          feature_key='targets',
                                          **unused_kwargs):
  """Concatenate tokens across examples, then split to fixed-size chunks.

  Chunk length is determined by sequence_length[feature_key].

  Args:
    dataset: a tf.data.Dataset
    sequence_length: a dict of ints.
    output_features: a dict mapping feature name to t5.data.Feature.
    feature_key: a string
  Returns:
    a tf.data.Dataset
  """
  dataset = dataset.map(lambda x: {feature_key: x[feature_key]})
  max_tokens = sequence_length[feature_key]
  if output_features[feature_key].add_eos:
    # Leave room to insert an EOS token.
    max_tokens -= 1
  return dataset.unbatch().batch(max_tokens)


@gin.configurable
def filter_by_string_length(dataset,
                            feature_key='targets',
                            min_length=1,
                            max_length=1000000,
                            **unused_kwargs):
  """Filter examples by string length.

  Args:
    dataset: a tf.data.Dataset (not tokenized)
    feature_key: a string
    min_length: an integer
    max_length: an integer
  Returns:
    a tf.data.Dataset
  """
  def my_fn(x):
    l = tf.strings.length(x[feature_key])
    return tf.logical_and(tf.greater_equal(l, min_length),
                          tf.less_equal(l, max_length))
  return dataset.filter(my_fn)


@gin.configurable
def random_spans_helper(inputs_length=gin.REQUIRED,
                        noise_density=gin.REQUIRED,
                        mean_noise_span_length=gin.REQUIRED,
                        extra_tokens_per_span_inputs=gin.REQUIRED,
                        extra_tokens_per_span_targets=gin.REQUIRED,
                        verbose=False):
  """Training parameters to avoid padding with random_spans_noise_mask.

  When training a model with random_spans_noise_mask, we would like to set the
  other training hyperparmeters in a way that avoids padding.  This function
  helps us compute these hyperparameters.

  We assume that each noise span in the input is replaced by
  extra_tokens_per_span_inputs sentinel tokens, and each non-noise span in the
  targets is replaced by extra_tokens_per_span_targets sentinel tokens.

  This function tells us the required number of tokens in the raw example (for
  split_tokens()) as well as the length of the encoded targets.

  Note that this function assumes the inputs and targets will have EOS appended
  and includes that in the reported length.

  Args:
    inputs_length: an integer - desired length of the tokenized inputs sequence
    noise_density: a float
    mean_noise_span_length: a float
    extra_tokens_per_span_inputs: an integer
    extra_tokens_per_span_targets: an integer
    verbose: a bool indicating whether to log sequence lengths
  Returns:
    tokens_length: length of original text in tokens
    targets_length: an integer - length in tokens of encoded targets sequence
  """
  def _tokens_length_to_inputs_length_targets_length(tokens_length):
    num_noise_tokens = int(round(tokens_length * noise_density))
    num_nonnoise_tokens = tokens_length - num_noise_tokens
    num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
    # inputs contain all nonnoise tokens, sentinels for all noise spans
    # and one EOS token.
    return (
        num_nonnoise_tokens +
        num_noise_spans * extra_tokens_per_span_inputs + 1,
        num_noise_tokens +
        num_noise_spans * extra_tokens_per_span_targets + 1)

  tokens_length = inputs_length - 1
  while (_tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0]
         <= inputs_length):
    tokens_length += 1
  inputs_length, targets_length = (
      _tokens_length_to_inputs_length_targets_length(tokens_length))
  # minor hack to get the targets length to be equal to inputs length
  # which is more likely to have been set to a nice round number.
  if noise_density == 0.5 and targets_length > inputs_length:
    tokens_length -= 1
    targets_length -= 1
  if verbose:
    logging.info(
        'tokens_length=%s inputs_length=%s targets_length=%s '
        'noise_density=%s mean_noise_span_length=%s ',
        tokens_length, inputs_length, targets_length,
        noise_density, mean_noise_span_length)
  return tokens_length, targets_length


@gin.configurable
def random_spans_tokens_length():
  """Helper for gin-configuring split_tokens with random_spans_noise_mask."""
  return random_spans_helper()[0]


@gin.configurable
def random_spans_targets_length():
  """Helper for gin-configuring the targets sequence length."""
  return random_spans_helper()[1]


# ========================== denoise and helpers ===============================


class DenoiseNoiseMaskFn(Protocol):

  def __call__(self, num_tokens: tf.Tensor, noise_density: float,
               seeds: tf.Tensor) -> tf.Tensor:
    """Computes the boolean makes. Seeds should have shape [2, 2]."""


class DenoiseInputsFn(Protocol):

  def __call__(self, tokens: tf.Tensor, noise_mask: tf.Tensor, vocabulary,
               seeds: tf.Tensor) -> tf.Tensor:
    """Computes the input tokens. Seeds should have shape [2, 2]."""


class DenoiseTargetsFn(Protocol):

  def __call__(self, tokens: tf.Tensor, noise_mask: tf.Tensor, vocabulary,
               seeds: tf.Tensor) -> tf.Tensor:
    """Computes the target tokens. Seeds should have shape [2, 2]."""


def single_example_denoise(features: FeatureType,
                           seed: tf.Tensor,
                           *,
                           output_features: Mapping[str, Any],
                           noise_density: float,
                           noise_mask_fn: DenoiseNoiseMaskFn,
                           inputs_fn: DenoiseInputsFn,
                           targets_fn: Optional[DenoiseTargetsFn] = None,
                           passthrough_feature_keys: Optional[
                               Sequence[str]] = None,
                           input_feature_key: str = 'inputs') -> FeatureType:
  """Preprocessing function for self-supervised denoising tasks.

  This function takes a dataset containing "targets" sequences,
  and turns each sequence into a dictionary containing:
  {
     "inputs": noisy version of the original sequence
     "targets": the full original sequence or missing parts of original sequence
  }

  In particular, for each sequence, we choose a boolean noise_mask identifying
  which tokens in the sequence to corrupt, as defined by the given
  noise_mask_fn.

  Given the sequence and the noise mask, we generate the inputs and targets
  using the given inputs_fn and targets_fn respectively.

  The self-supervised tasks vary along these axes:
    - noise_density: What fraction of the tokens to select as noise
    - noise_mask_fn: What pattern should the noise mask follow
         (iid, regular segments, etc.)
    - inputs_fn: How to apply the noise
         (drop noise tokens, replace with sentinels, etc.)
    - targets_fn: How to represent the output
         (full sequence, only non-noise tokens, etc.)

  Note: Some functionality has been deleted, which we may or may not want to
  restore at a later date.  The code for this functionality can be found in
  the deleted code for this CL.  In particular:
    - mixture of masking and random replacement
    - task labels prepended to the inputs

  Args:
    features: Flat dictionary of features.
    seed: Random seed to use.
    output_features: a dict mapping feature name to t5.data.Feature.
    noise_density: a float
    noise_mask_fn: a function from (length, noise_density) -> boolean mask
    inputs_fn: a function from (tokens, noise_mask, vocabulary) -> tokens
    targets_fn: a function from (tokens, noise_mask, vocabulary) -> tokens
    passthrough_feature_keys: names of additional features to include in output
    input_feature_key: name of feature to use as inputs

  Returns:
    A preprocessed features.
  """
  if passthrough_feature_keys and (input_feature_key in passthrough_feature_keys
                                   or 'targets' in passthrough_feature_keys):
    raise ValueError(
        f"passthrough keys cannot contain '{input_feature_key}' or 'targets'")

  seeds = tf.unstack(tf.random.experimental.stateless_split(seed, 6))
  tokens = features['targets']
  vocabulary = output_features['targets'].vocabulary
  if (input_feature_key in output_features and
      vocabulary != output_features[input_feature_key].vocabulary):
    raise ValueError(
        'denoise creates inputs based on tokenized targets but was applied '
        'to a task that uses different vocabularies for inputs and targets.')
  noise_mask = noise_mask_fn(tf.size(tokens), noise_density, seeds=seeds[:2])
  inputs = inputs_fn(tokens, noise_mask, vocabulary, seeds=seeds[2:4])
  if targets_fn:
    targets = targets_fn(tokens, noise_mask, vocabulary, seeds=seeds[4:6])
  else:
    targets = tokens
  return {
      input_feature_key: inputs,
      'targets': targets,
      **{
          k: features[k]
          for k in features
          if passthrough_feature_keys and k in passthrough_feature_keys
      }
  }


@gin.configurable()
def denoise(dataset,
            output_features,
            noise_density=gin.REQUIRED,
            noise_mask_fn=gin.REQUIRED,
            inputs_fn=gin.REQUIRED,
            targets_fn=None,
            passthrough_feature_keys: Optional[Sequence[str]] = None,
            input_feature_key='inputs',
            **unused_kwargs):
  """SeqIO wrapper for single_example_denoise()."""

  @seqio.map_over_dataset(num_seeds=1)
  def my_fn(features, seed):
    return single_example_denoise(
        features,
        seed,
        output_features=output_features,
        noise_density=noise_density,
        noise_mask_fn=noise_mask_fn,
        inputs_fn=inputs_fn,
        targets_fn=targets_fn,
        passthrough_feature_keys=passthrough_feature_keys,
        input_feature_key=input_feature_key)

  return my_fn(dataset)


@gin.configurable()
def iid_noise_mask(length, noise_density, seeds):
  """Independent and identically distributed token noise.

  Args:
    length: an int32 scalar.
    noise_density: a float - approximate density of output mask.
    seeds: an int32 Tensor, shaped (1, 2), the random seed.

  Returns:
    a boolean tensor with shape [length].
  """
  return tf.random.stateless_uniform([length], seed=seeds[0]) < noise_density


@gin.configurable()
def regular_noise_mask(length,
                       noise_density,
                       seeds,
                       min_span_length=1,
                       max_span_length=5):
  """Noise mask consisting of equally spaced spans of equal length.

  The span length and the offset are chosen randomly per-example.
  The beginning and end of the sequence may be part of shorter spans of noise.
  For example, if noise_density=0.25 and a span length of 2 is chosen,
  then the output might be:

  [T F F F F F F T T F F F F F F T T F F F F F F T T F F]

  Args:
    length: an int32 scalar.
    noise_density: a float - approximate density of output mask.
    seeds: an int32 Tensor, shaped (2, 2), the random seeds.
    min_span_length: an integer.
    max_span_length: an integer.

  Returns:
    a boolean tensor with shape [length].
  """
  span_length = tf.random.stateless_uniform(
      [],
      minval=min_span_length,
      maxval=max_span_length + 1,
      dtype=tf.int32,
      seed=seeds[0])
  period = tf.cast(
      tf.round(tf.cast(span_length, tf.float32) / noise_density), tf.int32)
  offset = tf.random.stateless_uniform(
      [],
      maxval=period,
      dtype=tf.int32,
      seed=seeds[1])
  return (tf.range(length, dtype=tf.int32) + offset) % period < span_length


@gin.configurable()
def random_spans_noise_mask(length,
                            noise_density,
                            seeds,
                            mean_noise_span_length=3.0,
                            random_roll=False):
  """Noise mask consisting of random spans of noise tokens.

  The number of noise tokens and the number of noise spans and non-noise spans
  are determined deterministically as follows:

    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(
       num_noise_tokens / mean_noise_span_length)

  Spans alternate between non-noise and noise, beginning with non-noise.
  Subject to the above restrictions, all masks are equally likely.

  Args:
    length: an int32 scalar (length of the incoming token sequence)
    noise_density: a float - approximate density of output mask
    seeds: an int32 Tensor, shaped (2, 2)
    mean_noise_span_length: a number
    random_roll: bool, whether to roll the mask by a random integer offset in
      [0, length). Set random_roll to True to get a more uniform distribution
      of masked positions. Specifically, when random_roll is False (default) and
      a single span is enough to satisfy the noise density requirement, this
      fuction masks only the last few positions.

  Returns:
    a boolean tensor with shape [length]
  """

  if noise_density == 0.0:
    return tf.zeros(length, tf.bool)

  orig_length = length
  # increase length to avoid degeneracy
  length = tf.maximum(length, 2)
  def to_int(x):
    return tf.cast(x, tf.int32)
  def to_float(x):
    return tf.cast(x, tf.float32)
  num_noise_tokens = to_int(tf.round(to_float(length) * noise_density))
  # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
  num_noise_tokens = tf.minimum(tf.maximum(num_noise_tokens, 1), length - 1)
  num_noise_spans = to_int(
      tf.round(to_float(num_noise_tokens) / mean_noise_span_length))
  # avoid degeneracy by ensuring positive number of noise spans
  num_noise_spans = tf.maximum(num_noise_spans, 1)
  num_nonnoise_tokens = length - num_noise_tokens
  # pick the lengths of the noise spans and the non-noise spans
  def _random_segmentation(num_items, num_segments, seed):
    """Partition a sequence of items randomly into non-empty segments.

    Args:
      num_items: an integer scalar > 0
      num_segments: an integer scalar in [1, num_items]
      seed: an integer seed
    Returns:
      a Tensor with shape [num_segments] containing positive integers that add
      up to num_items
    """
    first_in_segment = tf.pad(
        seqio.stateless_shuffle(
            to_int(tf.range(num_items - 1) < num_segments - 1),
            seed),
        [[1, 0]])
    segment_id = tf.cumsum(first_in_segment)
    segment_length = tf.math.segment_sum(tf.ones_like(segment_id), segment_id)
    return segment_length
  noise_span_lengths = _random_segmentation(
      num_noise_tokens, num_noise_spans, seeds[0])
  nonnoise_span_lengths = _random_segmentation(
      num_nonnoise_tokens, num_noise_spans, seeds[1])
  interleaved_span_lengths = tf.reshape(
      tf.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
      [num_noise_spans * 2])
  span_starts = tf.cumsum(interleaved_span_lengths)[:-1]
  span_start_indicator = tf.math.unsorted_segment_sum(
      tf.ones_like(span_starts), span_starts, length)
  span_num = tf.cumsum(span_start_indicator)
  is_noise = tf.equal(span_num % 2, 1)

  mask = is_noise[:orig_length]

  if random_roll:
    roll_seed = (seeds[0][0]+seeds[1][1], seeds[0][1]-seeds[1][0])  # new seed.
    # Roll the mask by a random offset e.g. for offset=2: [1,2,3,4] => [3,4,1,2]
    offset = tf.random.stateless_uniform(
        [1], seed=roll_seed, dtype=tf.int32, minval=0, maxval=length)[0]
    mask = tf.roll(mask, shift=offset, axis=0)

  return mask


@gin.configurable()
def random_prefix_noise_mask(length, noise_density, seeds):
  """First part of the sequence is noise (for prefix_lm).

  The length of the prefix is chosen uniformly between [1, length)
  noise_density must be 0.5.
  TODO(noam): figure out some distribution to use if noise_density != 0.5.

  Args:
    length: an int32 scalar.
    noise_density: a float - must equal 0.5.
    seeds: an int32 Tensor, shaped (1, 2), the random seed.

  Returns:
    a boolean tensor with shape [length].
  """
  if noise_density != 0.5:
    raise NotImplementedError(
        'noise density must equal 0.5 for random_prefix_noise_mask')
  max_input_tokens = length - 1
  min_input_tokens = tf.minimum(max_input_tokens, 1)
  num_input_tokens = tf.random.stateless_uniform(
      [],
      minval=min_input_tokens,
      maxval=max_input_tokens + 1,
      dtype=tf.int32,
      seed=seeds[0])
  return tf.range(length, dtype=tf.int32) < num_input_tokens


@gin.configurable()
def sentinel_id(vocabulary, return_value=None):
  """Token ID to use as a sentinel.

  By default, we use the last token in the vocabulary.

  Args:
    vocabulary: a t5.data.vocabularies.Vocabulary
    return_value: an optional integer
  Returns:
    an integer
  """
  if return_value is not None:
    return return_value
  return vocabulary.vocab_size - 1


@gin.configurable()
def noise_token_to_sentinel(tokens, noise_mask, vocabulary, seeds):
  """Replace each noise token with the given sentinel.

  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: a vocabulary.Vocabulary
    seeds: an unused int32 Tensor

  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  del seeds
  return tf.where(noise_mask,
                  tf.cast(sentinel_id(vocabulary), tokens.dtype),
                  tokens)


@gin.configurable()
def noise_span_to_sentinel(tokens, noise_mask, vocabulary, seeds):
  """Replace each run of consecutive noise tokens with a single sentinel.

  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: a vocabulary.Vocabulary
    seeds: an unused int32 Tensor
  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  del seeds
  tokens = tf.where(noise_mask,
                    tf.cast(sentinel_id(vocabulary), tokens.dtype),
                    tokens)
  prev_token_is_noise = tf.pad(noise_mask[:-1], [[1, 0]])
  subsequent_noise_tokens = tf.logical_and(noise_mask, prev_token_is_noise)
  return tf.boolean_mask(tokens, tf.logical_not(subsequent_noise_tokens))


@gin.configurable()
def nonnoise_span_to_sentinel(tokens, noise_mask, vocabulary, seeds):
  return noise_span_to_sentinel(
      tokens, tf.logical_not(noise_mask), vocabulary, seeds)


@gin.configurable()
def noise_span_to_unique_sentinel(tokens, noise_mask, vocabulary, seeds):
  """Replace each run of consecutive noise tokens with a different sentinel.

  The idea here is to be able to align the dropped spans in the inputs
  with the markers in the targets.

  We want to generate training examples like
  "We hold X to be Y that" -> "X these truths Y self evident Z"

  Sentinels assigned in decreasing order within the sequence starting at
  vocabulary.size - 1.  That is, we appropriate the last tokens in the
  vocabulary for additional use as sentinels.

  TODO(noam): we may want to try enlarging the vocabulary and leaving room
  for the sentinels instead.  However, this requires enlarging the embedding
  tables in the model, so that is a bigger change.

  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: a vocabulary.Vocabulary
    seeds: an unused int32 Tensor
  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  del seeds

  prev_token_is_noise = tf.pad(noise_mask[:-1], [[1, 0]])

  first_noise_tokens = tf.logical_and(
      noise_mask, tf.logical_not(prev_token_is_noise))
  subsequent_noise_tokens = tf.logical_and(noise_mask, prev_token_is_noise)

  sentinel = sentinel_id(vocabulary) + 1 - tf.cumsum(
      tf.cast(first_noise_tokens, tokens.dtype))

  tokens = tf.where(first_noise_tokens, sentinel, tokens)
  return tf.boolean_mask(tokens, tf.logical_not(subsequent_noise_tokens))


@gin.configurable()
def nonnoise_span_to_unique_sentinel(tokens, noise_mask, vocabulary, seeds):
  return noise_span_to_unique_sentinel(
      tokens, tf.logical_not(noise_mask), vocabulary, seeds)


@gin.configurable()
def drop_noise_tokens(tokens, noise_mask, vocabulary, seeds):
  """Drop noise tokens without inserting a sentinel.

  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: an unused vocabulary.Vocabulary
    seeds: an unused int32 Tensor

  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  del vocabulary
  del seeds
  return tf.boolean_mask(tokens, tf.logical_not(noise_mask))


@gin.configurable()
def drop_nonnoise_tokens(tokens, noise_mask, vocabulary, seeds):
  """Drop non-noise tokens without inserting a sentinel.

  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: an unused vocabulary.Vocabulary
    seeds: an unused int32 Tensor
  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  del vocabulary
  del seeds
  return tf.boolean_mask(tokens, noise_mask)


@gin.configurable()
def permute_noise_tokens(tokens, noise_mask, vocabulary, seeds):
  """Permute the noise tokens, keeping the non-noise tokens where they are.

  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: an unused vocabulary.Vocabulary
    seeds: an int32 Tensor, sized (1, 2)
  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  del vocabulary

  masked_only = tf.boolean_mask(tokens, noise_mask)
  permuted = seqio.stateless_shuffle(masked_only, seeds[0])
  # pad to avoid errors when it has size 0
  permuted = tf.pad(permuted, [[0, 1]])
  indices = tf.cumsum(tf.cast(noise_mask, tf.int32), exclusive=True)
  return tf.where(noise_mask,
                  tf.gather(permuted, indices),
                  tokens)


@gin.configurable()
def noise_token_to_gathered_token(tokens, noise_mask, vocabulary, seeds):
  """Replace each noise token with a random token from the sequence.

  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: an unused vocabulary.Vocabulary
    seeds: an int32 Tensor, sized (1, 2)
  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  del vocabulary

  indices = tf.random.stateless_uniform(
      shape=tf.shape(tokens),
      maxval=tf.size(tokens),
      dtype=tf.int32,
      seed=seeds[0])
  return tf.where(noise_mask,
                  tf.gather(tokens, indices),
                  tokens)


@gin.configurable()
def noise_token_to_random_token(
    tokens,
    noise_mask,
    vocabulary,
    seeds,
    num_reserved_tokens=3):
  """Replace each noise token with a random token from the vocabulary.



  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: a vocabulary.Vocabulary
    seeds: an int32 Tensor, shaped (1, 2)
    num_reserved_tokens: an integer
  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  return tf.where(noise_mask,
                  tf.random.stateless_uniform(
                      tf.shape(tokens),
                      minval=num_reserved_tokens,
                      maxval=vocabulary.vocab_size,
                      dtype=tokens.dtype,
                      seed=seeds[0]),
                  tokens)


@gin.configurable()
def noise_token_to_random_token_or_sentinel(
    tokens,
    noise_mask,
    vocabulary,
    seeds,
    random_prob=0.1):
  """Replace each noise token with a random token or a sentinel.

  For each masked token, with probability random_prob, we replace it by a
  random token from the vocabulary.  Otherwise, we replace it with a sentinel.

  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: a vocabulary.Vocabulary
    seeds: an int32 Tensor, shaped (2, 2).
    random_prob: a float
  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  use_random = (
      tf.random.stateless_uniform(tf.shape(tokens), seed=seeds[0]) <
      random_prob)
  return tf.where(
      use_random,
      noise_token_to_random_token(
          tokens, noise_mask, vocabulary, seeds=seeds[1:]),
      noise_token_to_sentinel(
          tokens, noise_mask, vocabulary, seeds=()))


# =============== EXPERIMENTAL preprocessors (not used for the T5 paper) =======


def trim_and_pad_dataset(dataset, sequence_length):
  """A wrapper to use `seqio.utils.trim_and_pad_dataset` as a preprocessor."""
  return seqio.utils.trim_and_pad_dataset(
      dataset, feature_lengths=sequence_length)


def targets_for_prefix_lm_objective(dataset, sequence_length, output_features):
  """Prepares targets to be used for prefix LM objective."""
  dataset = select_random_chunk(
      dataset, output_features, max_length=65536, feature_key='targets')
  dataset = seqio.preprocessors.append_eos(dataset, output_features)
  dataset = reduce_concat_tokens(dataset, batch_size=128)
  dataset = split_tokens(
      dataset, max_tokens_per_segment=sequence_length['targets'])
  dataset = trim_and_pad_dataset(dataset, sequence_length)
  return dataset


def pack_prefix_lm_encoder_decoder(ds, sequence_length, pad_id=0):
  """Pack two examples into one with the prefix LM objective."""
  packed_length = next(iter(sequence_length.values()))
  assert packed_length % 2 == 0
  assert all(l == packed_length for l in sequence_length.values())

  @seqio.utils.map_over_dataset(num_seeds=1)
  def pack_examples(example_pair, seed):
    split_point = tf.random.stateless_uniform((),
                                              minval=1,
                                              maxval=packed_length,
                                              seed=seed,
                                              dtype=tf.int32)
    inputs = tf.concat([
        example_pair['targets'][0][:split_point],
        example_pair['targets'][1][:packed_length - split_point]
    ],
                       axis=0)
    inputs = tf.reshape(inputs, (packed_length,))
    targets = tf.concat([
        example_pair['targets'][0][split_point:],
        example_pair['targets'][1][packed_length - split_point:]
    ],
                        axis=0)
    targets = tf.reshape(targets, (packed_length,))

    encoder_segment_ids = tf.cast(
        tf.range(packed_length) >= split_point, tf.int32) + 1
    decoder_segment_ids = tf.cast(
        tf.range(packed_length) >= (packed_length - split_point), tf.int32) + 1

    decoder_input_tokens = seqio.utils.make_autoregressive_inputs(
        targets, sequence_id=decoder_segment_ids)

    encoder_positions = tf.concat(
        [tf.range(split_point),
         tf.range(packed_length - split_point)], axis=0)
    encoder_positions = tf.reshape(encoder_positions, (packed_length,))
    decoder_positions = tf.concat(
        [tf.range(packed_length - split_point),
         tf.range(split_point)], axis=0)
    decoder_positions = tf.reshape(decoder_positions, (packed_length,))
    decoder_loss_weights = tf.cast(
        tf.not_equal(targets, pad_id), dtype=tf.int32)
    return {
        'encoder_input_tokens': inputs,
        'decoder_target_tokens': targets,
        'decoder_input_tokens': decoder_input_tokens,
        'encoder_segment_ids': encoder_segment_ids,
        'encoder_positions': encoder_positions,
        'decoder_segment_ids': decoder_segment_ids,
        'decoder_positions': decoder_positions,
        'decoder_loss_weights': decoder_loss_weights,
    }

  # Note that the batch requires the lengths to be the same.
  return pack_examples(ds.batch(2))


def pack_prefix_lm_decoder_only(ds,
                                sequence_length,
                                loss_on_targets_only=True,
                                pad_id=0):
  """Randomly split the tokens for the prefix LM objective."""
  packed_length = next(iter(sequence_length.values()))
  assert packed_length % 2 == 0
  assert all(l == packed_length for l in sequence_length.values())

  @seqio.utils.map_over_dataset(num_seeds=1)
  def pack_examples(example, seed):
    split_point = tf.random.stateless_uniform((),
                                              minval=1,
                                              maxval=packed_length,
                                              seed=seed,
                                              dtype=tf.int32)
    decoder_target_tokens = example['targets']
    decoder_input_tokens = seqio.utils.make_autoregressive_inputs(
        decoder_target_tokens)

    if loss_on_targets_only:
      decoder_loss_weights = tf.cast(
          tf.range(packed_length) >= split_point, tf.int32)
    else:
      decoder_loss_weights = tf.ones((packed_length,), dtype=tf.int32)

    padding_mask = tf.cast(
        tf.not_equal(decoder_target_tokens, pad_id), dtype=tf.int32)
    decoder_loss_weights *= padding_mask

    decoder_causal_attention = tf.cast(
        tf.range(packed_length) <= split_point, tf.int32)

    return {
        'decoder_target_tokens': decoder_target_tokens,
        'decoder_input_tokens': decoder_input_tokens,
        'decoder_loss_weights': decoder_loss_weights,
        'decoder_causal_attention': decoder_causal_attention,
    }

  return pack_examples(ds)