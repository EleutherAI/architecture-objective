import functools

from absl.testing import absltest
import gin
import seqio
from seqio import test_utils
from t5.data import preprocessors as prep
import tensorflow.compat.v2 as tf

tf.compat.v1.enable_eager_execution()

mock = absltest.mock
assert_dataset = test_utils.assert_dataset


class PreprocessorsTest(tf.test.TestCase):
  def test_span_corruption(self):
    vocab = test_utils.sentencepiece_vocab()
    inp = list(range(1, 100))
    og_dataset = tf.data.Dataset.from_tensor_slices({'targets': [inp]})
    og_dataset = og_dataset.repeat(100)
    output_features = {
        'targets': seqio.Feature(vocab),
        'inputs': seqio.Feature(vocab),
    }
    output_dataset = prep.span_corruption(
        og_dataset,
        sequence_length={'targets': 100, 'inputs': 100},
        output_features=output_features,
        merge_examples_to_reduce_padding=True)
    output_keys = list(output_dataset.as_numpy_iterator())[0].keys()
    self.assertSequenceEqual(['inputs', 'targets'], list(output_keys))

  def test_lm(self):
    dataset = tf.data.Dataset.from_tensor_slices({'text': ['That is good.']})
    dataset = prep.lm(dataset)
    assert_dataset(dataset, {'inputs': '', 'targets': 'That is good.'})


if __name__ == '__main__':
    absltest.main()