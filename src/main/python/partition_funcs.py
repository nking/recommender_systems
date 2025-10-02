

from typing import Union, Dict, Any, List
import bisect
import hashlib
import pickle

import tensorflow as tf

from tfx.proto import example_gen_pb2

#from #from ht_GeneratePartitionKeytps://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/components/example_gen/base_example_gen_executor.py
def _GeneratePartitionKey(record: Union[tf.train.Example,\
  tf.train.SequenceExample, bytes, Dict[str, Any]], \
  split_config: example_gen_pb2.SplitConfig) -> bytes:
  """Generates key for partition."""

  if not split_config.HasField('partition_feature_name'):
    if isinstance(record, bytes):
      return record
    if isinstance(record, dict):
      return pickle.dumps(record)
    return record.SerializeToString(deterministic=True)

  if isinstance(record, tf.train.Example):
    features = record.features.feature  # pytype: disable=attribute-error
  elif isinstance(record, tf.train.SequenceExample):
    features = record.context.feature  # pytype: disable=attribute-error
  else:
    raise RuntimeError(
      'Split by `partition_feature_name` is only supported '
      'for FORMAT_TF_EXAMPLE and FORMAT_TF_SEQUENCE_EXAMPLE '
      'payload format.')

  # Use a feature for partitioning the examples.
  feature_name = split_config.partition_feature_name
  if feature_name not in features:
    raise RuntimeError(
      'Feature name `{}` does not exist.'.format(feature_name))
  feature = features[feature_name]
  if not feature.HasField('kind'):
    raise RuntimeError('Partition feature does not contain any value.')
  if (not feature.HasField('bytes_list') and
    not feature.HasField('int64_list')):
    raise RuntimeError(
      'Only `bytes_list` and `int64_list` features are '
      'supported for partition.')
  return feature.SerializeToString(deterministic=True)

#from https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/components/example_gen/base_example_gen_executor.py#L72
def partitionFn(\
    record: Union[tf.train.Example, tf.train.SequenceExample, bytes, \
    Dict[str,Any]], \
    num_partitions: int, \
    buckets: List[int], \
    split_config: example_gen_pb2.SplitConfig,\
) -> int:
  """Partition function for the ExampleGen's output splits."""
  assert num_partitions == len(buckets), 'Partitions do not match bucket number.'
  partition_str = _GeneratePartitionKey(record, split_config)
  bucket = int(hashlib.sha256(partition_str).hexdigest(), 16) % buckets[-1]
  # For example, if buckets is [10,50,80], there will be 3 splits:
  #   bucket >=0 && < 10, returns 0
  #   bucket >=10 && < 50, returns 1
  #   bucket >=50 && < 80, returns 2
  return bisect.bisect(buckets, bucket)
