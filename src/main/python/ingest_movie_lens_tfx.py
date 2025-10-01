import apache_beam as beam
from ingest_movie_lens_beam import ingest_and_join

import os
import absl
import json
import pprint

from typing import Any, Dict, List, Text

import numpy as np
import tensorflow as tf
import apache_beam as beam

from absl import logging

from tfx.components.example_gen.base_example_gen_executor import BaseExampleGenExecutor
#from tfx.components.example_gen import utils
from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.base import base_component

#from tfx.types import artifact_utils
#from tfx.types import standard_artifacts
from tfx.types.standard_artifacts import Examples

#from tfx.dsl.component.experimental.annotations import InputArtifact
#from tfx.dsl.component.experimental.annotations import OutputArtifact
#from tfx.dsl.component.experimental.annotations import Parameter
#from tfx.dsl.component.experimental.decorators import component
#from tfx.types.experimental.simple_artifacts import Dataset

from tfx import v1 as tfx

'''
building a custom component
'''

# Set up logging.
tf.get_logger().propagate = False
absl.logging.set_verbosity(absl.logging.DEBUG)
pp = pprint.PrettyPrinter()

print(f"TensorFlow version: {tf.__version__}")
print(f"TFX version: {tfx.__version__}")

import json
#from tfx.utils import json_utils

from tfx.proto import example_gen_pb2

#tf.train.Example is for independent, fixed-size examples
#tf.train.SequenceExample is for variable-length sequential data,
#    such as sentences, time series, or videos.

class IngestMovieLensExecutorSpec(types.ComponentSpec):
  """ComponentSpec for Custom TFX MovieLensExecutor Component."""
  PARAMETERS = {
    # These are parameters that will be passed in the call to
    # create an instance of this component.
    'name': ExecutionParameter(type=Text),
    'ratings_uri' : ExecutionParameter(type=Text),
    'movies_uri' : ExecutionParameter(type=Text),
    'users_uri' : ExecutionParameter(type=Text),
    'headers_present' : ExecutionParameter(type=bool),
    'delim' : ExecutionParameter(type=Text),
    'ratings_key_col_dict' : ExecutionParameter(type=Dict[str,int]),
    'users_key_col_dict' : ExecutionParameter(type=Dict[str,int]),
    'movies_key_col_dict' : ExecutionParameter(type=Dict[str,int]),
    'bucket_names': ExecutionParameter(type=List[str]),
    'buckets' : ExecutionParameter(type=List[int])
  }
  INPUTS = {
    # these are tracked by MLMD.  they are usually the output artifacts
    #   of an upstream component
    # INPUTS will be a dictionary with input artifacts, including URIs

    #'ratings_uri' : ChannelParameter(type=standard_artifacts.String), \
    #'movies_uri' : ChannelParameter(type=standard_artifacts.String), \
    #'users_uri' : ChannelParameter(type=standard_artifacts.String), \
    #'headers_present' : ChannelParameter(type=standard_artifacts.JsonValue), \
    #'delim' : ChannelParameter(type=standard_artifacts.String), \
    #'ratings_key_dict' : ChannelParameter(type=standard_artifacts.JsonValue), \
    #'users_key_dict' : ChannelParameter(type=standard_artifacts.JsonValue), \
    #'movies_key_dict' : ChannelParameter(type=standard_artifacts.JsonValue), \
    #'bucket_names' : ChannelParameter(type=standard_artifacts.JsonValue)
    #'buckets' : ChannelParameter(type=standard_artifacts.JsonValue) \
  }
  OUTPUTS = {
    # these are tracked by MLMD
    # OUTPUTS will be a dictionary which this component will populate
    'output_examples': ChannelParameter(type=standard_artifacts.Examples),
  }

# for a TFX pipeline, we want the ingestion to be performed by
# an ExampleGen component that accepts input data and formats it as tf.Examples
class IngestMovieLensExecutor(BaseExampleGenExecutor):
  def GenerateExamplesByBeam(self, \
    pipeline: beam.Pipeline, \
    exec_properties: Dict[str, Any]\
  ) -> Dict[str, beam.pvalue.PCollection]:
    """
    :param pipeline:
    :param exec_properties: is a json string serialized dictionary holding:

      key = "ratings_uri", value = uri for ratings file in ml1m format,
      key = "movies_uri", value = uri for movies file in ml1m format,
      key = "users_uri", value = uri for users file in ml1m format,
      key = headers_present, value = Bool for whether of not the first
        line in the ratings, movies, and users files are headers.
      key = delim, value = the delimiter used between columns in a row.
        e.g. "::"
      key = "ratings_key_col_dict", value = dictionary for ratings_uri of
        keys=coulmn names and values = the column numbers,
      key = "movies_key_col_dict", value = dictionary for ratings_uri of
        keys=coulmn names and values = the column numbers,
      key = "users_key_col_dict", value = dictionary for ratings_uri of
        keys=coulmn names and values = the column numbers,
      key = buckets, value = a list of integers as percents of the whole,
        e.g. [80, 10, 10]
      key = bucket_names, value = a list of strings of names for the
        buckets,  e.g. ['train', 'eval', test']

    :return:  a dictionary of the merged, split data as keys=name, values
      = PCollection for a partition
    """

    print(f'DEBUG exec_properties')
    print(f'type={type(exec_properties)}')
    for k, v in exec_properties.items():
      print(f'key={k}, value={v}')

    '''
    try:
      exec_properties = json.loads(exec_properties)
    except Exception as ex:
      err = f'exec_properties must hold the result of json.dumps(dict)'
      logger.error(f'ERROR: {err}: {ex}')
      raise ValueError(f'ERROR: {err}: {ex}')
    '''
    headers_present = exec_properties['headers_present']
    bucket_names = exec_properties['bucket_names']
    buckets = exec_properties['buckets']

    '''
    try:
      headers_present = json.loads(headers_present)
    except Exception as ex:
      err = f'exec_properties["headers_present"] must hold the result of json.dumps(True or False)'
      logger.error(f'ERROR: {err}: {ex}')
      raise ValueError(f'ERROR: {err}: {ex}')

    try:
      buckets = json.loads(buckets)
    except Exception as ex:
      err = f'exec_properties["buckets"] must hold the result of json.dumps(list of int)'
      logger.error(f'ERROR: {err}: {ex}')
      raise ValueError(f'ERROR: {err}: {ex}')

    try:
      bucket_names = json.loads(bucket_names)
    except Exception as ex:
      err = f'exec_properties["bucket_names"] must hold the result of json.dumps(list of str)'
      logger.error(f'ERROR: {err}: {ex}')
      raise ValueError(f'ERROR: {err}: {ex}')
    '''

    if len(buckets) != len(bucket_names):
      err = (f'deserialized buckets must be same length as deserialized bucket_names'
             f' buckets={buckets}, bucket_names={bucket_names}')
      logger.error(f'ERROR: {err}: {ex}')
      raise ValueError(f'ERROR: {err}: {ex}')

    #apache_beam.pvalue.DoOutputsTuple
    ratings_tuple = ingest_and_join(pipeline=pipeline, \
      ratings_uri=exec_properties['ratings_uri'], \
      movies_uri=exec_properties['movies_uri'], \
      users_uri=exec_properties['users_uri'], \
      headers_present=headers_present, \
      delim=exec_properties['delim'], \
      ratings_key_dict=exec_properties['ratings_key_col_dict'], \
      users_key_dict=exec_properties['users_key_col_dict'], \
      movies_key_dict=exec_properties['movies_key_col_dict'])

    #https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/components/example_gen/base_example_gen_executor.py#L120
    #line 200-ish
    #split the data

    splits = []
    for n, b in zip(bucket_names, buckets):
      splits.append(example_gen_pb2.SplitConfig.Split(name=n, percentage=b))

    output_config = example_gen_pb2.Output(
      split_config=example_gen_pb2.SplitConfig(
        splits=splits
      )
    )

    #_PartitionFn is from BaseExampleGenExecutor via base_example_gen_executor.py
    example_splits = (
      pipeline | 'SplitData' >> beam.Partition(_PartitionFn, len(buckets),
      buckets, output_config.split_config))

    result = {}
    for index, example_split in enumerate(example_splits):
      result[split_names[index]] = example_split
    return result

class IngestMovieLensComponent(base_component.BaseComponent):
  SPEC_CLASS = IngestMovieLensExecutorSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(IngestMovieLensExecutor)

  def __init__(self,\
    name : Optional[Text],
    ratings_uri : Text, \
    movies_uri : Text, \
    users_uri : Text, \
    headers_present : bool, \
    delim : Text, \
    ratings_key_col_dict : Dict[str,int], \
    users_key_col_dict : Dict[str,int], \
    movies_key_col_dict : Dict[str,int], \
    bucket_names : List[str], \
    buckets : List[int], \
    output_examples : Optional[types.Channel] = None):

    print(f'DEBUG IngestMovieLensComponent init')

    if not output_examples:
      output_examples = types.Channel(type=standard_artifacts.Examples)

    spec = IngestMovieLensExecutorSpec(
      name=name, ratings_uri=ratings_uri, movies_uri=movies_uri,\
      users_uri=users_uri, headers_present=headers_present,\
      delim=delim, ratings_key_col_dict=ratings_key_col_dict,\
      users_key_col_dict=users_key_col_dict,\
      movies_key_col_dict=movies_key_col_dict,\
      bucket_names=bucket_names, buckets=buckets,\
      output_examples=output_examples)

    super().__init__(spec=spec)


if __name__ == "__main__":

  #TODO: move this out of source code and into test code
  #  ... unittest.mock in Python
  # see https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/base_example_gen_executor_test.py

  from tfx.dsl.components.base import executor_spec
  print(f'executor_spec={executor_spec.ExecutorClassSpec(IngestMovieLensExecutor)}')

  from tfx.orchestration.experimental.interactive.interactive_context import \
    InteractiveContext

  kaggle = True
  if kaggle:
    prefix = '/kaggle/working/ml-1m/'
  else:
    prefix = "../resources/ml-1m/"
  ratings_uri = f"{prefix}ratings.dat"
  movies_uri = f"{prefix}movies.dat"
  users_uri = f"{prefix}users.dat"

  ratings_key_col_dict = {"user_id": 0, "movie_id": 1, "rating": 2,\
                          "timestamp": 3}
  movies_key_col_dict = {"movie_id": 0, "title": 1, "genres": 2}
  users_key_col_dict = {"user_id": 0, "gender": 1, "age": 2, \
                        "occupation": 3, "zipcode": 4}
  delim = "::"

  #these might need to be serialized into strings for tfx,
  # use json.dumps
  headers_present = False
  buckets = [80, 10, 10]
  bucket_names = ['train', 'evel', 'test']

  #exec_properties = {ratings_uri:ratings_uri, movies_uri:movies_uri, \
  #  users_uri:users_uri, ratings_key_col_dict:ratings_key_col_dict, \
  #  movies_key_col_dict:movies_key_col_dict, \
  #  users_key_col_dict:users_key_col_dict, \
  #  headers_present:headers_present, delim:delim, buckets:buckets \
  #}
  #exec_properties = json.dumps(exec_properties)

  context = InteractiveContext()

  ratings_example_gen = IngestMovieLensComponent( \
    name=name, ratings_uri=ratings_uri, movies_uri=movies_uri, \
    users_uri=users_uri, headers_present=headers_present, \
    delim=delim, ratings_key_col_dict=ratings_key_col_dict, \
    users_key_col_dict=users_key_col_dict, \
    movies_key_col_dict=movies_key_col_dict, \
    bucket_names=bucket_names, buckets=buckets \
  )

  context.run(component=ratings_example_gen, enable_cache=True)

  print("context run finished")

  #pipeline.Pipeline( components=[example_gen, hello, statistics_gen, ...]

  #https://www.tensorflow.org/tfx/tutorials/tfx/recommenders#create_inspect_examples_utility
  def inspect_examples(component, channel_name='examples', split_name='train', num_examples=1):
    # Get the URI of the output artifact, which is a directory
    full_split_name = 'Split-{}'.format(split_name)

    print(\
      'channel_name: {}, split_name: {} (\"{}\"), num_examples: {}\n'.format(\
      channel_name, split_name, full_split_name, num_examples))

    train_uri = os.path.join(component.outputs[channel_name].get()[0].uri, full_split_name)

    # Get the list of files in this directory (all compressed TFRecord files)
    tfrecord_filenames = [os.path.join(train_uri, name) for name in os.listdir(train_uri)]

    # Create a `TFRecordDataset` to read these files
    dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")

    # Iterate over the records and print them
    for tfrecord in dataset.take(num_examples):
      serialized_example = tfrecord.numpy()
      example = tf.train.Example()
      example.ParseFromString(serialized_example)
      pp.pprint(example)

  inspect_examples(ratings_example_gen)

  print("tests done")
