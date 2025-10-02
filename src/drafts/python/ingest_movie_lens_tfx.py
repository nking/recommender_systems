import apache_beam as beam

#NOTE: this component is not correct yet.
#TODO: consider building the output_config outside of classes and passing
# the arugment in.
# in current condition, the error returned is:
#File "/usr/local/envs/my_tfx_env/lib/python3.10/site-packages/tfx/types/standard_artifact_utils.py",
# line ..., in get_split_uri
#    raise ValueError(
#ValueError: Expected exactly one artifact with split 'train', but found matching artifacts [].

import os
import absl
import json
import pprint

from typing import Any, Dict, List, Text, Optional, Union

import numpy as np
import tensorflow as tf

from absl import logging

from tfx.components.example_gen.base_example_gen_executor import BaseExampleGenExecutor
#from tfx.components.example_gen import utils
from tfx.components.example_gen import write_split
from tfx.components.util import examples_utils
from tfx.dsl.components.base import executor_spec, base_component, base_beam_component
from tfx import types
from tfx.types.component_spec import ChannelParameter, ComponentSpec, ExecutionParameter

from tfx.types import artifact_utils, standard_artifacts, standard_component_specs

#from tfx.dsl.component.experimental.annotations import InputArtifact
#from tfx.dsl.component.experimental.annotations import OutputArtifact
#from tfx.dsl.component.experimental.annotations import Parameter
#from tfx.dsl.component.experimental.decorators import component
#from tfx.types.experimental.simple_artifacts import Dataset

from ingest_movie_lens_beam import ingest_and_join
from partition_funcs import partitionFn

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

class IngestMovieLensExecutorSpec(ComponentSpec):
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
    # these are tracked by MLMD.  They are usually the output artifacts
    #   of an upstream component.
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
    # these are tracked by MLMD.
    # OUTPUTS will be a dictionary which this component will populate
    'output_examples': ChannelParameter(type=standard_artifacts.Examples),
  }

# for a TFX pipeline, we want the ingestion to be performed by
# an ExampleGen component that accepts input data and formats it as tf.Examples

# we're extending BaseExampleGenExecutor. it invokes it's own Do method.
# see https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/components/example_gen/base_example_gen_executor.py#L222
# the Do method invokes GenerateExamplesByBeam.
# GenerateExamplesByBeam uses GetInputSourcesToExamplePTransform.
# The later is abstract in BaseExampleGenExecutor so must be implemented.
#
# The reason for overrriding the Do method of BaseExampleGenExecutor here
# is because using a split (data partitions by directory or file name)
# naively, seems like it might be more complicated for uris that are
# not file sources, and writing an input_config example_gen_pb2.Input.Split
# pattern when all files are in the same directory isn't clear.
#
# It might be simpler to extend
# base_executor.BaseExecutor and override the Do method instead of
# extending BaseExampleGenExecutor as is done here.

class IngestMovieLensExecutor(BaseExampleGenExecutor):
  """executor to ingest movie lens data, join, and split into buckets"""
  @beam.ptransform_fn
  @beam.typehints.with_input_types(beam.Pipeline)
  @beam.typehints.with_output_types(tf.train.Example)
  def _MovieLensToExample( # pylint: disable=invalid-name
    self, pipeline: beam.Pipeline, exec_properties: Dict[str, Any],
    split_pattern: str) -> Dict[str, beam.pvalue.PCollection]:
    """Read ratings, movies, and users files, join them,
    and transform to TF examples.

    Args:
      pipeline: beam pipeline.
      exec_properties: A dict of execution properties.
        - input_base: input dir that contains Avro data.
      split_pattern: Split.pattern in Input config, glob relative file pattern
        that maps to input files with root directory given by input_base.
        Not used.

    Returns:
      PCollection of TF examples.
    """

    print(f'DEBUG exec_properties')
    #print(f'type={type(exec_properties)}')
    #for k, v in exec_properties.items():
    #  print(f'key={k}, value={v}')

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

    return ratings_tuple

  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for ratings, movies, users joined to TF examples."""
    logging.debug("in IngestMovieLensExecutor.GetInputSourceToExamplePTransform")
    return self._MovieLensToExample

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

    #https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/components/example_gen/base_example_gen_executor.py#L120
    #line 200-ish
    #split the data

    logging.debug(\
      "in IngestMovieLensExecutor.GenerateExamplesByBeam")

    ratings_tuple = self.GetInputSourceToExamplePTransform()

    bucket_names = exec_properties['bucket_names']
    buckets = exec_properties['buckets']

    splits = []
    for n, b in zip(bucket_names, buckets):
      #see https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/proto/example_gen.proto#L146
      splits.append(example_gen_pb2.SplitConfig.Split(name=n, hash_buckets=b))

    output_config = example_gen_pb2.Output(
      split_config=example_gen_pb2.SplitConfig(
        splits=splits
      )
    )

    total = sum(buckets)
    s = 0
    cumulative_buckets = []
    for b in buckets:
      s += int(100 * (b / total))
      cumulative_buckets.append(s)

    #_PartitionFn is from BaseExampleGenExecutor via base_example_gen_executor.py
    example_splits = (\
      pipeline | 'SplitData' >> beam.Partition(partitionFn, len(buckets),\
      cumulative_buckets, output_config.split_config))

    result = {}
    for index, example_split in enumerate(example_splits):
      result[bucket_names[index]] = example_split
    return result

  def Do(
    self,
    input_dict: Dict[str, List[types.Artifact]],
    output_dict: Dict[str, List[types.Artifact]],
    exec_properties: Dict[str, Any],
  ) -> None:
    """Takes input data sources and generates serialized data splits.

    Args:
      input_dict: Input dict from input key to a list of Artifacts. Depends on
        detailed example gen implementation.
      output_dict: Output dict from output key to a list of Artifacts.
        - examples: splits of serialized records.
      exec_properties: A dict of execution properties.

    Returns:
      None
    """
    logging.debug("in IngestMovieLensExecutor.Do")
    #https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/components/example_gen/base_example_gen_executor.py#L281
    with self._make_beam_pipeline() as pipeline:
      example_splits = self.GenerateExamplesByBeam(pipeline, exec_properties)

      # pylint: disable=expression-not-assigned, no-value-for-parameter
      for split_name, example_split in example_splits.items():
        logging.debug(f"isplit_name={split_name}")
        (example_split | f'WriteSplit[{split_name}]' >> write_split.WriteSplit( \
        artifact_utils.get_split_uri( \
        output_dict[standard_component_specs.EXAMPLES_KEY], \
        split_name), output_file_format, exec_properties))
      # pylint: enable=expression-not-assigned, no-value-for-parameter

    output_payload_format = exec_properties.get(\
      standard_component_specs.OUTPUT_DATA_FORMAT_KEY)
    if output_payload_format:
      for output_examples_artifact in output_dict[standard_component_specs.EXAMPLES_KEY]:
        examples_utils.set_payload_format(output_examples_artifact, \
          output_payload_format)

    if output_file_format:
      for output_examples_artifact in output_dict[standard_component_specs.EXAMPLES_KEY]:
        examples_utils.set_file_format(output_examples_artifact,\
          write_split.to_file_format_str(output_file_format))

    logging.info('Examples generated.')

#class IngestMovieLensComponent(base_component.BaseComponent):
class IngestMovieLensComponent(base_beam_component.BaseBeamComponent):
  SPEC_CLASS = IngestMovieLensExecutorSpec
  #EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(IngestMovieLensExecutor)
  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(IngestMovieLensExecutor)

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

    #super().__init__(spec=spec)
    super(IngestMovieLensComponent, self).__init__(spec=spec)

if __name__ == "__main__":

  from tfx.orchestration.local.local_dag_runner import LocalDagRunner

  #TODO: move this out of source code and into test code
  #  ... unittest.mock in Python
  # see https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/base_example_gen_executor_test.py

  print(f'begin test')

  from tfx.dsl.components.base import executor_spec
  print(f'executor_spec={executor_spec.ExecutorClassSpec(IngestMovieLensExecutor)}')

  kaggle = True
  if kaggle:
    prefix = '/kaggle/working/ml-1m/'
    output_dir = '/kaggle/working/bin'
  else:
    prefix = "../resources/ml-1m/"
    output_dir = '../../bin'
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
  bucket_names = ['train', 'eval', 'test']

  name = "test_tfx_component"

  #exec_properties = {ratings_uri:ratings_uri, movies_uri:movies_uri, \
  #  users_uri:users_uri, ratings_key_col_dict:ratings_key_col_dict, \
  #  movies_key_col_dict:movies_key_col_dict, \
  #  users_key_col_dict:users_key_col_dict, \
  #  headers_present:headers_present, delim:delim, buckets:buckets \
  #}
  #exec_properties = json.dumps(exec_properties)

  #context = InteractiveContext()

  ratings_example_gen = IngestMovieLensComponent( \
    name=name, ratings_uri=ratings_uri, movies_uri=movies_uri, \
    users_uri=users_uri, headers_present=headers_present, \
    delim=delim, ratings_key_col_dict=ratings_key_col_dict, \
    users_key_col_dict=users_key_col_dict, \
    movies_key_col_dict=movies_key_col_dict, \
    bucket_names=bucket_names, buckets=buckets \
  )

  statistics_gen = tfx.components.StatisticsGen(\
    examples=ratings_example_gen.outputs['output_examples'])

  PIPELINE_NAME = "MovieLensIngestTest"
  PIPELINE_ROOT = os.path.join(output_dir, 'pipelines', PIPELINE_NAME)
  # Path to a SQLite DB file to use as an MLMD storage.
  METADATA_PATH = os.path.join(output_dir, 'metadata', PIPELINE_NAME, \
    'metadata.db')

  os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)
  os.makedirs(PIPELINE_ROOT, exist_ok=True)

  my_pipeline = tfx.dsl.Pipeline(\
    pipeline_name=PIPELINE_NAME, \
    pipeline_root=PIPELINE_ROOT,\
    metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH),\
    components=[ratings_example_gen], enable_cache=True)

  LocalDagRunner().run(my_pipeline)

  #https://www.tensorflow.org/tfx/tutorials/tfx/recommenders#create_inspect_examples_utility
  def inspect_examples(component, channel_name='examples', split_name='train', num_examples=1):
    # Get the URI of the output artifact, which is a directory
    full_split_name = 'Split-{}'.format(split_name)

    print(\
      'channel_name: {}, split_name: {} (\"{}\"), num_examples: {}\n'.format(\
      channel_name, split_name, full_split_name, num_examples))

    train_uri = os.path.join(\
      component.outputs[channel_name].get()[0].uri, full_split_name)

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

  inspect_examples(ratings_example_gen, channel_name='output_examples')

  print("tests done")
