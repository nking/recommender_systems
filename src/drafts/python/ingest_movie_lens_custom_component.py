import apache_beam as beam

#NOTE: this component is not correct yet.
#TODO: consider building the output_config outside of classes and passing
# the argument in.

import absl
import pprint
import time
import json

from typing import Any, Dict, List, Text, Optional, Union, Tuple

import tensorflow as tf

from absl import logging

from tfx.components.example_gen.base_example_gen_executor import BaseExampleGenExecutor
#from tfx.components.example_gen import utils
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

from tfx.proto import example_gen_pb2

from ingest_movie_lens_beam import ingest_and_join
from movie_lens_utils import *

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

#from tfx.utils import json_utils

#tf.train.Example is for independent, fixed-size examples
#tf.train.SequenceExample is for variable-length sequential data,
#    such as sentences, time series, or videos.

## TODO: follow up on adding output splitconfig as input to the
#        Component instead of using separate arguments for
#        buckets and bucket_names.
#        it can only be placed in ComponentSpec Parameters
#        not INPUTS, so no automatic MLMD registration unfortunately


class IngestMovieLensExecutorSpec(ComponentSpec):
  """ComponentSpec for Custom TFX MovieLensExecutor Component."""
  #PARAMETERS, INPUTS, and OUTPUTS are static instead of instance vars
  PARAMETERS = {
    # These are parameters that will be passed in the call to
    # create an instance of this component.
    'name': ExecutionParameter(type=Text),
    'infiles_dict_ser' : ExecutionParameter(type=Text),
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
    #'output_split_config': ChannelParameter(type=standard_artifacts.SplitConfig),
  }

# for a TFX pipeline, we want the ingestion to be performed by
# an ExampleGen component that accepts input data and formats it as tf.Examples

# we're extending BaseExampleGenExecutor. it invokes its own Do method.
# see https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/components/example_gen/base_example_gen_executor.py#L222
# the Do method invokes GenerateExamplesByBeam.
# GenerateExamplesByBeam uses GetInputSourcesToExamplePTransform.
# The later is abstract in BaseExampleGenExecutor so must be implemented.
#
# The reason for overriding the Do method of BaseExampleGenExecutor here
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

  def GetInputSourceToExamplePTransform(self, \
    pipeline : beam.Pipeline, \
    infiles_dict: Dict[str, Union[str, Dict]]) -> beam.PTransform:
    """Returns PTransform for ratings, movies, users joined to TF examples."""
    return ingest_and_join

  def GenerateExamplesByBeam(self, \
    pipeline: beam.Pipeline, \
    exec_properties: Dict[str, Any], output_dict: Dict[str, List[types.Artifact]]\
  ) -> Tuple[Dict[str, beam.PCollection], List[Tuple[str, Any]]]:
    """
    :param pipeline:
    :param exec_properties: is a json string serialized dictionary holding:
      key = infiles_dict_ser which is a json serialization of the
        infiles_dict
      key = buckets, value = a list of integers as percents of the whole,
        e.g. [80, 10, 10]
      key = bucket_names, value = a list of strings of names for the
        buckets,  e.g. ['train', 'eval', test']
    :param output_dict:
    :return:  tuple of :
      a dictionary of the merged, split data as keys=name, values
         = PCollection for a partition,
      and a list of tuple of the column names and types
    """

    #following https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/components/example_gen/base_example_gen_executor.py#L103

    logging.debug(\
      "in IngestMovieLensExecutor.GenerateExamplesByBeam")

    try:
      infiles_dict = json.loads(infiles_dict_ser.decode('utf-8'))
    except Exception as ex:
      logging.error(f'ERROR: {ex}')
      raise ValueError(f'ERROR: {ex}')

    ratings, column_name_type_list = \
      self.GetInputSourceToExamplePTransform(pipeline=pipelne, infiles_dict=infiles_dict)

    bucket_names = exec_properties['bucket_names']
    buckets = exec_properties['buckets']
    if len(buckets) != len(bucket_names):
      err = (f'deserialized buckets must be same length as deserialized bucket_names'
             f' buckets={buckets}, bucket_names={bucket_names}')
      logging.error(f'ERROR: {err}')
      raise ValueError(f'ERROR: {err}')

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

    #type: apache_beam.DoOutputsTuple
    ratings_tuple = ratings | f'split_{time.time_ns()}' >> beam.Partition( \
      partition_fn, len(buckets), cumulative_buckets, \
      output_config.split_config)

    result = {}
    for index, example_split in enumerate(ratings_tuple):
      result[bucket_names[index]] = example_split
    # pass to Do method
    return result, column_name_type_list

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
      # apache_beam.DoOutputsTuple
      ratings_dict, column_name_type_list = \
        self.GenerateExamplesByBeam(pipeline, exec_properties, output_dict)

      write_to_tfrecords = True
      output_examples = output_dict['output_examples']
      if output_examples is None:
        logging.error(
          "ERROR: fix coding error for missing output_examples")
        raise ValueError(
          "Error: fix coding error for missing output_examples")

      bucket_names = exec_properties['bucket_names']

      output_examples.splits = bucket_names.copy()

      # https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/proto/example_gen.proto#L44
      # If VERSION is specified, but not SPAN, an error will be thrown.
      # output_examples.version = infiles_dict['version']

      # output_examples.set_string_custom_property('description',\
      #  'ratings file created from left join of ratings, users, movies')

      # https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/types/standard_artifacts/Examples
      # files should be written as {uri}/Split-{split_name1}

      #could use WriteSplit method instead:
      #https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/components/example_gen/base_example_gen_executor.py#L281

      if not write_to_tfrecords:
        # write to csv
        column_names = ",".join([t[0] for t in column_name_type_list])
        for name, example in enumerate(ratings_dict):
          prefix_path = f'{output_examples.uri}/Split-{name}'
          write_to_csv(pcollection=example, \
                       column_names=column_names,
                       prefix_path=prefix_path, delim='_')
        logging.info('Examples written to output_examples as CSV.')
      else:
        # write to TFRecords
        for name, example in enumerate(ratings_dict):
          prefix_path = f'{output_examples.uri}/Split-{name}'
          convert_to_tf_example(example, column_name_type_list) \
            | f"Serialize {time.time_ns()}" >> beam.Map(
            lambda x: x.SerializeToString()) \
            | f"write_to_tf {time.time_ns()}" >> beam.io.tfrecordio.WriteToTFRecord( \
            file_path_prefix=prefix_path, file_name_suffix='.tfrecord')
        logging.info(
          'Examples written to output_examples as TFRecords.')
      # no return

#class IngestMovieLensComponent(base_component.BaseComponent):
class IngestMovieLensComponent(base_beam_component.BaseBeamComponent):
  SPEC_CLASS = IngestMovieLensExecutorSpec
  #EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(IngestMovieLensExecutor)
  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(IngestMovieLensExecutor)

  def __init__(self,\
    name : Optional[Text],
    infiles_dict_ser : Text,\
    bucket_names : List[str], \
    buckets : List[int], \
    output_examples : Optional[types.Channel] = None):

    print(f'DEBUG IngestMovieLensComponent init')

    if not output_examples:
      examples_artifact = standard_artifacts.Examples()
      examples_artifact.splits = bucket_names.copy()
      output_examples = channel_utils.as_channel([examples_artifact])

    spec = IngestMovieLensExecutorSpec(
      name=name, infiles_dict_ser=infiles_dict_ser, \
      bucket_names=bucket_names, buckets=buckets,\
      output_examples=output_examples)

    #super().__init__(spec=spec)
    super(IngestMovieLensComponent, self).__init__(spec=spec)

'''
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

'''