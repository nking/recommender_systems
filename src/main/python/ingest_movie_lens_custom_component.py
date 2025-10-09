import apache_beam as beam

import absl
import pprint
import time
import random
import os

from typing import Any, Dict, List, Text, Optional, Union, Tuple

import tensorflow as tf

from absl import logging

from tfx.components.example_gen.base_example_gen_executor import BaseExampleGenExecutor
#from tfx.components.example_gen import utils
from tfx.components.util import examples_utils
from tfx.dsl.components.base import executor_spec, base_component, base_beam_component
from tfx import types
from tfx.types.component_spec import ChannelParameter, ComponentSpec, ExecutionParameter

from tfx.types import artifact_utils, standard_artifacts, \
  standard_component_specs, channel_utils

#from tfx.dsl.component.experimental.annotations import InputArtifact
#from tfx.dsl.component.experimental.annotations import OutputArtifact
#from tfx.dsl.component.experimental.annotations import Parameter
#from tfx.dsl.component.experimental.decorators import component
#from tfx.types.experimental.simple_artifacts import Dataset

from tfx.proto import example_gen_pb2

from ingest_movie_lens_beam import *
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

class IngestMovieLensExecutorSpec(ComponentSpec):
  """ComponentSpec for Custom TFX MovieLensExecutor Component."""
  #PARAMETERS, INPUTS, and OUTPUTS are static instead of instance vars
  PARAMETERS = {
    # These are parameters that will be passed in the call to
    # create an instance of this component.
    'name': ExecutionParameter(type=Text),
    'infiles_dict_ser' : ExecutionParameter(type=Text),
    #output_config has to be string serialzied because the orchestrator uses pickle to save it
    #'output_config': ExecutionParameter(type=example_gen_pb2.Output, use_proto=True),
    'output_config_ser': ExecutionParameter(type=Text),
    #output_config should include split_config
  }
  INPUTS = {
    # these are tracked by MLMD.  They are usually the output artifacts
    #   of an upstream component.
    # INPUTS will be a dictionary with input artifacts, including URIs
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

class IngestMovieLensExecutor(BaseExampleGenExecutor):
  """executor to ingest movie lens data, left join on ratings, and split"""

  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for ratings, movies, users joined to TF examples."""

    @beam.ptransform_fn
    @beam.typehints.with_input_types(beam.Pipeline, Dict[str, Union[str, Dict]])
    @beam.typehints.with_output_types(Tuple[tf.train.Example, List[Tuple[str, Any]]])
    def ingest_to_examples(pipeline: beam.Pipeline, exec_properties: Dict[str, Any]) \
      -> Tuple[tf.train.Example, List[Tuple[str, Any]]]:
      try:
        infiles_dict_ser = exec_properties['infiles_dict_ser']
        infiles_dict = deserialize(infiles_dict_ser)
        logging.debug(f'infiles_dict_ser={infiles_dict_ser}\ninfiles_dict={infiles_dict}')
      except Exception as ex:
        logging.error(f'ERROR: {ex}')
        raise ValueError(f'ERROR: {ex}')

      (ratings_pc, column_name_type_list) = \
        pipeline | f"IngestAndJoin_{random.randint(0, 1000000000000)}" \
        >> IngestAndJoin(infiles_dict=infiles_dict)

      ratings_example = ratings_pc \
        | f'ToTFExample_{random.randint(0, 1000000000000)}' \
        >> beam.Map(create_example, column_name_type_list)
      return ratings_example, column_name_type_list

    return ingest_to_examples

  def GenerateExamplesByBeam(self, \
    pipeline: beam.Pipeline, \
    exec_properties: Dict[str, Any], output_dict: Dict[str, List[types.Artifact]]\
  ) -> Tuple[Dict[str, tf.train.Example], List[Tuple[str, Any]]]:
    """
    :param pipeline:
    :param exec_properties: is a json string serialized dictionary holding:
      key = infiles_dict_ser which is a json serialization of the
        infiles_dict
      key = output_config_ser which is string serialized and must contain split_config
    :param output_dict:
    :return:  tuple of (
      a dictionary of the merged, split data as keys=name, values
         = PCollection for a partition,
      and a list of tuple of the column names and types)
    """

    #following https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/components/example_gen/base_example_gen_executor.py#L103

    logging.debug(\
      "in IngestMovieLensExecutor.GenerateExamplesByBeam")

    input_to_examples = self.GetInputSourceToExamplePTransform()

    logging.debug( \
      "about to read input and transform to tf.train.Example")

    ratings_example, column_name_type_list = \
      pipeline | f"GetInputSource{random.randint(0, 1000000000000)}" \
      >> input_to_examples(exec_properties)

    try:
      output_config_ser = exec_properties['output_config_ser']
      output_config = deserialize_to_proto(output_config_ser)
    except Exception as ex:
      logging.error(f"ERROR: {ex}")
      raise ValueError(ex)

    if not output_config or not output_config.HasField('split_config') \
      or not output_config.split_config.splits:
      raise ValueError("parameters must include output_config which"
        f" must contain split_config.splits.  output_config={output_config}")

    total = sum([split.hash_buckets for split in output_config.split_config.splits])
    s = 0
    cumulative_buckets = []
    for split in output_config.split_config.splits:
      s += int(100 * (split.hash_buckets / total))
      cumulative_buckets.append(s)

    logging.debug(f'cumulative_buckets={cumulative_buckets}')

    #type: apache_beam.DoOutputsTuple
    ratings_tuple = ratings_example | f'split_{time.time_ns()}' \
      >> beam.Partition( \
      partition_fn, len(cumulative_buckets), cumulative_buckets, output_config.split_config)

    split_names = [split.name for split in output_config.split_config.splits]
    result = {split_name : example for split_name, example in zip(ratings_tuple, split_names)}
    # pass back to Do method
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

      logging.debug( \
        "have read, left joined, converted to tf.tran.Example, and split."
        "  about to write to uri")

      output_examples = output_dict['output_examples']
      logging.debug(f"output_examples TYPE={type(output_examples)}")
      logging.debug(f"output_examples={output_examples}")

      if isinstance(output_dict['output_examples'], list):
        output_uri = artifact_utils.get_single_instance(output_dict['output_examples']).uri
      else:
        output_uri = output_examples.uri

      if output_examples is None:
        logging.error(
          "ERROR: fix coding error for missing output_examples")
        raise ValueError(
          "Error: fix coding error for missing output_examples")

      # https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/proto/example_gen.proto#L44
      # If VERSION is specified, but not SPAN, an error will be thrown.
      # output_examples.version = infiles_dict['version']

      # output_examples.set_string_custom_property('description',\
      #  'ratings file created from left join of ratings, users, movies')

      # https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/types/standard_artifacts/Examples
      # files should be written as {uri}/Split-{split_name1}

      #could use WriteSplit method instead:
      #https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/components/example_gen/base_example_gen_executor.py#L281

      DEFAULT_FILE_NAME = 'data_tfrecord'
      # write to TFRecords
      for split_name, example in ratings_dict.items():
        prefix_path = os.path.join(output_uri, split_name)
        logging.debug(f"prefix_path={prefix_path}")
        example | f"Serialize_{random.randint(0, 1000000000000)}" \
          >> beam.Map(lambda x: x.SerializeToString()) \
          | f"write_to_tfrecord_{random.randint(0, 1000000000000)}" \
          >> beam.io.tfrecordio.WriteToTFRecord( \
          os.path.join(prefix_path, DEFAULT_FILE_NAME), \
          file_name_suffix='.gz')
      logging.info('output_examples written as TFRecords')
      # no return

#class IngestMovieLensComponent(base_component.BaseComponent):
class IngestMovieLensComponent(base_beam_component.BaseBeamComponent):
  SPEC_CLASS = IngestMovieLensExecutorSpec
  #EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(IngestMovieLensExecutor)
  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(IngestMovieLensExecutor)

  def __init__(self,\
    name : Optional[Text],
    infiles_dict_ser : Text,\
    output_config_ser : Text,\
    output_examples : Optional[types.Channel] = None):

    logging.debug(f'DEBUG IngestMovieLensComponent init')

    if not output_examples:
      output_config = deserialize_to_proto(output_config_ser)
      split_names = [split.name for split in output_config.split_config.splits]
      examples_artifact = standard_artifacts.Examples()
      examples_artifact.splits = split_names
      output_examples = channel_utils.as_channel([examples_artifact])

    spec = IngestMovieLensExecutorSpec(
      name=name, \
      infiles_dict_ser=infiles_dict_ser, \
      output_config_ser=output_config_ser,\
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