import apache_beam as beam
from ingest_movie_lens_1m_beam import join_and_split

import os
import absl
import json
import pprint
import tempfile

from typing import Any, Dict, List, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import apache_beam as beam

from absl import logging

from tfx.components.example_gen.base_example_gen_executor import BaseExampleGenExecutor
from tfx.components.example_gen.component import FileBasedExampleGen
from tfx.components.example_gen import utils
from tfx.dsl.components.base import executor_spec

from tfx.types import artifact
from tfx.types import artifact_utils
from tfx.types import channel
from tfx.types import standard_artifacts
from tfx.types.standard_artifacts import Examples

from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.decorators import component
from tfx.types.experimental.simple_artifacts import Dataset

from tfx import v1 as tfx
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

# Set up logging.
tf.get_logger().propagate = False
absl.logging.set_verbosity(absl.logging.INFO)
pp = pprint.PrettyPrinter()

print(f"TensorFlow version: {tf.__version__}")
print(f"TFX version: {tfx.__version__}")

import json
from tfx.utils import json_utils

# for a TFX pipeline, we want the ingestion to be performed by
# an ExampleGen component
# that accepts input data and formats it as tf.Examples
#can constrain the python version with:
class MovieLens1mExecutor(BaseExampleGenExecutor):
  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for TF Dataset to TF examples."""
    return _TFDatasetToExample

@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _TIngestMergeSplitExample(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline,
    exec_properties: Dict[str, Any],
    split_pattern: str
    ) -> beam.pvalue.PCollection:
    """Read a TensorFlow Dataset and create tf.Examples"""
    custom_config = json.loads(exec_properties['custom_config'])
    dataset_name = custom_config['dataset']
    split_name = custom_config['split']

    builder = tfds.builder(dataset_name)
    builder.download_and_prepare()

    return (pipeline
            | 'MakeExamples' >> tfds.beam.ReadFromTFDS(builder, split=split_name)
            | 'AsNumpy' >> beam.Map(tfds.as_numpy)
            | 'ToDict' >> beam.Map(dict)
            | 'ToTFExample' >> beam.Map(utils.dict_to_example)
            )

# base_image="python:3.11-slim")
@tfx.dsl.components.component(use_beam=True)
def ReadMergeAndSplitComponent(\
  input_dict_ser: standard_artifacts.String,\
  output_examples: tfx.dsl.components.OutputArtifact[Examples]\
  ) -> None:
  '''
  :param input_dict_ser: a json stringified dictionary of the folling:
    key = "ratings.dat", value = uri for ratings file in ml1m format,
    key = "movies.dat", value = uri for movies file in ml1m format,
    key = "users.dat", value = uri for users file in ml1m format,
    key = "ratings_key_dict", value = dictionary for ratings.dat of
      keys=coumn names and values = the column numbers,
    key = "movies_key_dict", value = dictionary for movies.dat of
      keys=coumn names and values = the column numbers,
    key = "users_key_dict", value = dictionary for users.dat of
      keys=coumn names and values = the column numbers,
    key = partitions, value = a list of integers as percents of the whole

    NOTE: to convert the dictionary into a string see stringify_ingest_params.py

  :param output_examples: where the ratings with user and movie merge
    partitioned Examples are stored.  Do not pass this in as input.
    This is declared in the method but should not be passed in by user.
    The output_examples.uri is managed by TFX's ML Metadata (MLMD)
    system and the orchestrator (e.g., Apache Airflow,
    Kubeflow Pipelines). When this component executes,
    TFX automatically creates a directory in the pipeline's workspace
    for this output artifact and provides your component with the
    path to write the Examples.data
  '''
  input_dict = json.loads(input_dict_ser)

  ratings_uri = input_dict['ratings.dat']
  movies_uri = input_dict['movies.dat']
  users_uri = input_dict['users.dat']

  ratings_key_dict = input_dict['ratings_key_dict']
  movies_key_dict = input_dict['movies_key_dict']
  users_key_dict = input_dict['users_key_dict']

  partitions = input_dict['partitions']

  #pipeline = self._MakeBeamPipeline()

  with beam.Pipeline() as pipeline:
    # ratings is a tuple of the partitions
    ratings = join_and_split(pipeline=pipeline, \
                             ratings_uri=ratings_uri, movies_uri=movies_uri, \
                             users_uri=users_uri, \
                             ratings_key_dict=ratings_key_dict, \
                             users_key_dict=users_key_dict, \
                             movies_key_dict=movies_key_dict, buckets=partitions)

    #The output_examples.uri is managed by TFX's ML Metadata (MLMD)
    # system and the orchestrator (e.g., Apache Airflow,
    # Kubeflow Pipelines). When this component executes,
    # TFX automatically creates a directory in the pipeline's workspace
    # for this output artifact and provides your component with the
    # path to write the Examples.data

    #write to output.  this could be improved
    names = []
    for i, part in enumerate(ratings):
      # Write to the output artifact's URI
      # io_utils.get_uri_for_writing_result(output_examples.uri)
      names.append(f'part_{i}_')
      output_split_path = os.path.join(output_examples.uri, names[-1])
      _ = part | f'write split {i}' >> beam.io.tfrecordio.WriteToTFRecord(\
        file_path_prefix=output_split_path,
        shard_name_template='',
        # Use empty template if you want a single file or manage sharding explicitly
        #file_name_suffix='.tfrecord',
        coder=beam.coders.BytesCoder()
    )
    output_examples.split_names = str(names)
