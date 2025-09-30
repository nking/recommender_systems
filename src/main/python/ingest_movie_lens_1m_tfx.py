import apache_beam as beam
from ingest_movie_lens_1m_beam import merge_and_split

import os
import tensorflow as tf
from tfx.components.example_gen import base_example_gen_executor
#from tfx.dsl.component.experimental.decorators import component
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx_bsl.public import tfxio

import typing
from tfx.types import artifact
from tfx.types import channel
from tfx.types import standard_artifacts
from tfx.types.standard_artifacts import Examples
from tfx.types.artifact import Outputs

import json
from tfx.utils import json_utils

#can constrain the python version with:
# base_image="python:3.11-slim")
@tfx.dsl.components.component(use_beam=True)
def ReadMergeAndSplitComponent(\
  input_dict_ser: standard_artifacts.String,
  output_examples: Outputs[Examples]
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
    ratings = merge_and_split(pipeline=pipeline, \
                            ratings_uri=ratings_uri, movies_uri=movies_uri, \
                            users_uri=users_uri, \
                            ratings_key_dict=ratings_key_dict, \
                            users_key_dict=users_key_dict, \
                            movies_key_dict=movies_key_dict, partitions=partitions)

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
