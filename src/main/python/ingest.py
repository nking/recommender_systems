import apache_beam as beam

from ingest_beam import merge_and_split

import os
import tensorflow as tf
from tfx.components.example_gen import base_example_gen_executor
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx_bsl.public import tfxio
import time

class ReadMergeAndSplitExecutor(base_example_gen_executor.BaseExampleGenExecutor):
    """Custom executor for ExampleGen that handles multiple files."""
  def Do (
    self,
    input_dict: tf.TfList[standard_artifacts.ExternalSource],
    output_dict: tf.TfList[standard_artifacts.Examples],
    exec_properties: tf.TfDict[str, tf.TfAny]
  ) -> None:
    # Get the input file paths from the input_dict.
    # The keys will depend on your component definition.
    ratings_uri = input_dict['ratings.dat'].uri
    movies_uri = input_dict['movies.dat'].uri
    users_uri = input_dict['users.dat'].uri

    ratings_key_dict = input_dict['ratings_key_dict']
    movies_key_dict = input_dict['movies_key_dict']
    users_key_dict = input_dict['users_key_dict']

    pipeline = self._MakeBeamPipeline()

    ratings = merge_and_split(pipeline=pipeline, \
                              ratings_uri='ratings.dat', movies_uri='movies.dat', \
                              users_uri='users.dat', \
                              ratings_key_dict=ratings_key_dict, \
                              users_key_dict=users_key_dict, \
                              movies_key_dict=movies_key_dict, \
                              partition_percents=partition_percents)

    #not yet finished

    #write to output files if wanted
    # The output path is determined by the output_dict.
    output_path = os.path.join(output_dict['examples'].uri, self._MakeSplitName())
    _ = (
    merged_data
    | 'WriteExamples' >> tfxio.WriteTFXIOToTFRecords(output_path)
    )

    # Run the Beam pipeline?
    pipeline.run()
