# edited from unit test for Avro component at:
# https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/custom_executors/avro_component_test.py
# which has copyright:
#
# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Tests for using avro_executor with example_gen component."""

import os

import pickle
import base64
import random

from unittest import mock
import tensorflow as tf
from tfx.dsl.components.base import executor_spec
from tfx.dsl.io import fileio
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import publisher
from tfx.orchestration.launcher import in_process_component_launcher
from tfx.proto import example_gen_pb2
from tfx.utils import name_utils

import pprint
import absl
from absl import logging
absl.logging.set_verbosity(absl.logging.DEBUG)

from ingest_movie_lens_custom_component import *
from movie_lens_utils import *

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.metadata_store import metadata_store

from apache_beam.options.pipeline_options import PipelineOptions

class IngestMovieLensCustomComponentTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    kaggle = True
    if kaggle:
      prefix = '/kaggle/working/ml-1m/'
    else:
      prefix = "../resources/ml-1m/"
    ratings_uri = f"{prefix}ratings.dat"
    movies_uri = f"{prefix}movies.dat"
    users_uri = f"{prefix}users.dat"

    ratings_col_names = ["user_id", "movie_id", "rating", "timestamp"]
    ratings_col_types = [int, int, int, int]  # for some files, ratings are floats
    movies_col_names = ["movie_id", "title", "genres"]
    movies_col_types = [int, str, str]
    users_col_names = ["user_id", "gender", "age", "occupation", "zipcode"]
    users_col_types = [int, str, int, int, str]

    ratings_dict = create_infile_dict(for_file='ratings', \
                                      uri=ratings_uri,
                                      col_names=ratings_col_names, \
                                      col_types=ratings_col_types,
                                      headers_present=False, delim="::")

    movies_dict = create_infile_dict(for_file='movies', \
                                     uri=movies_uri,
                                     col_names=movies_col_names, \
                                     col_types=movies_col_types,
                                     headers_present=False, delim="::")

    users_dict = create_infile_dict(for_file='users', \
                                    uri=users_uri,
                                    col_names=users_col_names, \
                                    col_types=users_col_types,
                                    headers_present=False, delim="::")

    self.infiles_dict = create_infiles_dict(ratings_dict=ratings_dict, \
                                      movies_dict=movies_dict, \
                                      users_dict=users_dict, version=1)

    buckets = [80, 10, 10]
    bucket_names = ['train', 'eval', 'test']
    self.output_config = example_gen_pb2.Output(
      split_config=example_gen_pb2.SplitConfig(
        splits=[
          example_gen_pb2.SplitConfig.Split(name=n, hash_buckets=b) \
            for n, b in zip(bucket_names, buckets)]
      )
    )
    logging.debug(f"test self.output_config={self.output_config}")

    self.name = 'test run of ingest with tfx'

  def testRun2(self):

    test_num = "fully_custom_comp_1"
    infiles_dict_ser = serialize_to_string(self.infiles_dict)

    name = "test_fully_custom_component"

    #implement the task
    ratings_example_gen = IngestMovieLensComponent( \
      name=name,\
      infiles_dict_ser=infiles_dict_ser, \
      output_config=self.output_config)

    components = [ratings_example_gen]

    PIPELINE_NAME = 'TestFullyCustomCompPipeline'
    #output_data_dir = os.path.join(os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',self.get_temp_dir()),self._testMethodName)
    output_data_dir = os.path.join('/kaggle/working/bin/', test_num, self._testMethodName)
    PIPELINE_ROOT = os.path.join(output_data_dir, PIPELINE_NAME)
    METADATA_PATH = os.path.join(PIPELINE_ROOT, 'tfx_metadata', 'metadata.db')
    os.makedirs(os.path.join(PIPELINE_ROOT, 'tfx_metadata'), exist_ok=True)

    alt_output_data_dir = os.path.join(
      os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',
                     self.get_temp_dir()), self._testMethodName)
    print(f'alt_output_data_dir={alt_output_data_dir}')

    ENABLE_CACHE = False

    if not ENABLE_CACHE:
      if os.path.exists(METADATA_PATH):
        os.remove(METADATA_PATH)

    #metadata_connection_config = metadata_store_pb2.ConnectionConfig()
    #metadata_connection_config.sqlite.SetInParent()
    #metadata_connection = metadata.Metadata(metadata_connection_config)
    metadata_connection_config = metadata.sqlite_metadata_connection_config(METADATA_PATH)

    # apache-beam 2.59.0 - 2.68.0 with SparkRunner supports pyspark 3.2.x
    # but not 4.0.0
    # pyspark 3.2.4 is compatible with java >= 8 and <= 11 and python >= 3.6 and <= 3.9
    # start Docker, then use portable SparkRunner
    # https://beam.apache.org/documentation/runners/spark/
    # from pyspark import SparkConf

    my_pipeline = tfx.dsl.Pipeline(
      pipeline_name=PIPELINE_NAME,
      pipeline_root=PIPELINE_ROOT,
      components=components,
      enable_cache=ENABLE_CACHE,
      metadata_connection_config=metadata_connection_config,
      #beam_pipeline_args=beam_pipeline_args,
    )

    tfx.orchestration.LocalDagRunner().run(my_pipeline)

    # Check output paths.
    self.assertTrue(fileio.exists(os.path.join(PIPELINE_ROOT, \
      ratings_example_gen.id)))

    for key, value in ratings_example_gen.outputs.items():
      print(f'key={key}, value={value}')


    #list files in alt_output_data_dir and in output_data_dir
    print(f'listing files in output_data_dir {PIPELINE_ROOT}:')
    for dirname, _, filenames in os.walk(PIPELINE_ROOT):
      for filename in filenames:
        print(os.path.join(dirname, filename))

    store = metadata_store.MetadataStore(metadata_connection_config)
    artifact_types = store.get_artifact_types()
    logging.debug(f"MLMD store artifact_types={artifact_types}")
    self.assertEqual(1, len(artifact_types))
    artifacts = store.get_artifacts()
    logging.debug(f"MLMD store artifacts={artifacts}")
    self.assertEqual(1, len(artifacts))
    executions = store.get_executions()
    logging.debug(f"MLMD store executions={executions}")
    self.assertEqual(1, len(executions))
    #executions has custom_properties.key: "infiles_dict_ser"
    #    and custom_properties.key: "output_config_ser"
    artifact_count = len(artifacts)
    execution_count = len(executions)
    self.assertGreaterEqual(artifact_count, execution_count)

    """
    #self.assertIsNotNone(ratings_example_gen.outputs['output'].get()[0])
    #output_path = ratings_example_gen.outputs['output'].get()[0].uri
    #self.assertTrue(fileio.exists(output_path))

    #following https://www.tensorflow.org/tfx/tutorials/mlmd/mlmd_tutorial#query_the_mlmd_database
    
    # Query all registered Artifact types.
    artifact_types = store.get_artifact_types()
    # All TFX artifacts are stored in the base directory
    base_dir = \
      connection_config.sqlite.filename_uri.split('metadata.sqlite')[0]
    def display_types(types):
      # Helper function to render dataframes for the artifact and execution types
      table = {'id': [], 'name': []}
      for a_type in types:
        table['id'].append(a_type.id)
        table['name'].append(a_type.name)
      return pd.DataFrame(data=table)

    def display_artifacts(store, artifacts):
      # Helper function to render dataframes for the input artifacts
      table = {'artifact id': [], 'type': [], 'uri': []}
      for a in artifacts:
        table['artifact id'].append(a.id)
        artifact_type = store.get_artifact_types_by_id([a.type_id])[0]
        table['type'].append(artifact_type.name)
        table['uri'].append(a.uri.replace(base_dir, './'))
      return pd.DataFrame(data=table)

    def display_properties(store, node):
      # Helper function to render dataframes for artifact and execution properties
      table = {'property': [], 'value': []}
      for k, v in node.properties.items():
        table['property'].append(k)
        table['value'].append(
            v.string_value if v.HasField('string_value') else v.int_value)
      for k, v in node.custom_properties.items():
        table['property'].append(k)
        table['value'].append(
            v.string_value if v.HasField('string_value') else v.int_value)
      return pd.DataFrame(data=table)

    display_types(store.get_artifact_types())
    
    execution_type = store.get_execution_type('IngestMovieLensComponent')
    # Get the latest execution of this type
    executions = store.get_executions_by_type(execution_type.name)
    ingestion_execution = executions[-1] # Assuming the latest run is the one to check
    
    # Assert the execution is complete
    assert ingestion_execution.last_known_state == metadata_store_pb2.Execution.State.COMPLETE


    """

  def testDo(self):
    #EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(IngestMovieLensExecutor)

    test_num = "fully_custom_comp_2"
    output_data_dir = os.path.join('/kaggle/working/bin/', test_num,
                                   self._testMethodName)
    os.makedirs(output_data_dir, exist_ok=True)

    output_examples = standard_artifacts.Examples()
    output_examples.uri = os.path.join(output_data_dir, 'output_examples')

    output_dict = {'output_examples':output_examples}
    exec_properties = {'name': 'IngestMovieLensExecutor',
      'infiles_dict_ser':serialize_to_string(self.infiles_dict),
      'output_config': self.output_config}
    ratings_example_gen = IngestMovieLensExecutor()
    ratings_example_gen.Do({}, output_dict, exec_properties)

    self.assertTrue(os.path.exists(output_examples.uri))
    self.assertTrue(len(os.listdir(output_examples.uri)) > 0)
