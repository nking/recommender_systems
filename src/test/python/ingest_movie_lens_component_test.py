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
import pprint
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

from ingest_movie_lens_component import *
from movie_lens_utils import *

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.metadata_store import metadata_store

class IngestMovieLensComponentTest(tf.test.TestCase):

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

    ratings_col_names = ["user_id", "movie_id", "rating"]
    ratings_col_types = [int, int,
                         int]  # for some files, ratings are floats
    movies_col_names = ["movie_id", "title", "genres"]
    movies_col_types = [int, str, str]
    users_col_names = ["user_id", "gender", "age", "occupation",
                       "zipcode"]
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

    self.buckets = [80, 10, 10]
    self.bucket_names = ['train', 'eval', 'test']
    self.buckets_ser = (base64.b64encode(pickle.dumps(self.buckets))).decode('utf-8')
    self.bucket_names_ser = (base64.b64encode(pickle.dumps(self.bucket_names))).decode('utf-8')

    self.name = 'test run of ingest with tfx'

  @mock.patch.object(publisher, 'Publisher')
  def testRun(self, mock_publisher):

    test_num = "1"

    infiles_dict_ser = (base64.b64encode(pickle.dumps(self.infiles_dict))).decode('utf-8')

    mock_publisher.return_value.publish_execution.return_value = {}

    ratings_example_gen = (ingest_movie_lens_component( \
      infiles_dict_ser=infiles_dict_ser, \
      bucket_names_ser=self.bucket_names_ser, \
      buckets_ser=self.buckets_ser))

    #output_data_dir = os.path.join(os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',self.get_temp_dir()),self._testMethodName)
    output_data_dir = os.path.join('/kaggle/working/bin/', test_num, self._testMethodName)
    pipeline_root = os.path.join(output_data_dir, 'TestPythonFuncCustomCompPipeline')
    os.makedirs(pipeline_root, exist_ok=True)

    alt_output_data_dir = os.path.join(
      os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',
                     self.get_temp_dir()), self._testMethodName)
    print(f'alt_output_data_dir={alt_output_data_dir}')

    pipeline_info = data_types.PipelineInfo(
        pipeline_name='TestPythonFuncCustomCompPipeline', pipeline_root=pipeline_root, run_id=test_num)

    driver_args = data_types.DriverArgs(enable_cache=False)

    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.SetInParent()
    metadata_connection = metadata.Metadata(connection_config)
    store = metadata_store.MetadataStore(connection_config)

    launcher = in_process_component_launcher.InProcessComponentLauncher.create(
        component=ratings_example_gen,
        pipeline_info=pipeline_info,
        driver_args=driver_args,
        metadata_connection=metadata_connection,
        beam_pipeline_args=[],
        additional_pipeline_args={})

    self.assertEqual(
        launcher._component_info.component_type,
        name_utils.get_full_name(ingest_movie_lens_component))

    launcher.launch()
    mock_publisher.return_value.publish_execution.assert_called_once()

    # Check output paths.
    self.assertTrue(fileio.exists(os.path.join(pipeline_root, ratings_example_gen.id)))

    for key, value in ratings_example_gen.outputs.items():
      print(f'key={key}, value={value}')

    #list files in alt_output_data_dir and in output_data_dir
    print(f'listing files in output_data_dir {output_data_dir}:')
    for dirname, _, filenames in os.walk(output_data_dir):
      for filename in filenames:
        print(os.path.join(dirname, filename))

    artifact_types = store.get_artifact_types()
    logging.debug(f"MLMD store artifact_types={artifact_types}")
    artifacts = store.get_artifacts()
    logging.debug(f"MLMD store artifacts={artifacts}")
    executions = store.get_executions()
    logging.debug(f"MLMD store executions={executions}")
    for execution in executions:
      logging.debug(f"execution.properties={execution.properties}")
    artifact_count = len(artifacts)
    execution_count = len(executions)
    self.assertGreaterEqual(artifact_count, execution_count)

    #self.assertIsNotNone(ratings_example_gen.outputs['output'].get()[0])
    #output_path = ratings_example_gen.outputs['output'].get()[0].uri
    #self.assertTrue(fileio.exists(output_path))