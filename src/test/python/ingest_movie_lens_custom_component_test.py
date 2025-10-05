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

from ingest_movie_lens_custom_component import *

from ml_metadata.proto import metadata_store_pb2

class IngestMovieLensTFXTest(tf.test.TestCase):

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

    self._assert_dict_content(ratings_dict)

    movies_dict = create_infile_dict(for_file='movies', \
                                     uri=movies_uri,
                                     col_names=movies_col_names, \
                                     col_types=movies_col_types,
                                     headers_present=False, delim="::")

    self._assert_dict_content(movies_dict)

    users_dict = create_infile_dict(for_file='users', \
                                    uri=users_uri,
                                    col_names=users_col_names, \
                                    col_types=users_col_types,
                                    headers_present=False, delim="::")

    self._assert_dict_content(users_dict)

    self.infiles_dict = create_infiles_dict(ratings_dict=ratings_dict, \
                                      movies_dict=movies_dict, \
                                      users_dict=users_dict)

    self.buckets = [80, 10, 10]
    self.bucket_names = ['train', 'eval', 'test']

    self.name = 'test run of ingest with tfx'

  @mock.patch.object(publisher, 'Publisher')
  def testRun(self, mock_publisher):

    infiles_dict_ser = pickle.dumps(infiles_dict)

    mock_publisher.return_value.publish_execution.return_value = {}

    ratings_example_gen = (IngestMovieLensComponent( \
      infiles_dict_ser=infiles_dict_ser, bucket_names=self.bucket_names, \
      buckets=self.buckets))

    output_data_dir = os.path.join('/kaggle/working/bin/', self._testMethodName)
    pipeline_root = os.path.join(output_data_dir, 'Test')
    fileio.makedirs(pipeline_root, exist_ok=True)

    pipeline_info = data_types.PipelineInfo(
        pipeline_name='Test', pipeline_root=pipeline_root, run_id='123')

    driver_args = data_types.DriverArgs(enable_cache=True)

    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.SetInParent()
    metadata_connection = metadata.Metadata(connection_config)

    launcher = in_process_component_launcher.InProcessComponentLauncher.create(
        component=ratings_example_gen,
        pipeline_info=pipeline_info,
        driver_args=driver_args,
        metadata_connection=metadata_connection,
        beam_pipeline_args=[],
        additional_pipeline_args={})

    self.assertEqual(
        launcher._component_info.component_type,
        name_utils.get_full_name(IngestMovieLensComponent))

    launcher.launch()
    mock_publisher.return_value.publish_execution.assert_called_once()

    # Check output paths.
    self.assertTrue(fileio.exists(os.path.join(pipeline_root, ratings_example_gen.id)))