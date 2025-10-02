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

from unittest import mock
import tensorflow as tf
from ingest_movie_lens_tfx import IngestMovieLensComponent, IngestMovieLensExecutor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.io import fileio
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import publisher
from tfx.orchestration.launcher import in_process_component_launcher
from tfx.proto import example_gen_pb2
from tfx.utils import name_utils

from ml_metadata.proto import metadata_store_pb2

class IngestMovieLensComponentTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    kaggle = True
    if kaggle:
      prefix = '/kaggle/working/ml-1m/'
    else:
      prefix = "../resources/ml-1m/"
    self.ratings_uri = f"{prefix}ratings.dat"
    self.movies_uri = f"{prefix}movies.dat"
    self.users_uri = f"{prefix}users.dat"

    self.ratings_key_col_dict = {"user_id": 0, "movie_id": 1, "rating": 2, \
                            "timestamp": 3}
    self.movies_key_col_dict = {"movie_id": 0, "title": 1, "genres": 2}
    self.users_key_col_dict = {"user_id": 0, "gender": 1, "age": 2, \
                          "occupation": 3, "zipcode": 4}
    self.delim = "::"

    # these might need to be serialized into strings for tfx,
    # use json.dumps
    self.headers_present = False
    self.buckets = [80, 10, 10]
    self.bucket_names = ['train', 'eval', 'test']

    self.name = 'test run of ingest with tfx'

  @mock.patch.object(publisher, 'Publisher')
  def testRun(self, mock_publisher):

    mock_publisher.return_value.publish_execution.return_value = {}

    ratings_example_gen = IngestMovieLensComponent( \
      name=self.name, ratings_uri=self.ratings_uri, movies_uri=self.movies_uri, \
      users_uri=self.users_uri, headers_present=self.headers_present, \
      delim=self.delim, ratings_key_col_dict=self.ratings_key_col_dict, \
      users_key_col_dict=self.users_key_col_dict, \
      movies_key_col_dict=self.movies_key_col_dict, \
      bucket_names=self.bucket_names, buckets=self.buckets \
    )

    output_data_dir = os.path.join('/kaggle/working/bin/', self._testMethodName)
    pipeline_root = os.path.join(output_data_dir, 'Test')
    fileio.makedirs(pipeline_root)

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