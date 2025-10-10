# following from unit tests at:
# https://github.com/tensorflow/tfx/
# which have Google LLC. copyrights under Apache License, Version 2.0
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
import shutil

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
from tfx.components import StatisticsGen

from ingest_movie_lens_component import *
from movie_lens_utils import *

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.metadata_store import metadata_store
import absl
from absl import logging
absl.logging.set_verbosity(absl.logging.DEBUG)

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

    ratings_col_names = ["user_id", "movie_id", "rating", "timestamp"]
    ratings_col_types = [int, int,int,int]  # for some files, ratings are floats
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
    self.split_names = ['train', 'eval', 'test']
    output_config = example_gen_pb2.Output(
      split_config=example_gen_pb2.SplitConfig(
        splits=[
          example_gen_pb2.SplitConfig.Split(name=n, hash_buckets=b) \
          for n, b in zip(self.split_names, buckets)]
      )
    )
    logging.debug(f"test output_config={output_config}")
    self.output_config_ser = serialize_proto_to_string(output_config)

    self.name = 'test run of ratings ingestion w/ python custom comp func'

  def test_ingest_movie_lens_component(self):

    test_num = "py_custom_comp_1"

    infiles_dict_ser = serialize_to_string(self.infiles_dict)

    ratings_example_gen = (ingest_movie_lens_component( \
      infiles_dict_ser=infiles_dict_ser, \
      output_config_ser = self.output_config_ser))

    logging.debug(f'TYPE of ratings_example_gen={type(ratings_example_gen)}')

    components = [ratings_example_gen]

    PIPELINE_NAME = 'TestPythonFuncCustomCompPipeline'
    #output_data_dir = os.path.join(os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',self.get_temp_dir()),self._testMethodName)
    output_data_dir = os.path.join('/kaggle/working/bin/', test_num, self._testMethodName)
    PIPELINE_ROOT = os.path.join(output_data_dir, PIPELINE_NAME)
    #remove results from previous test runs:
    try:
      shutil.rmtree(PIPELINE_ROOT)
    except OSError as e:
      pass
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

    my_pipeline = tfx.dsl.Pipeline(
      pipeline_name=PIPELINE_NAME,
      pipeline_root=PIPELINE_ROOT,
      components=components,
      enable_cache=ENABLE_CACHE,
      metadata_connection_config=metadata_connection_config,
      #beam_pipeline_args=beam_pipeline_args,
    )

    tfx.orchestration.LocalDagRunner().run(my_pipeline)

    # creates output_examples.uri=
    # PIPELINE_ROOT/ingest_movie_lens_component/output_examples/1/
    # files are Split-train/data_*, etc

    # Check output paths.
    logging.debug(f'ratings_example_gen={ratings_example_gen}')
    logging.debug(f"ratings_example_gen.id={ratings_example_gen.id}")
    self.assertTrue(fileio.exists(os.path.join(PIPELINE_ROOT, ratings_example_gen.id)))

    for key, value in ratings_example_gen.outputs.items():
      print(f'key={key}\n  value={value}')

    print(f'listing files in PIPELINE_ROOT {PIPELINE_ROOT}:')
    for dirname, _, filenames in os.walk(PIPELINE_ROOT):
      for filename in filenames:
        print(os.path.join(dirname, filename))

    #metadata_connection = metadata.Metadata(metadata_connection_config)
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
    # executions has custom_properties.key: "infiles_dict_ser"
    #    and custom_properties.key: "output_config_ser"
    artifact_count = len(artifacts)
    execution_count = len(executions)
    self.assertGreaterEqual(artifact_count, execution_count)
    self.assertGreaterEqual(artifact_count, execution_count)

    artifact_uri = artifacts[0].uri
    for split_name in self.split_names:
      dir_path = f'{artifact_uri}/{get_split_dir_name(split_name)}'
      file_paths = [os.path.join(dir_path, name) for name in
                    os.listdir(dir_path)]
      #  file_paths = get_output_files(ratings_example_gen, 'output_examples', split_name)
      self.assertGreaterEqual(len(file_paths), 1)
      
    """
    def get_latest_artifact_path_statistics(metadata_store, pipeline_name, component_name):
      # Find the component by name
      component_artifacts = metadata_store.get_artifacts_by_type_and_name(
          type_name='Statistics', name=f'{pipeline_name}.{component_name}.statistics'
      )
      if not component_artifacts:
        print(f"No artifacts found for component '{component_name}'.")
        return None
      # Sort artifacts by creation time to get the latest one
      latest_artifact = sorted(component_artifacts, \
        key=lambda a: a.create_time_since_epoch, reverse=True)[0]

      # Get the URI and find the 'statistics.pb' file
      artifact_uri = latest_artifact.uri
      stats_path = os.path.join(artifact_uri, 'Split-train', 'stats_tfrecord')

      return stats_path
    """
    #stats_file_path = get_latest_artifact_path_statistics(store, PIPELINE_NAME, 'StatisticsGen')
    #if stats_file_path:
    #    stats_proto = tfdv.load_statistics(stats_file_path)
    #    print("Successfully loaded statistics. Here is some example output:")
    #    for dataset in stats_proto.datasets:
    #        print(f"Statistics for dataset: {dataset.name}")
    #        for feature in dataset.features:
    #            print(f"  Feature: {feature.path.step[0]}, Type: {feature.type}")
    #            if feature.HasField('num_stats'):
    #                print(f"    Min: {feature.num_stats.min}, Max: {feature.num_stats.max}, Mean: {feature.num_stats.mean}")

    # TODO: change to use pipeline path, and do another assert with MLMD info
    # for split_name in self.split_names:
    #  file_list = get_output_files(ratings_example_gen, 'output_examples', split_name)
    #  self.assertGreaterEqual(len(file_list), 1)

    #self.assertIsNotNone(ratings_example_gen.outputs['output_examples'].get()[0])
    #output_path = ratings_example_gen.outputs['output'].get()[0].uri
    #self.assertTrue(fileio.exists(output_path))