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
import random

import tensorflow as tf
import tensorflow_data_validation as tfdv
from tfx.dsl.components.base import executor_spec
from tfx.dsl.io import fileio
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration.launcher import in_process_component_launcher
from tfx.proto import example_gen_pb2
from tfx.components import StatisticsGen, SchemaGen
from tfx.types import standard_component_specs
from tfx.utils import proto_utils

from helper import *

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
    self.infiles_dict_ser, self.output_config_ser, self.split_names = \
      get_test_data()

    self.name = 'test run of ratings ingestion w/ fully custom comp func'

  def testRun2(self):

    test_num = "fully_custom_comp_1"
    name = "test_fully_custom_component"

    #implement the task
    ratings_example_gen = IngestMovieLensComponent( \
      name=name,\
      infiles_dict_ser=self.infiles_dict_ser, \
      output_config_ser=self.output_config_ser)

    statistics_gen = StatisticsGen(examples = ratings_example_gen.outputs['output_examples'])

    schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=True)

    components = [ratings_example_gen, statistics_gen, schema_gen]

    PIPELINE_NAME = 'TestFullyCustomCompPipeline'
    #output_data_dir = os.path.join(os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',self.get_temp_dir()),self._testMethodName)
    output_data_dir = os.path.join('/kaggle/working/bin/', test_num, self._testMethodName)
    PIPELINE_ROOT = os.path.join(output_data_dir, PIPELINE_NAME)
    # remove results from previous test runs:
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

    #LocalDagRunner is trying to pickle arguments of the components,
    #  so output_config

    tfx.orchestration.LocalDagRunner().run(my_pipeline)

    # Check output paths.
    self.assertTrue(fileio.exists(os.path.join(PIPELINE_ROOT, \
      ratings_example_gen.id)))

    logging.debug(f'TYPE of ratings_example_gen={type(ratings_example_gen)}')
    logging.debug(f'ratings_example_gen={ratings_example_gen}')
    logging.debug(f'ratings_example_gen.outputs["output_examples"]='
                  f'{ratings_example_gen.outputs["output_examples"]}')

    for key, value in ratings_example_gen.outputs.items():
      print(f'key={key}, value={value}')

    #editing
    # creates output_examples.uri=
    # PIPELINE_ROOT/IngestMovieLensComponent/output_examples/1/
    # files are Split-train/data_*, etc

    #list files in alt_output_data_dir and in output_data_dir
    print(f'listing files in output_data_dir {PIPELINE_ROOT}:')
    for dirname, _, filenames in os.walk(PIPELINE_ROOT):
      for filename in filenames:
        print(os.path.join(dirname, filename))

    # metadata_connection = metadata.Metadata(metadata_connection_config)
    store = metadata_store.MetadataStore(metadata_connection_config)
    artifact_types = store.get_artifact_types()
    logging.debug(f"MLMD store artifact_types={artifact_types}")
    artifacts = store.get_artifacts()
    logging.debug(f"MLMD store artifacts={artifacts}")
    executions = store.get_executions()
    logging.debug(f"MLMD store executions={executions}")
    self.assertEqual(3, len(artifact_types))
    self.assertEqual(3, len(artifacts))
    self.assertEqual(3, len(executions))
    #executions has custom_properties.key: "infiles_dict_ser"
    #    and custom_properties.key: "output_config_ser"
    artifact_count = len(artifacts)
    execution_count = len(executions)
    self.assertGreaterEqual(artifact_count, execution_count)

    artifact_uri = artifacts[0].uri
    for split_name in self.split_names:
      dir_path = f'{artifact_uri}/{get_split_dir_name(split_name)}'
      file_paths = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
      #  file_paths = get_output_files(ratings_example_gen, 'output_examples', split_name)
      self.assertGreaterEqual(len(file_paths), 1)

    #=============== verify statistics_gen results ==============

    logging.debug(f"statistics_gen.id={statistics_gen.id}") #StatisticsGen
    #logging.debug(f"statistics_gen={statistics_gen}")
    self.assertTrue(fileio.exists(os.path.join(PIPELINE_ROOT, statistics_gen.id)))

    stats_artifacts_list = store.get_artifacts_by_type("ExampleStatistics")
    logging.debug(f"stats_artifacts_list={stats_artifacts_list}")
    latest_stats_artifact = sorted(stats_artifacts_list, \
      key=lambda x: x.create_time_since_epoch, reverse=True)[0]
    #or use last_update_time_since_epoch
    stats_uri = latest_stats_artifact.uri
    logging.debug(f"loading stats_uri={stats_uri}")
    stats_path_train = os.path.join(stats_uri, get_split_dir_name("train"), 'FeatureStats.pb')
    stats_path_eval = os.path.join(stats_uri, get_split_dir_name("eval"), 'FeatureStats.pb')
    stats_path_test = os.path.join(stats_uri, get_split_dir_name("test"),'FeatureStats.pb')
    self.assertTrue(os.path.exists(stats_path_train))
    self.assertTrue(os.path.exists(stats_path_eval))
    self.assertTrue(os.path.exists(stats_path_test))

    logging.debug(f"loading stats_path_train={stats_path_train}")
    #statistics_pb2.DatasetFeatureStatisticsList
    stats_proto_train = tfdv.load_stats_binary(stats_path_train)
    self.assertIsNotNone(stats_proto_train)
    #logging.debug(f'stats_proto_train={str(stats_proto_train)}')
    #print("Successfully loaded statistics. Here is some example output:")
    #self.assertLess(stats_proto_train.datasets.num_examples, 1000)

    # =============== verify schema_gen results ==============
    logging.debug(f"schema_gen.id={schema_gen.id}")
    logging.debug(f"schema_gen={schema_gen}")
    self.assertTrue(fileio.exists(os.path.join(PIPELINE_ROOT, schema_gen.id)))

    schema_artifacts_list = store.get_artifacts_by_type("Schema")
    logging.debug(f"schema_artifacts_list={schema_artifacts_list}")
    latest_schema_artifact = sorted(schema_artifacts_list, \
      key=lambda x: x.create_time_since_epoch, reverse=True)[0]
    # or use last_update_time_since_epoch
    schema_uri = latest_schema_artifact.uri
    logging.debug(f'schema_uri={schema_uri}')
    schema_path_train = os.path.join(schema_uri, 'schema.pbtxt')
    #schema_pb2.Schema
    schema = tfdv.load_schema_text(schema_path_train)
    self.assertIsNotNone(schema)
    logging.debug(f"schema={schema}")
    #TODO: consider asserting the schema:
    # user_id,movie_id,rating,gender,age,occupation,zipcode,genres
    # int,    int      int     str    int  int       str     str

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
      'infiles_dict_ser': self.infiles_dict_ser,
      'output_config_ser': self.output_config_ser}
    ratings_example_gen = IngestMovieLensExecutor()
    ratings_example_gen.Do({}, output_dict, exec_properties)

    self.assertTrue(os.path.exists(output_examples.uri))
    self.assertTrue(len(os.listdir(output_examples.uri)) > 0)
