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

import numpy as np

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

from ingest_movie_lens_component import *
from movie_lens_utils import *

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.metadata_store import metadata_store
import absl
from absl import logging
absl.logging.set_verbosity(absl.logging.DEBUG)

from helper import *

class IngestMovieLensComponentTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.infiles_dict_ser, self.output_config_ser, self.split_names = \
      get_test_data()

    self.name = 'test run of ratings ingestion w/ python custom comp func'

  def test_MovieLensExampleGen(self):

    test_num = "py_custom_comp_1"

    ratings_example_gen = (MovieLensExampleGen( \
      infiles_dict_ser=self.infiles_dict_ser, \
      output_config_ser = self.output_config_ser))

    logging.debug(f'TYPE of ratings_example_gen={type(ratings_example_gen)}')

    statistics_gen = StatisticsGen(examples = ratings_example_gen.outputs['output_examples'])

    schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=True)

    components = [ratings_example_gen, statistics_gen, schema_gen]

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
    logging.debug(f'alt_output_data_dir={alt_output_data_dir}')

    ENABLE_CACHE = False

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
    # PIPELINE_ROOT/MovieLensExampleGen/output_examples/1/
    # files are Split-train/data_*, etc

    ## ================= ratings_example_gen unit tests ============

    logging.debug(f'ratings_example_gen={ratings_example_gen}')
    #MovieLensExampleGen:
    logging.debug(f"ratings_example_gen.id={ratings_example_gen.id}")
    self.assertTrue(fileio.exists(os.path.join(PIPELINE_ROOT, ratings_example_gen.id)))

    for key, value in ratings_example_gen.outputs.items():
      logging.debug(f'key={key}\n  value={value}')

    logging.debug(f'listing files in PIPELINE_ROOT {PIPELINE_ROOT}:')
    for dirname, _, filenames in os.walk(PIPELINE_ROOT):
      for filename in filenames:
        logging.debug(os.path.join(dirname, filename))

    #https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/types/standard_artifacts.py#L487
    #Examples
    #ExampleStatistics
    #Schema

    #metadata_connection = metadata.Metadata(metadata_connection_config)
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
    # executions has custom_properties.key: "infiles_dict_ser"
    #    and custom_properties.key: "output_config_ser"
    artifact_count = len(artifacts)
    execution_count = len(executions)
    self.assertGreaterEqual(artifact_count, execution_count)
    self.assertGreaterEqual(artifact_count, execution_count)

    artifact_uri = artifacts[0].uri
    for split_name in self.split_names:
      dir_path = f'{artifact_uri}/{get_split_dir_name(split_name)}'
      file_paths = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
      #  file_paths = get_output_files(ratings_example_gen, 'output_examples', split_name)
      self.assertGreaterEqual(len(file_paths), 1)

      logging.debug(f"file_paths={file_paths}")
      col_name_feature_types = get_expected_col_name_feature_types2()

      dataset = tf.data.TFRecordDataset(file_paths, compression_type="GZIP")
      #dataset is TFRecordDatasetV2 element_spec=TensorSpec(shape=(), dtype=tf.string, name=None)
      logging.debug(f"dataset={dataset}")

      # user_id,movie_id,rating,gender,age,occupation,genres
      logging.debug(f"tf.executing_eagerly()={tf.executing_eagerly()}")

      #parse into dictionaries.
      #{'age': <tf.Tensor: shape=(), dtype=int64, numpy=50>,
      # 'gender': <tf.Tensor: shape=(), dtype=string, numpy=b'F'>,
      # 'genres': <tf.Tensor: shape=(), dtype=string,
      #   numpy=b"Animation|Children's|Comedy">,
      # 'movie_id': <tf.Tensor: shape=(), dtype=int64, numpy=1>,
      # 'occupation': <tf.Tensor: shape=(), dtype=int64, numpy=9>,
      # 'rating': <tf.Tensor: shape=(), dtype=int64, numpy=4>,
      # 'user_id': <tf.Tensor: shape=(), dtype=int64, numpy=6>}
      def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto,
          col_name_feature_types)
      try:
        parsed_dataset = dataset.map(_parse_function)
        for parsed_example in parsed_dataset.take(1):
          pass
      except Exception as e:
        self.fail(e)

      try:
        for tfrecord in dataset.take(1):
          example = tf.train.Example()
          example.ParseFromString(tfrecord.numpy())
          # print(f"EXAMPLE={example}")
      except Exception as e:
        self.fail(e)

    #=============== verify statistics_gen results ==============

    logging.debug(f"statistics_gen.id={statistics_gen.id}") #StatisticsGen
    logging.debug(f"statistics_gen={statistics_gen}")
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
    #logging.debug(f"schema_gen={schema_gen}")
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
    # user_id,movie_id,rating,gender,age,occupation,genres
    # int,    int      int     str    int  int       str
    col_name_feature_types = get_expected_col_name_feature_types()
    for feature in schema.feature:
      self.assertTrue(feature.name in col_name_feature_types)
      expected_type = col_name_feature_types[feature.name].pop()
      match feature.type:
        case schema_pb2.INT:
          self.assertTrue(expected_type == tf.train.Int64List)
        case schema_pb2.FLOAT:
          self.assertTrue(expected_type == tf.train.FloatList)
        case schema_pb2.BYTES:
          self.assertTrue(expected_type == tf.train.BytesList)
        case _:
          self.fail(f"unexpected feature type in feature={feature}")