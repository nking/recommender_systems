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
from transform_movie_lens import *

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

    self.name = 'test run of ratings transform'

  def test_MovieLensExampleGen(self):

    test_num = "transform_1"

    ratings_example_gen = (MovieLensExampleGen( \
      infiles_dict_ser=self.infiles_dict_ser, \
      output_config_ser = self.output_config_ser))

    statistics_gen = StatisticsGen(examples = ratings_example_gen.outputs['output_examples'])

    schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=True)

    ratings_transform = tfx.components.Transform(
      examples=ratings_example_gen.outputs['output_examples'],
      schema=schema_gen.outputs['schema'],
      module_file=os.path.abspath('transform_movie_lens.py'))

    components = [ratings_example_gen, statistics_gen, schema_gen, ratings_transform]

    PIPELINE_NAME = 'TestPythonTransformPipeline'
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

    ENABLE_CACHE = False

    #metadata_connection_config = metadata_store_pb2.ConnectionConfig()
    #metadata_connection_config.sqlite.SetInParent()
    #metadata_connection = metadata.Metadata(metadata_connection_config)
    metadata_connection_config = metadata.sqlite_metadata_connection_config(METADATA_PATH)

    beam_pipeline_args = [
      '--direct_running_mode=multi_processing',
      '--direct_num_workers=0'
    ]

    #imple is tfx.v1.dsl.Pipeline  where tfx is aliased in import
    my_pipeline = tfx.dsl.Pipeline(
      pipeline_name=PIPELINE_NAME,
      pipeline_root=PIPELINE_ROOT,
      components=components,
      enable_cache=ENABLE_CACHE,
      metadata_connection_config=metadata_connection_config,
      beam_pipeline_args=beam_pipeline_args,
    )

    tfx.orchestration.LocalDagRunner().run(my_pipeline)

    #metadata_connection = metadata.Metadata(metadata_connection_config)
    store = metadata_store.MetadataStore(metadata_connection_config)
    artifact_types = store.get_artifact_types()
    logging.debug(f"MLMD store artifact_types={artifact_types}")
    artifacts = store.get_artifacts()
    logging.debug(f"MLMD store artifacts={artifacts}")
    executions = store.get_executions()
    logging.debug(f"MLMD store executions={executions}")
    self.assertEqual(4, len(artifact_types))
    self.assertEqual(4, len(artifacts))
    self.assertEqual(4, len(executions))
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

    logging.debug(f"ratings_transform.id={ratings_transform.id}") #StatisticsGen
    logging.debug(f"ratings_transform={ratings_transform}")
    self.assertTrue(fileio.exists(os.path.join(PIPELINE_ROOT, ratings_transform.id)))

    #https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/types/standard_artifacts.py#L487
    #TransformCache
    #TransformGraph
    transform_graph_list = store.get_artifacts_by_type("TransformGraph")
    logging.debug(f"transform_graph_list={transform_graph_list}")
    latest_transform_graph_artifact = sorted(transform_graph_list, \
      key=lambda x: x.create_time_since_epoch, reverse=True)[0]
    #or use last_update_time_since_epoch
    transform_graph_uri = latest_transform_graph_artifact.uri
    logging.debug(f"loading transform_graph_uri={transform_graph_uri}")

    #component outpurs contains: https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Transform#example
    #transform_graph: Channel of type standard_artifacts.TransformGraph,
    #   includes an exported Tensorflow graph suitable for both training and serving.
    #transformed_examples: Channel of type standard_artifacts.Examples
    #   for materialized transformed examples, which includes transform splits as specified in splits_config
    logging.debug(f"component transformed_examples uri={ratings_transform.outputs['transformed_examples'].get()[0].uri}")
    logging.debug(f"component post_transform_schema uri={ratings_transform.outputs['post_transform_schema'].get()[0].uri}")
    stats_path_train = os.path.join(transform_graph_uri, get_split_dir_name("train"), 'FeatureStats.pb')
    stats_path_eval = os.path.join(transform_graph_uri, get_split_dir_name("eval"), 'FeatureStats.pb')
    stats_path_test = os.path.join(transform_graph_uri, get_split_dir_name("test"),'FeatureStats.pb')
    self.assertTrue(os.path.exists(stats_path_train))
    self.assertTrue(os.path.exists(stats_path_eval))
    self.assertTrue(os.path.exists(stats_path_test))

    tfrecord_filenames = [os.path.join(stats_path_train, name) for name in os.listdir(stats_path_train)]
    dataset = tf.data.TFRecordDataset(tfrecord_filenames)
    for tfrecord in dataset.take(5):
      example = tf.train.Example()
      example.ParseFromString(tfrecord.numpy())
      logging.debug(f"a transform example={example}")

