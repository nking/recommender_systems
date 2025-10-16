import os
import shutil

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

from ingest_movie_lens_custom_component import *
from movie_lens_utils import *

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.metadata_store import metadata_store

logging.set_verbosity(logging.WARNING)
logging.set_stderrthreshold(logging.WARNING)

class CSVExampleGenTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.name = 'run with CSVExampleGen to examine automatic MLMD data'

  @mock.patch.object(publisher, 'Publisher')
  def testRun(self, mock_publisher):

    test_num = "csv_comp_1"
    kaggle = True
    if kaggle:
      prefix = '/kaggle/working/ml-1m/'
      prefix2 = '/kaggle/working/ml-1m/tmp/'
    else:
      prefix = "../resources/ml-1m/"
      prefix2 = '/kaggle/working/ml-1m/tmp'
    os.makedirs(prefix2, exist_ok=True)
    users_uri = f"{prefix2}users2.dat"

    #a work around to use CSVExampleGen.  the workaround isn't scalable.
    #users.dat delimeter is ::
    #the moview.dat title field has commas
     #replace delimiters in users2.dat '_'
    _fln = f"{prefix}users.dat"
    command = "LC_ALL=UTF8 sed 's/::/,/g' " + _fln + " > " + users_uri
    os.system(command)
    self.assertTrue(os.path.exists(users_uri))

    mock_publisher.return_value.publish_execution.return_value = {}

    name = "test_csvgenexample"

    output_data_dir = os.path.join('/kaggle/working/bin/', test_num, \
      self._testMethodName)
    pipeline_root = os.path.join(output_data_dir, name)
    os.makedirs(pipeline_root, exist_ok=True)

    pipeline_info = data_types.PipelineInfo(
      pipeline_name=name, pipeline_root=pipeline_root, run_id=test_num)

    driver_args = data_types.DriverArgs(enable_cache=True)

    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.SetInParent() #uses in-memory database
    metadata_connection = metadata.Metadata(connection_config)

    output_config = example_gen_pb2.Output(
      split_config=example_gen_pb2.SplitConfig(
        splits=[
          example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
          example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1),
          example_gen_pb2.SplitConfig.Split(name='test', hash_buckets=1)
        ]))

    users_example_gen = tfx.components.CsvExampleGen(\
      input_base=prefix2,\
      output_config = output_config)

    #statistics_gen = StatisticsGen(examples=users_example_gen.outputs['examples'])

    launcher = in_process_component_launcher.InProcessComponentLauncher.create(
        component=users_example_gen,
        pipeline_info=pipeline_info,
        driver_args=driver_args,
        metadata_connection=metadata_connection,
        beam_pipeline_args=[],
        additional_pipeline_args={})

    self.assertEqual(
        launcher._component_info.component_type,
        name_utils.get_full_name(tfx.components.CsvExampleGen))

    launcher.launch()
    mock_publisher.return_value.publish_execution.assert_called_once()

    # Check output paths.
    self.assertTrue(fileio.exists(os.path.join(pipeline_root, users_example_gen.id)))

    for key, value in users_example_gen.outputs.items():
      print(f'key={key}, value={value}')

    #list files in alt_output_data_dir and in output_data_dir
    print(f'listing files in output_data_dir {output_data_dir}:')
    for dirname, _, filenames in os.walk(output_data_dir):
      for filename in filenames:
        print(os.path.join(dirname, filename))

    store = metadata_store.MetadataStore(connection_config)
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
    logging.debug(f'contexts={store.get_contexts()}')

    try:
      logging.debug(f'users_example_gen={users_example_gen}')
      get_output_files(users_example_gen, 'examples', 'train')
    except Exception as ex:
      pass

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

  #def testDo(self):
  #  #EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(IngestMovieLensExecutor)
  #  #ratings_example_gen = IngestMovieLensExecutor()
  #  #ratings_example_gen.Do({}, output_dict, exec_properties)
  #  pass