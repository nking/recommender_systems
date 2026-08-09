import os
import shutil

from tfx.orchestration import metadata

from ml_metadata.proto import metadata_store_pb2
from ml_metadata.metadata_store import metadata_store
from tensorflow_transform.tf_metadata import schema_utils

from tfx import v1 as tfx

import tensorflow as tf
from absl import logging
tf.get_logger().propagate = False
logging.set_verbosity(logging.WARNING)
logging.set_stderrthreshold(logging.WARNING)
from helper import *

from movie_lens_tfx.PipelineComponentsFactory import *

tf.get_logger().propagate = False
from absl import logging

@component()
def GPUAvailCheck() -> None:
    print("begin avail check")
    if tf.config.list_physical_devices('TPU'):
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
                tpu='local')
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
            print("Hardware auto-detected: TPU")
        except Exception as ex:
            print(f"ERROR: TPU detected but failed to initialize: {ex}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # MirroredStrategy handles both single-GPU and multi-GPU configurations automatically
            strategy = tf.distribute.MirroredStrategy()
            print(f"Hardware auto-detected: {len(gpus)} GPU(s)")
        except Exception as ex:
            print(f"ERROR: GPU detected but strategy failed: {ex}")
    # NOTE a multihost strategy should use  tf.distribute.MultiWorkerMirroredStrategy
    # Fallback to default CPU strategy
    strategy = tf.distribute.get_strategy()
    print("Hardware auto-detected: CPU fallback")

class GPUsAvailableTest(tf.test.TestCase):
    
    def setUp(self):
        super().setUp()
    
    def test_1(self):
        print(f'run GPU availability check')
        
        test_num = "1"
        PIPELINE_NAME = 'Test_GPU_AVAIL'
        # output_data_dir = os.path.join(os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',self.get_temp_dir()),self._testMethodName)
        output_data_dir = os.path.join(get_bin_dir(), PIPELINE_NAME, test_num)
        PIPELINE_ROOT = output_data_dir
        METADATA_PATH = os.path.join(PIPELINE_ROOT, 'tfx_metadata', 'metadata.db')
        # remove results from previous test runs:
        try:
            logging.debug(f"removing: {PIPELINE_ROOT}")
            shutil.rmtree(os.path.join(get_bin_dir(), PIPELINE_NAME))
        except OSError as e:
            pass
        
        os.makedirs(os.path.join(PIPELINE_ROOT, 'tfx_metadata'), exist_ok=True)
        
        ENABLE_CACHE = True
        
        # metadata_connection_config = metadata_store_pb2.ConnectionConfig()
        # metadata_connection_config.sqlite.SetInParent()
        # metadata_connection = metadata.Metadata(metadata_connection_config)
        metadata_connection_config = metadata.sqlite_metadata_connection_config(METADATA_PATH)
        
        SETUP_FILE_PATH = os.path.join(get_project_dir(), 'setup.py')
        
        beam_pipeline_args = [
            '--direct_running_mode=multi_processing',
            '--direct_num_workers=0',
            f'--setup_file={SETUP_FILE_PATH}',
        ]
        
        baseline_components = [GPUAvailCheck()]
        
        # create baseline model
        my_pipeline = tfx.dsl.Pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            components=baseline_components,
            enable_cache=ENABLE_CACHE,
            metadata_connection_config=metadata_connection_config,
            beam_pipeline_args=beam_pipeline_args,
        )
        
        tfx.orchestration.LocalDagRunner().run(my_pipeline)
        logging.debug("GPU AVAIL pipeline finished")
        
