
import shutil

from tfx.dsl.io import fileio
from tfx.orchestration import metadata
from tfx.components import StatisticsGen, SchemaGen, ExampleValidator
from tfx.utils import io_utils
from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_transform.tf_metadata import schema_utils

from movie_lens_tfx.PipelineComponentsFactory import *
from movie_lens_tfx.ingest_pyfunc_component.ingest_movie_lens_component import *
#import trainer_movie_lens

from ml_metadata.metadata_store import metadata_store
from movie_lens_tfx.tune_train_movie_lens import *

from helper import *

tf.get_logger().propagate = False
from absl import logging
logging.set_verbosity(logging.INFO)
logging.set_stderrthreshold(logging.INFO)
import warnings
# Suppress DeprecationWarnings from 3rd-party libraries
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*parse_example_dataset.*")

"""
NOTE: to run this method,  edit in class TwoTowerDNN the attribute to:
          self.calc_table_B_diagnostic = True

stdout from this method for 1 epoch:

Total unique items in table_B: 3416
============================================================
Min value:  1.15
Max value:  16.55
Mean value: 10.04
------------------------------------------------------------
20th percentile:     7.92
50th percentile:     9.90
75th percentile:    11.81
80th percentile:    12.21
85th percentile:    12.63
90th percentile:    13.20
95th percentile:    13.95
99th percentile:    15.37
------------------------------------------------------------
Recommendation: For a 20% head cutoff, set head_threshold = 7.92
Recommendation: For an 80% tail cutoff, set b_threshold = 12.21
============================================================

ASCII Histogram (B_new Distribution):
Range of B_new           | Count   | Distribution
------------------------------------------------------------
[    1.1 -     2.7) |       7 |
[    2.7 -     4.2) |       7 |
[    4.2 -     5.8) |      66 | ███
[    5.8 -     7.3) |     342 | ███████████████
[    7.3 -     8.9) |     738 | █████████████████████████████████
[    8.9 -    10.4) |     770 | ███████████████████████████████████
[   10.4 -    11.9) |     701 | ███████████████████████████████
[   11.9 -    13.5) |     517 | ███████████████████████
[   13.5 -    15.0) |     216 | █████████
[   15.0 -    16.6) |      52 | ██
============================================================
"""
class TableBTest(tf.test.TestCase):

  def setUp(self):
      super().setUp()
      self.n_users = 6040
      self.n_movies = 3952
      self.n_genres = N_GENRES
      self.n_occupations = 21
      self.num_examples = 80_000
      self.MIN_EVAL_SIZE = 50
      self.name = 'test run for table_b'
      self.BATCH_SIZE = 2056
    
  def test_1_epoch_table(self):
      test_num = "1"
      
      PIPELINE_NAME = 'Test_table_b'
      # output_data_dir = os.path.join(os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',self.get_temp_dir()),self._testMethodName)
      output_data_dir = os.path.join(get_bin_dir(), PIPELINE_NAME, test_num)
      PIPELINE_ROOT = output_data_dir
      METADATA_PATH = os.path.join(PIPELINE_ROOT, 'tfx_metadata', 'metadata.db')
      
      # remove results from previous test runs:
      try:
          logging.debug(f"removing: {PIPELINE_ROOT}")
          shutil.rmtree(PIPELINE_ROOT)
      except OSError as e:
          pass
      
      os.makedirs(os.path.join(PIPELINE_ROOT, 'tfx_metadata'), exist_ok=True)
      
      ENABLE_CACHE = True
      
      # metadata_connection_config = metadata_store_pb2.ConnectionConfig()
      # metadata_connection_config.sqlite.SetInParent()
      # metadata_connection = metadata.Metadata(metadata_connection_config)
      metadata_connection_config = metadata.sqlite_metadata_connection_config(
          METADATA_PATH)
      
      store = metadata_store.MetadataStore(metadata_connection_config)
      
      tr_dir = os.path.join(get_project_dir(), "src/main/python/movie_lens_tfx")
      
      serving_model_dir = os.path.join(PIPELINE_ROOT, 'serving_model')
      query_model_dir = os.path.join(PIPELINE_ROOT, 'serving_query_model')
      candidate_model_dir = os.path.join(PIPELINE_ROOT, 'serving_candidate_model')
      os.makedirs(serving_model_dir, exist_ok=True)
      os.makedirs(query_model_dir, exist_ok=True)
      os.makedirs(candidate_model_dir, exist_ok=True)
      
      infiles_dict_of_dicts_ser = get_contrastive_split_infiles_set(ds=DataSize.FULL)
      num_examples = 800167
      
      pipeline_factory = PipelineComponentsFactory(
          num_examples=num_examples,
          infiles_dict_ser=infiles_dict_of_dicts_ser,
          output_config_ser=None,
          transform_dir=tr_dir, n_users=self.n_users,
          n_movies=self.n_movies,
          n_genres=self.n_genres,
          min_eval_size=self.MIN_EVAL_SIZE,
          batch_size=self.BATCH_SIZE, num_epochs=1,
          serving_model_dir=serving_model_dir)
      
      SETUP_FILE_PATH = os.path.join(get_project_dir(), 'setup.py')
      
      beam_pipeline_args = [
          '--direct_running_mode=multi_processing',
          '--direct_num_workers=0',
          f'--setup_file={SETUP_FILE_PATH}',
      ]
      
      components = pipeline_factory.build_components(PIPELINE_TYPE.TABLE_B_DIAGNOSTIC)
      
      my_pipeline = tfx.dsl.Pipeline(
          pipeline_name=PIPELINE_NAME,
          pipeline_root=PIPELINE_ROOT,
          components=components,
          enable_cache=ENABLE_CACHE,
          metadata_connection_config=metadata_connection_config,
          beam_pipeline_args=beam_pipeline_args,
      )
      
      tfx.orchestration.LocalDagRunner().run(my_pipeline)
      logging.debug("pipeline finished")
      
    
   
    
      