
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
logging.set_verbosity(logging.WARNING)
logging.set_stderrthreshold(logging.WARNING)

class WriteParquetTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.infiles_dict_ser, self.output_config_ser, self.split_names = \
      get_test_data()
    self.n_users = 6040
    self.n_movies = 3952
    self.n_genres = N_GENRES
    self.n_age_groups = N_AGE_GROUPS
    self.n_occupations = 21
    self.name = 'test run of ratings transform'
    self.num_examples = 80_000
    self.MIN_EVAL_SIZE = 50
    
  def test_write_joined_tfrecords(self):
    
    for i, rating_file in enumerate([
      "ratings_train.dat", "ratings_val.dat", "ratings_test.dat", "ratings_train_liked.dat"]):
      
      PIPELINE_NAME = f'TFRecordTest{i}'
      # output_data_dir = os.path.join(os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',self.get_temp_dir()),self._testMethodName)
      output_data_dir = os.path.join(get_bin_dir(), PIPELINE_NAME)
      
      PIPELINE_ROOT = os.path.join(output_data_dir,  self._testMethodName)
      print(f"PIPELINE_ROOT={PIPELINE_ROOT}")
      METADATA_PATH = os.path.join(PIPELINE_ROOT, 'tfx_metadata', 'metadata.db')
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
      
      infiles_dict_ser, _, __ =  get_test_data(use_small=False)
      infiles_dict = deserialize(infiles_dict_ser)
      infiles_dict['ratings']['uri'] = os.path.join(get_project_dir(),
          "src/main/resources/ml-1m/", rating_file)
      infiles_dict_ser = serialize_to_string(infiles_dict)
      
      output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
          splits=[example_gen_pb2.SplitConfig.Split(
            name=f'train', hash_buckets=1)]
        )
      )
      output_config_ser = serialize_proto_to_string(output_config)
      output_parquet_path = os.path.join(PIPELINE_ROOT, "transformed_parquet")
      
      pipeline_factory = PipelineComponentsFactory(
        num_examples=self.num_examples,
        infiles_dict_ser=infiles_dict_ser,
        output_config_ser=output_config_ser,
        transform_dir=tr_dir, n_users=self.n_users,
        n_movies=self.n_movies,
        n_genres=self.n_genres, n_age_groups=self.n_age_groups,
        min_eval_size=self.MIN_EVAL_SIZE,
        batch_size=32, num_epochs=2, device="CPU",
        serving_model_dir=serving_model_dir,
        output_parquet_path=output_parquet_path
      )
      
      SETUP_FILE_PATH = os.path.join(get_project_dir(), 'setup.py')
    
      beam_pipeline_args = [
        '--direct_running_mode=multi_processing',
        '--direct_num_workers=0',
        f'--setup_file={SETUP_FILE_PATH}',
      ]
      
      baseline_components = pipeline_factory.build_components(PIPELINE_TYPE.PREPROCESSING)
      
      my_pipeline = tfx.dsl.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        components=baseline_components,
        enable_cache=ENABLE_CACHE,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args,
      )
      
      tfx.orchestration.LocalDagRunner().run(my_pipeline)
      logging.debug(fr"PREPOCESSING pipeline finished for {rating_file}")
    
