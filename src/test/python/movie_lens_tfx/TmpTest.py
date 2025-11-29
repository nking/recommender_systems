
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

class TuneTrainTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.infiles_dict_ser, self.output_config_ser, self.split_names = \
      get_test_data()
    self.user_id_max = 6040
    self.movie_id_max = 3952
    self.n_genres = N_GENRES
    self.n_age_groups = N_AGE_GROUPS
    self.n_occupations = 21
    self.name = 'test run of ratings transform'
    self.num_examples = 80_000
    self.MIN_EVAL_SIZE = 50

  def test_signature(self):
    
    run_pipeline = False
    
    PIPELINE_NAME = 'TMPTest'
    # output_data_dir = os.path.join(os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',self.get_temp_dir()),self._testMethodName)
    output_data_dir = os.path.join(get_bin_dir(), PIPELINE_NAME)
    PIPELINE_ROOT = os.path.join(output_data_dir, self._testMethodName)
    PIPELINE_ROOT = os.path.join(output_data_dir, self._testMethodName)
    METADATA_PATH = os.path.join(PIPELINE_ROOT, 'tfx_metadata', 'metadata.db')
    
    if run_pipeline:
      try:
        logging.debug(f"removing: {PIPELINE_ROOT}")
        shutil.rmtree(PIPELINE_ROOT)
      except OSError as e:
        pass
    
      os.makedirs(os.path.join(PIPELINE_ROOT, 'tfx_metadata'),
                exist_ok=True)
  
    ENABLE_CACHE = True
    
    # metadata_connection_config = metadata_store_pb2.ConnectionConfig()
    # metadata_connection_config.sqlite.SetInParent()
    # metadata_connection = metadata.Metadata(metadata_connection_config)
    metadata_connection_config = metadata.sqlite_metadata_connection_config(
      METADATA_PATH)
    
    store = metadata_store.MetadataStore(metadata_connection_config)
    
    tr_dir = os.path.join(get_project_dir(), "src/main/python/movie_lens_tfx")
    
    serving_model_dir = os.path.join(PIPELINE_ROOT, 'serving_model')

    pipeline_factory = PipelineComponentsFactory(
      num_examples=self.num_examples,
      infiles_dict_ser=self.infiles_dict_ser, output_config_ser=self.output_config_ser,
      transform_dir=tr_dir, user_id_max=self.user_id_max, movie_id_max=self.movie_id_max,
      n_genres=self.n_genres, n_age_groups=self.n_age_groups, min_eval_size=self.MIN_EVAL_SIZE,
      batch_size=32, num_epochs=2, device="CPU", serving_model_dir=serving_model_dir)
    
    SETUP_FILE_PATH = os.path.join(get_project_dir(), 'setup.py')
    
    beam_pipeline_args = [
      '--direct_running_mode=multi_processing',
      '--direct_num_workers=0',
      f'--setup_file={SETUP_FILE_PATH}',
    ]
    
    baseline_components = pipeline_factory.build_components(PIPELINE_TYPE.BASELINE)
    
    # create baseline model
    my_pipeline = tfx.dsl.Pipeline(
      pipeline_name=PIPELINE_NAME,
      pipeline_root=PIPELINE_ROOT,
      components=baseline_components,
      enable_cache=ENABLE_CACHE,
      metadata_connection_config=metadata_connection_config,
      beam_pipeline_args=beam_pipeline_args,
    )
    
    if run_pipeline:
      tfx.orchestration.LocalDagRunner().run(my_pipeline)
      logging.debug("BASELINE pipeline finished")
    
    model_artifact = sorted(store.get_artifacts_by_type("Model"),
                            key=lambda
                              x: x.last_update_time_since_epoch,
                            reverse=True)[0]
    model_uri = os.path.join(model_artifact.uri, "Format-Serving")
    loaded_saved_model = tf.saved_model.load(model_uri)
    logging.debug(
      f'test: loaded SavedModel signatures: {loaded_saved_model.signatures}')
    infer_twotower = loaded_saved_model.signatures["serving_default"]
    infer_query = loaded_saved_model.signatures["serving_query"]
    infer_candidate = loaded_saved_model.signatures["serving_candidate"]
    transform_raw = loaded_saved_model.signatures["transform_features"]
    infer_twotower_transformed = loaded_saved_model.signatures[
      "serving_twotower_transformed"]
    infer_query_transformed = loaded_saved_model.signatures[
      "serving_query_transformed"]
    infer_canndidate_transformed = loaded_saved_model.signatures[
      "serving_candidate_transformed"]
    
    infer_query_for_dict = loaded_saved_model.signatures["serving_query_dict"]
    
    schema_artifacts = sorted(store.get_artifacts_by_type("Schema"),
       key=lambda x: x.last_update_time_since_epoch, reverse=True)
    for artifact in schema_artifacts:
      if "pre_transform_schema" in artifact.uri:
        schema_file_path = [os.path.join(artifact.uri, name) for name in
                            os.listdir(artifact.uri)][0]
        schema = tfx.utils.parse_pbtxt_file(schema_file_path, schema_pb2.Schema())
        feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec
        
    for artifact in store.get_artifacts_by_type("Examples"):
      if "MovieLensExampleGen" in artifact.uri:
        raw_examples_uri = os.path.join(artifact.uri, "Split-test")
        break
    file_paths = [os.path.join(raw_examples_uri, name) for name in
                  os.listdir(raw_examples_uri)]
    
    BATCH_SIZE=2
    
    test_raw_ds_ser = tf.data.TFRecordDataset(file_paths, compression_type="GZIP")
    
    def parse_tf_example(example_proto, feature_spec):
      return tf.io.parse_single_example(example_proto, feature_spec)
    
    test_raw_ds = test_raw_ds_ser.map(lambda x: parse_tf_example(x, feature_spec))
    
    def remove_rating(element):
      out = {k: v for k, v in element.items() if k != 'rating'}
      return out
    
    x = test_raw_ds.map(remove_rating)
    test_raw_ds = x.batch(BATCH_SIZE)
    test_raw_ds_ser = test_raw_ds_ser.batch(BATCH_SIZE)
    
    Q_INPUT_KEY = list(infer_query.structured_input_signature[1].keys())[0]
    NEW_Q_INPUT_KEY = list(infer_query_for_dict.structured_input_signature[1].keys())[0]
    
    for serialized_batch in test_raw_ds_ser:
      q_input_dict = {Q_INPUT_KEY: serialized_batch}
      query_embeddings = infer_query(**q_input_dict)['outputs']
      break
      
    for batch in test_raw_ds:
      #input_dict = {NEW_Q_INPUT_KEY: batch}
      #new_query_embeddings = infer_query_for_dict(**q_input_dict)['outputs']
      new_query_embeddings = infer_query_for_dict(
        age=batch['age'],
        gender=batch['gender'],
        genres=batch['genres'],
        movie_id=batch['movie_id'],
        occupation=batch['occupation'],
        timestamp=batch['timestamp'],
        user_id=batch['user_id'])
      
  def _est_io(self):
    file_paths = ["/home/nichole/projects/github/recommender_systems/bin/tune_train_1/test_tune_and_train/TestPythonTransformPipeline/MovieLensExampleGen/output_examples/1/Split-train/tfrecord-00000-of-00004.gz"]
    test_raw_ds_ser = tf.data.TFRecordDataset(file_paths, compression_type="GZIP")
    
    raw_schema_file_path = "/home/nichole/projects/github/recommender_systems/bin/tune_train_1/test_tune_and_train/TestPythonTransformPipeline/Transform/pre_transform_schema/5/schema.pbtxt"
    raw_schema = tfx.utils.parse_pbtxt_file(raw_schema_file_path, schema_pb2.Schema())
    raw_feature_spec = schema_utils.schema_as_feature_spec(raw_schema).feature_spec
  
    def parse_tf_example(example_proto, feature_spec):
      return tf.io.parse_single_example(example_proto, feature_spec)
    raw_ds = test_raw_ds_ser.map(lambda x: parse_tf_example(x, raw_feature_spec))
   
    raw_ds_row = next(iter(raw_ds))
    raw_example = next(iter(test_raw_ds_ser))
    
    # raw_example:
    #    <tf.Tensor: shape=(), dtype=string, numpy=b'\n\x9f\x01\n\x1a\n\x06genres\x12\x10\n\x0e\n\x0cAction|Drama\n\x0f\n\x06gender\x12\x05\n\x03\n\x01M\n\x0f\n\x06rating\x12\x05\x1a\x03\n\x01\x04\n\x16\n\ttimestamp\x12\t\x1a\x07\n\x05\xf4\xab\xbe\xd2\x03\n\x0c\n\x03age\x12\x05\x1a\x03\n\x01-\n\x12\n\x08movie_id\x12\x06\x1a\x04\n\x02\xda\x1a\n\x13\n\noccupation\x12\x05\x1a\x03\n\x01\x07\n\x10\n\x07user_id\x12\x05\x1a\x03\n\x01\x04'>
    
    # raw_ds_row:
    #   {'age': <tf.Tensor: shape=(1,), dtype=int64, numpy=array([45])>, 'gender': <tf.Tensor: shape=(1,), dtype=string, numpy=array([b'M'], dtype=object)>, 'genres': <tf.Tensor: shape=(1,), dtype=string, numpy=array([b'Action|Drama'], dtype=object)>, 'movie_id': <tf.Tensor: shape=(1,), dtype=int64, numpy=array([3418])>, 'occupation': <tf.Tensor: shape=(1,), dtype=int64, numpy=array([7])>, 'rating': <tf.Tensor: shape=(1,), dtype=int64, numpy=array([4])>, 'timestamp': <tf.Tensor: shape=(1,), dtype=int64, numpy=array([978294260])>, 'user_id': <tf.Tensor: shape=(1,), dtype=int64, numpy=array([4])>}
    
    parsed_example = tf.train.Example.FromString(raw_example.numpy())
    
    q = convert_dict_inputs_to_tfexample_ser(raw_ds_row)
    for x in raw_ds.batch(2):
      tf_example = convert_dict_inputs_to_tfexample_ser(x)
      break
    
    tt = 4

    #  features {
    #   feature {
    #     key: "age"
    #     value {
    #       int64_list {
    #         value: 45
    #       }
    #     }
    #   }
    #   feature {
    #     key: "gender"
    #     value {
    #       bytes_list {
    #         value: "M"
    #       }
    #     }
    #   }
    #   feature {
    #     key: "genres"
    #     value {
    #       bytes_list {
    #         value: "Action|Drama"
    #       }
    #     }
    #   }
    #   feature {
    #     key: "movie_id"
    #     value {
    #       int64_list {
    #         value: 3418
    #       }
    #     }
    #   }
    #   feature {
    #     key: "occupation"
    #     value {
    #       int64_list {
    #         value: 7
    #       }
    #     }
    #   }
    #   feature {
    #     key: "rating"
    #     value {
    #       int64_list {
    #         value: 4
    #       }
    #     }
    #   }
    #   feature {
    #     key: "timestamp"
    #     value {
    #       int64_list {
    #         value: 978294260
    #       }
    #     }
    #   }
    #   feature {
    #     key: "user_id"
    #     value {
    #       int64_list {
    #         value: 4
    #       }
    #     }
    #   }
    # }
    
