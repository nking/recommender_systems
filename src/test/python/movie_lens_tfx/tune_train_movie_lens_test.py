
import shutil

from tfx.dsl.io import fileio
from tfx.orchestration import metadata
from tfx.components import StatisticsGen, SchemaGen, ExampleValidator
from tfx.utils import io_utils
from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_transform.tf_metadata import schema_utils

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

  def test_tune_and_train(self):
    test_num = "tune_train_1"
    ratings_example_gen = (MovieLensExampleGen(
      infiles_dict_ser=self.infiles_dict_ser,
      output_config_ser = self.output_config_ser))

    statistics_gen = StatisticsGen(examples = ratings_example_gen.outputs['output_examples'])

    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=True)
    
    # Performs anomaly detection based on statistics and data schema.
    example_validator = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])

    if get_kaggle():
      tr_dir = "/kaggle/working/"
    else:
      tr_dir = os.path.join(get_project_dir(), "src/main/python/movie_lens_tfx")

    ratings_transform = tfx.components.Transform(
      examples=ratings_example_gen.outputs['output_examples'],
      schema=schema_gen.outputs['schema'],
      module_file=os.path.join(tr_dir, 'transform_movie_lens.py'))

    tuner_custom_config = {
      'user_id_max' : self.user_id_max,
      'movie_id_max' : self.movie_id_max,
      'n_genres' : self.n_genres,
      'n_age_groups' : self.n_age_groups,
      'run_eagerly' : True,
      "use_bias_corr" : False,
      'incl_genres':False,
      'feature_acronym':"",
      "NUM_EPOCHS" : 2,
      "BATCH_SIZE" : 10,
      "device":"CPU",
      "MAX_TUNE_TRIALS" : 1,
      "num_examples": 1000
    }
    
    tuner = tfx.components.Tuner(
      module_file=os.path.join(tr_dir, 'tune_train_movie_lens.py'),
      examples=ratings_transform.outputs['transformed_examples'],
      #schema is already in the transform graph
      transform_graph=ratings_transform.outputs['transform_graph'],
      #args: splits, num_steps.  splits defaults are assumed if none given
      train_args=tfx.proto.TrainArgs(num_steps=5),
      eval_args=tfx.proto.EvalArgs(num_steps=5),
      custom_config=tuner_custom_config,
    )
    #'user_id_max' 'movie_id_max' 'n_genres' 'run_eagerly'
    
    trainer_custom_config = {
      'device': "CPU",
    }
    
    #see https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local.py
    #trainer = trainer_movie_lens.MovieLensTrainer(
    trainer=tfx.components.Trainer(
      module_file=os.path.join(tr_dir, 'tune_train_movie_lens.py'),
      examples=ratings_transform.outputs['transformed_examples'],
      transform_graph=ratings_transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      # If Tuner is in the pipeline, Trainer can take Tuner's output
      # best_hyperparameters artifact as input and utilize it in the user module
      # code.
      #
      # If there isn't Tuner in the pipeline, either use Importer to import
      # a previous Tuner's output to feed to Trainer, or directly use the tuned
      # hyperparameters in user module code and set hyperparameters to None
      # here.
      #
      # Example of Importer,
      #   hparams_importer = Importer(
      #     source_uri='path/to/best_hyperparameters.txt',
      #     artifact_type=HyperParameters).with_id('import_hparams')
      #   ...
      #   hyperparameters = hparams_importer.outputs['result'],
      hyperparameters=(tuner.outputs['best_hyperparameters']),
      train_args=tfx.proto.TrainArgs(num_steps=5),
      eval_args=tfx.proto.EvalArgs(num_steps=5),
      custom_config=trainer_custom_config,
    )

    #Tuner Component `outputs` contains:
    #- `model`: Channel of type [`standard_artifacts.Model`][tfx.v1.types.standard_artifacts.Model] for trained model.
    #- `model_run`: Channel of type [`standard_artifacts.ModelRun`][tfx.v1.types.standard_artifacts.ModelRun], as the working
    #              dir of models, can be used to output non-model related output
    #              (e.g., TensorBoard logs).

    #TODO: continue with components;
    # see https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local.py

    components = [ratings_example_gen, statistics_gen, schema_gen, example_validator,
                  ratings_transform, tuner, trainer]

    PIPELINE_NAME = 'TestPythonTransformPipeline'
    #output_data_dir = os.path.join(os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',self.get_temp_dir()),self._testMethodName)
    output_data_dir = os.path.join(get_bin_dir(), test_num, self._testMethodName)
    PIPELINE_ROOT = os.path.join(output_data_dir, PIPELINE_NAME)
    #remove results from previous test runs:
    try:
      logging.debug(f"removing: {PIPELINE_ROOT}")
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
    
    SETUP_FILE_PATH = os.path.join(get_project_dir(), 'setup.py')
    
    beam_pipeline_args = [
      '--direct_running_mode=multi_processing',
      '--direct_num_workers=0',
      f'--setup_file={SETUP_FILE_PATH}',
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
    self.assertLessEqual(7, len(artifact_types))
    self.assertLessEqual(7, len(artifacts))
    self.assertLessEqual(7, len(executions))
    # executions has custom_properties.key: "infiles_dict_ser"
    #    and custom_properties.key: "output_config_ser"
    artifact_count = len(artifacts)
    execution_count = len(executions)
    self.assertGreaterEqual(artifact_count, execution_count)
    self.assertGreaterEqual(artifact_count, execution_count)
    
    #ExampleValidator produces standard_artifacts.ExampleAnomalies
    anomalies0_list = store.get_artifacts_by_type("ExampleAnomalies")
    for artifact in anomalies0_list:
      if "post_transform_anomalies" in artifact.uri:
        anomalies_tranformed_uri = os.path.join(artifact.uri, "SchemaDiff.pb")
      else:
        anomalies_raw_uri = os.path.join(artifact.uri, "Split-train", "SchemaDiff.pb")
    anomalies_raw = anomalies_pb2.Anomalies()
    anomalies_raw.ParseFromString(
      io_utils.read_bytes_file(anomalies_raw_uri)
    )
    self.assertNotEqual(anomalies_raw.ByteSize(), 0,
      "The protobuf message for SchemaDiff for ExampleAnomalies should not be empty")
    self.assertTrue(anomalies_raw.IsInitialized())
    
    logging.debug(f"tuner.id={tuner.id}")
    logging.debug(f"trainer.id={trainer.id}")
    self.assertTrue(fileio.exists(os.path.join(PIPELINE_ROOT, tuner.id)))
    self.assertTrue(
      fileio.exists(os.path.join(PIPELINE_ROOT, trainer.id)))

    #https://github.com/tensorflow/tfx/blob/v1.16.0/tfx/types/standard_artifacts.py
    # #TransformCache
    #TransformGraph
    tuner_list = store.get_artifacts_by_type("TunerResults")
    logging.debug(f"tuner_list={tuner_list}")
    self.assertEqual(len(tuner_list), 1)
    latest_tuner_artifact = tuner_list[0]
    #or use last_update_time_since_epoch
    tuner_uri = latest_tuner_artifact.uri
    logging.debug(f"tuner_uri={tuner_uri}")
    file_paths = [os.path.join(tuner_uri, name) for name in os.listdir(tuner_uri)]
    self.assertEqual(len(file_paths), 1)
    for file_path in file_paths:
      tuner_results = json.loads(open(file_path, 'r').read())
      logging.debug(f"tuner_results={tuner_results}")
    
    hyperparameter_list = store.get_artifacts_by_type("HyperParameters")
    logging.debug(f"hyperparameter_list={hyperparameter_list}")
    latest_hyperparameter_artifact = sorted(hyperparameter_list, \
                                            key=lambda
                                              x: x.create_time_since_epoch,
                                            reverse=True)[0]
    # or use last_update_time_since_epoch
    hyperparameter_uri = latest_hyperparameter_artifact.uri
    logging.debug(f"hyperparameter_uri={hyperparameter_uri}")
    self.assertTrue(fileio.exists(hyperparameter_uri))
    file_paths = [os.path.join(hyperparameter_uri, name) for name in os.listdir(hyperparameter_uri)]
    self.assertEqual(len(file_paths), 1)
    for file_path in file_paths:
      hparams = json.loads(open(file_path, 'r').read())
      logging.debug(f"best_hyperparameters.txt keys()={hparams.keys()}")
    
    # =================================
    model_run_list = store.get_artifacts_by_type("ModelRun")
    model_run_artifact = model_run_list[0]
    model_run_uri = model_run_artifact.uri
    logging.debug(f"model_run_uri={model_run_uri}")
    
    model_list = store.get_artifacts_by_type("Model")
    model_artifact = model_list[0]
    model_uri = os.path.join(model_artifact.uri, "Format-Serving")
    logging.debug(f"test: model_uri={model_uri}")
    
    # --- get the transformed test dataset to check that can run the model with expected input structure
    examples_list = store.get_artifacts_by_type("Examples")
    for artifact in examples_list:
      if "transformed_examples" in artifact.uri:
        transfomed_examples_uri = os.path.join(artifact.uri, "Split-test")
        break
    for artifact in examples_list:
      if "MovieLensExampleGen" in artifact.uri:
        raw_examples_uri = os.path.join(artifact.uri, "Split-test")
        break
    logging.debug(f"fraw_examples_uri={raw_examples_uri}")
    logging.debug(f"transfomed_examples_uri={transfomed_examples_uri}")
    
    latest_schema_artifact = sorted(store.get_artifacts_by_type("Schema"),
      key=lambda x: x.last_update_time_since_epoch, reverse=True)[0]
    # or use last_update_time_since_epoch
    schema_uri = latest_schema_artifact.uri
    schema_uri = schema_uri.replace("pre_transform_schema", "post_transform_schema")
    logging.debug(f"schema_uri={schema_uri}")
    schema_file_path = [os.path.join(schema_uri, name) for name in os.listdir(schema_uri)][0]
    
    schema = tfx.utils.parse_pbtxt_file(schema_file_path, schema_pb2.Schema())
    feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec
    
    file_paths = [os.path.join(transfomed_examples_uri, name) for name in os.listdir(transfomed_examples_uri)]
    test_trans_ds_ser = tf.data.TFRecordDataset(file_paths, compression_type="GZIP")
    file_paths = [os.path.join(raw_examples_uri, name) for name in os.listdir(raw_examples_uri)]
    test_raw_ds_ser = tf.data.TFRecordDataset(file_paths, compression_type="GZIP")
    
    def parse_tf_example(example_proto, feature_spec):
      return tf.io.parse_single_example(example_proto, feature_spec)
    test_trans_ds = test_trans_ds_ser.map(lambda x: parse_tf_example(x, feature_spec))
    #test_trans_ds2 = tf.io.parse_example(test_trans_ds_ser, feature_spec) this fails
    
    #might need to remove 'rating' column
    def remove_rating(element):
      out = {k:v for k,v in element.items() if k != 'rating'}
      return out
    
    x = test_trans_ds.map(remove_rating)
    
    ds = x #expected to work when saved has no signatures configured.  default config
    
    loaded_saved_model = tf.saved_model.load(model_uri)
    logging.debug(f'test: loaded SavedModel signatures: {loaded_saved_model.signatures}')
    infer_twotower = loaded_saved_model.signatures["serving_default"]
    infer_query = loaded_saved_model.signatures["serving_query"]
    infer_candidate = loaded_saved_model.signatures["serving_candidate"]
    transform_raw = loaded_saved_model.signatures["transform_features"]
    infer_twotower_transformed = loaded_saved_model.signatures["serving_twotower_transformed"]
    infer_query_transformed = loaded_saved_model.signatures["serving_query_transformed"]
    infer_canndidate_transformed = loaded_saved_model.signatures["serving_candidate_transformed"]

    #test the signatures for tansformed data:
    predictions = []
    query_embeddings = []
    candidate_embeddings = []
    for batch in ds:
      predictions.append(
        infer_twotower_transformed(age=batch['age'], gender=batch['gender'],
          genres=batch['genres'],
          hr=batch['hr'],
          hr_wk=batch['hr_wk'],
          month=batch['month'],
          movie_id=batch['movie_id'],
          occupation=batch['occupation'],
          sec_into_yr=batch['sec_into_yr'],
          user_id=batch['user_id'],
          weekday=batch['weekday'],
          yr=batch['yr']))
      
      query_embeddings.append(
        infer_query_transformed(age=batch['age'], gender=batch['gender'],
          genres=batch['genres'],
          hr=batch['hr'], hr_wk=batch['hr_wk'],
          month=batch['month'],
          movie_id=batch['movie_id'],
          occupation=batch['occupation'],
          sec_into_yr=batch['sec_into_yr'],
          user_id=batch['user_id'],
          weekday=batch['weekday'],
          yr=batch['yr']))
      candidate_embeddings.append(
        infer_canndidate_transformed(age=batch['age'], gender=batch['gender'],
          genres=batch['genres'],
          hr=batch['hr'], hr_wk=batch['hr_wk'],
          month=batch['month'],
          movie_id=batch['movie_id'],
          occupation=batch['occupation'],
          sec_into_yr=batch['sec_into_yr'],
          user_id=batch['user_id'],
          weekday=batch['weekday'],
          yr=batch['yr']))
      
    num_rows = ds.reduce(0, lambda x, _: x + 1).numpy()
    self.assertEqual(len(predictions), num_rows)
    self.assertEqual(len(query_embeddings), num_rows)
    self.assertEqual(len(candidate_embeddings), num_rows)
    
    ## test the signatures for raw data
    TT_INPUT_KEY = list(infer_twotower.structured_input_signature[1].keys())[0]
    Q_INPUT_KEY = list(infer_query.structured_input_signature[1].keys())[0]
    C_INPUT_KEY = list(infer_candidate.structured_input_signature[1].keys())[0]
    TR_INPUT_KEY = list(transform_raw.structured_input_signature[1].keys())[0]
    BATCH_SIZE = 1
    batched_ds = test_raw_ds_ser.batch(BATCH_SIZE)
    predictions = []
    query_embeddings = []
    candidate_embeddings = []
    transformed = []
    for serialized_batch in batched_ds:
      tt_input_dict = {TT_INPUT_KEY: serialized_batch}
      predictions.append(infer_twotower(**tt_input_dict)['outputs'])
      q_input_dict = {Q_INPUT_KEY: serialized_batch}
      query_embeddings.append(infer_query(**q_input_dict)['outputs'])
      c_input_dict = {C_INPUT_KEY: serialized_batch}
      candidate_embeddings.append(infer_candidate(**c_input_dict)['outputs'])
      tr_input_dict = {TR_INPUT_KEY: serialized_batch}
      transformed.append(transform_raw(**tr_input_dict))
    
    self.assertEqual(len(predictions), num_rows)
    self.assertEqual(len(query_embeddings), num_rows)
    self.assertEqual(len(candidate_embeddings), num_rows)
    self.assertEqual(len(transformed), num_rows)
    
    print("Inference completed.")
    
    
    #TODO: add use of fingerprint to show example of using it to verify model
    #fingerprint = tf.saved_model.experimental.read_fingerprint(saved_model_path)
    
    #for ways to load data for checking the model:
    # https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/docs/tutorials/transform/census.ipynb#L1496
    
    #TODO: for examining the SavedModel, see
    # https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/saved_model.md#cli-to-inspect-and-execute-savedmodel
    # to see the signatures and their inputs:
    # saved_model_cli show --dir <path_to_saved_model_dir> --all
    

