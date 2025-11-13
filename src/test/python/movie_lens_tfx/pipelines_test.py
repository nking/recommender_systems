
import shutil

from tfx.dsl.io import fileio
from tfx.orchestration import metadata

#import trainer_movie_lens

import tensorflow_transform as tft

from ml_metadata.proto import metadata_store_pb2
from ml_metadata.metadata_store import metadata_store
from tensorflow_transform.tf_metadata import schema_utils

from movie_lens_tfx.tune_train_movie_lens import *

from helper import *

from movie_lens_tfx.PipelineComponentsFactory import *

tf.get_logger().propagate = False
from absl import logging
logging.set_verbosity(logging.INFO)
logging.set_stderrthreshold(logging.INFO)

class PipelinesTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.infiles_dict_ser, self.output_config_ser, self.split_names = \
      get_test_data()
    self.num_examples = 1000
    self.user_id_max = 6040
    self.movie_id_max = 3952
    self.n_genres = N_GENRES
    self.n_age_groups = N_AGE_GROUPS
    self.n_occupations = 21
    self.MIN_EVAL_SIZE = 50 #make this larger for production pipeline
    self.name = 'test run of pipelines'

  def test1(self):
    test_num = "1"
    
    PIPELINE_NAME = 'TestPipelines'
    # output_data_dir = os.path.join(os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',self.get_temp_dir()),self._testMethodName)
    output_data_dir = os.path.join(get_bin_dir(), self._testMethodName, test_num)
    PIPELINE_ROOT = os.path.join(output_data_dir, PIPELINE_NAME)
    # remove results from previous test runs:
    try:
      logging.debug(f"removing: {PIPELINE_ROOT}")
      shutil.rmtree(PIPELINE_ROOT)
    except OSError as e:
      pass
    METADATA_PATH = os.path.join(PIPELINE_ROOT, 'tfx_metadata',
                                 'metadata.db')
    os.makedirs(os.path.join(PIPELINE_ROOT, 'tfx_metadata'),
                exist_ok=True)
    
    ENABLE_CACHE = True
    
    # metadata_connection_config = metadata_store_pb2.ConnectionConfig()
    # metadata_connection_config.sqlite.SetInParent()
    # metadata_connection = metadata.Metadata(metadata_connection_config)
    metadata_connection_config = metadata.sqlite_metadata_connection_config(
      METADATA_PATH)
    
    store = metadata_store.MetadataStore(metadata_connection_config)
    
    if get_kaggle():
      tr_dir = "/kaggle/working/"
    else:
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
    
    tfx.orchestration.LocalDagRunner().run(my_pipeline)
    logging.debug("BASELINE pipeline finished")
    
    artifact_types = store.get_artifact_types()
    logging.debug(f"MLMD store artifact_types={artifact_types}")
    artifacts = store.get_artifacts()
    logging.debug(f"MLMD store artifacts={artifacts}")
    
    components = pipeline_factory.build_components(PIPELINE_TYPE.PRODUCTION)
    # simulate experimentation of one model family
    my_pipeline = tfx.dsl.Pipeline(
      pipeline_name=PIPELINE_NAME,
      pipeline_root=PIPELINE_ROOT,
      components=components,
      enable_cache=ENABLE_CACHE,
      metadata_connection_config=metadata_connection_config,
      beam_pipeline_args=beam_pipeline_args,
    )
    
    tfx.orchestration.LocalDagRunner().run(my_pipeline)
    logging.debug("PRODUCTION pipeline finished")
    
    # Use built-in conditional features of KFP, Airflow, etc. to enable conditional logic in the workflow.
    # evaluate models and if passed,
    # then human verification, then if passed, push to serving
    
    artifact_types = store.get_artifact_types()
    logging.debug(f"MLMD store artifact_types={artifact_types}")
    artifacts = store.get_artifacts()
    logging.debug(f"MLMD store artifacts={artifacts}")
   
    executions = store.get_executions()
    logging.debug(f"MLMD store executions={executions}")
    self.assertLessEqual(10, len(artifact_types))
    self.assertLessEqual(10, len(artifacts))
    self.assertLessEqual(10, len(executions))
    # executions has custom_properties.key: "infiles_dict_ser"
    #    and custom_properties.key: "output_config_ser"
    artifact_count = len(artifacts)
    execution_count = len(executions)
    self.assertGreaterEqual(artifact_count, execution_count)
    self.assertGreaterEqual(artifact_count, execution_count)

    #can make an enterprise yaml file for kubernetes
    # using @dsl.pipeline deocration on a method accepting start arguments, and invoking the task
    # then kfp.compiler.Compiler().compile()
    
    #recommender_systems/bin/test1/1/TestPipelines/Trainer/model/15
    #type MODEL
    
    #uri: "recommender_systems/bin/test1/1/TestPipelines/Evaluator/evaluation/16"
    #type ModelEvaluation
    
    #uri: "recommender_systems/bin/test1/1/TestPipelines/Evaluator/blessing/16"
    #type ModelBlessing
    
    #uri: "recommender_systems/bin/test1/1/TestPipelines/Pusher/pushed_model/17"
    #type PushedModel
    
    #===================================================
    #validate extraction and use of saved_model signatures
    latest_schema_artifact = sorted(store.get_artifacts_by_type("Schema"),
      key=lambda x: x.last_update_time_since_epoch, reverse=True)[0]
    # or use last_update_time_since_epoch
    schema_uri = latest_schema_artifact.uri
    schema_uri = schema_uri.replace("pre_transform_schema", "post_transform_schema")
    logging.debug(f"schema_uri={schema_uri}")
    schema_file_path = [os.path.join(schema_uri, name) for name in os.listdir(schema_uri)][0]
    
    schema = tfx.utils.parse_pbtxt_file(schema_file_path, schema_pb2.Schema())
    feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec
    
    examples_list = store.get_artifacts_by_type("Examples")
    for artifact in examples_list:
      if "transformed_examples" in artifact.uri:
        transfomed_examples_uri = os.path.join(artifact.uri, "Split-test")
        break
    for artifact in examples_list:
      if "MovieLensExampleGen" in artifact.uri:
        raw_examples_uri = os.path.join(artifact.uri, "Split-test")
        break
    file_paths = [os.path.join(transfomed_examples_uri, name) for name
                  in os.listdir(transfomed_examples_uri)]
    test_trans_ds_ser = tf.data.TFRecordDataset(file_paths,
                                                compression_type="GZIP")
    file_paths = [os.path.join(raw_examples_uri, name) for name in
                  os.listdir(raw_examples_uri)]
    test_raw_ds_ser = tf.data.TFRecordDataset(file_paths,
                                              compression_type="GZIP")
    
    def parse_tf_example(example_proto, feature_spec):
      return tf.io.parse_single_example(example_proto, feature_spec)
    
    test_trans_ds = test_trans_ds_ser.map(
      lambda x: parse_tf_example(x, feature_spec))
    
    # test_trans_ds2 = tf.io.parse_example(test_trans_ds_ser, feature_spec) this fails
    
    # might need to remove 'rating' column
    def remove_rating(element):
      out = {k: v for k, v in element.items() if k != 'rating'}
      return out
    
    x = test_trans_ds.map(remove_rating)
    
    ds = x  # expected to work when saved has no signatures configured.  default config
    
    model_artifact = sorted(store.get_artifacts_by_type("Model"),
           key=lambda x: x.last_update_time_since_epoch, reverse=True)[0]
    model_uri = os.path.join(model_artifact.uri, "Format-Serving")
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
      prediction = infer_twotower(**tt_input_dict)['outputs']
      predictions.append(prediction.numpy())
      q_input_dict = {Q_INPUT_KEY: serialized_batch}
      query_embeddings.append(infer_query(**q_input_dict)['outputs'])
      c_input_dict = {C_INPUT_KEY: serialized_batch}
      candidate_embeddings.append(infer_candidate(**c_input_dict)['outputs'])
      tr_input_dict = {TR_INPUT_KEY: serialized_batch}
      transformed_element = transform_raw(**tr_input_dict)
      transformed.append(transformed_element)
    
    self.assertEqual(len(predictions), num_rows)
    self.assertEqual(len(query_embeddings), num_rows)
    self.assertEqual(len(candidate_embeddings), num_rows)
    self.assertEqual(len(transformed), num_rows)
    #====================================================
    
    eval_list = store.get_artifacts_by_type("ModelEvaluation")
    print(f"model evaluation list={eval_list}")
    latest_eval_artifact = sorted(eval_list, key=lambda
      x: x.create_time_since_epoch, reverse=True)[0]
    # or use last_update_time_since_epoch
    latest_eval_uri = latest_eval_artifact.uri
    print(f"latest_evaluri={latest_eval_uri}")
    eval_result = tfma.load_eval_result(latest_eval_uri)
    print(f'eval_result={eval_result}')
    
    #eval_attr_dict = eval_result.get_attributions_for_all_slices()
    eval_metrics_dict = eval_result.get_metrics_for_all_slices()
    #print(f"eval_attr_dict={eval_attr_dict}")
    print(f"eval_metrics_dict={eval_metrics_dict}")
    
    """for notebook:
    tfma.view.render_slicing_metrics(eval_result)
    
    # Render a plot to compare the candidate model vs. the baseline for a specific metric (e.g., Mean Absolute Error)
    tfma.view.render_plot(
      eval_result,
      metric_name='MeanAbsoluteError',
      slicing_column='house_type'  # Or any feature you sliced on
    )
    
    ## Load the validation results if eval config included thresholds:
    #validation_result = tfma.load_validation_result(output_path=output_path)
    ## You can print the result to see which thresholds passed or failed
    #print(validation_result)
    """
    
