
import shutil
import os
from apache_beam.io.tfrecordio_test import TestTFRecordUtil
from tfx.dsl.io import fileio
from tfx.orchestration import metadata

import tensorflow as tf
from tensorflow_serving.apis import prediction_log_pb2
import os
import glob
from typing import Text, Any, Dict

import tensorflow_transform as tft
import random

from ml_metadata.proto import metadata_store_pb2
from ml_metadata.metadata_store import metadata_store
from tensorflow_transform.tf_metadata import schema_utils

from movie_lens_tfx.tune_train_movie_lens import *

from helper import *

from movie_lens_tfx.PipelineComponentsFactory import *

tf.get_logger().propagate = False
from absl import logging
logging.set_verbosity(logging.WARNING)
logging.set_stderrthreshold(logging.WARNING)

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

  def test_main_model(self):
    
    run_pipeline_before_bulk_infer = True
    run_bulk_infer = True
    
    test_num = "1"
    
    PIPELINE_NAME = 'TestPipelines'
    # output_data_dir = os.path.join(os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',self.get_temp_dir()),self._testMethodName)
    output_data_dir = os.path.join(get_bin_dir(), self._testMethodName, test_num)
    PIPELINE_ROOT = os.path.join(output_data_dir, PIPELINE_NAME)
    METADATA_PATH = os.path.join(PIPELINE_ROOT, 'tfx_metadata', 'metadata.db')
    
    if run_pipeline_before_bulk_infer:
      # remove results from previous test runs:
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
    
    if run_pipeline_before_bulk_infer:
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
      
    ## ========================
    # to test batch inference, will use existing pipeline root and the trained model within.
    # it's MLMD store.
    if run_bulk_infer:
    
      # make fake incoming rating data that will be missing ratings (entered as 0)
      # by taking the user_ids from ratings_1000.dat and taking random movie_ids from the range of movies
      # and filter out any already seen by user.
      # then write to bin directory
      # and create new infiles_dict_ser for those files
      infiles_dict = deserialize(self.infiles_dict_ser)
      ratings_uri = infiles_dict['ratings']['uri']
      dataset = tf.data.TextLineDataset(ratings_uri)
      
      def create_fake_data(line_string):
        # This entire function runs in standard Python
        line = line_string.numpy().decode('utf-8')
        fields = line.split('::')
        fields[1] = str(random.randint(1, self.movie_id_max))
        fields[2] = "0"
        fields[3] = str(int(fields[3]) + 315360000) #adding 10 years
        rejoined_string = "::".join(fields)
        return tf.constant(rejoined_string)
      
      # Apply the Python function to the dataset
      dataset2 = dataset.map(lambda line: tf.py_function(
        func=create_fake_data, inp=[line], Tout=tf.string)
      )
      
      output_file_path = os.path.join(get_bin_dir(), "ratings_to_infer.dat")
      with open(output_file_path, 'w') as f:
        for element in dataset2.as_numpy_iterator():
          f.write(element.decode('utf-8') + '\n')
          
      #infiles_dict['ratings']['uri'] = output_file_path
      
      print(f'serving_model_dir={serving_model_dir}')
      
      artifact_list = store.get_artifacts_by_type("PushedModel")
      print(f"model artifact list={artifact_list}")
      artifact_list = sorted(artifact_list, key=lambda
        x: x.create_time_since_epoch, reverse=True)
      for artifact in artifact_list:
        if "Pusher" in artifact.uri:
          model_uri = artifact.uri
          break
      print(f'model_uri={model_uri}')
      
      output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
          splits=[example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=1)]
        )
      )
      output_config_ser = serialize_proto_to_string(output_config)
      
      pipeline_factory = PipelineComponentsFactory(
        num_examples=self.num_examples,
        infiles_dict_ser=serialize_to_string(infiles_dict),
        output_config_ser=output_config_ser,
        transform_dir=tr_dir, user_id_max=self.user_id_max,
        movie_id_max=self.movie_id_max,
        n_genres=self.n_genres, n_age_groups=self.n_age_groups,
        min_eval_size=self.MIN_EVAL_SIZE,
        batch_size=32, num_epochs=2, device="CPU",
        serving_model_dir=model_uri)
      
      components = pipeline_factory.build_components( PIPELINE_TYPE.BATCH_INFERENCE)
      
      # create baseline model
      my_pipeline = tfx.dsl.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        components=components,
        enable_cache=ENABLE_CACHE,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args,
      )
      
      print(f'begin batch inferrence')
      tfx.orchestration.LocalDagRunner().run(my_pipeline)
      print('end batch inferrence')
      
      #Component BatchInfer outputs contains:
      # inference_result: Channel of type standard_artifacts.InferenceResult to store the inference results.
      # output_examples: Channel of type standard_artifacts.Examples to store the output examples.
      artifact_list = store.get_artifacts_by_type("InferenceResult")
      print(f"model artifact list={artifact_list}")
      latest_artifact = sorted(artifact_list, key=lambda
        x: x.create_time_since_epoch, reverse=True)[0]
      inference_result_uri = latest_artifact.uri
      print(f'inference_result_uri={inference_result_uri}')
      self.assertTrue(os.path.exists(inference_result_uri))
     
      def read_prediction_logs_from_directory(directory_path: Text):
        """
        Reads and parses PredictionLog records from all gzipped TFRecord files
        in the specified directory.
        """
        search_pattern = os.path.join(directory_path,'prediction_logs-*.gz')
        log_files = glob.glob(search_pattern)
        
        if not log_files:
          print(
            f"Error: No prediction log files found in {directory_path} matching pattern 'prediction_logs-*.gz'")
          return
        
        total_records_processed = 0
        
        # Create a PredictionLog message object outside the loop for reuse
        prediction_log = prediction_log_pb2.PredictionLog()
        print(f"--- Found {len(log_files)} files to process ---")
        
        for file_path in log_files:
          print(f"\nProcessing file: {os.path.basename(file_path)}")
          
          # Use TFRecordDataset with the compression_type='GZIP'
          raw_dataset = tf.data.TFRecordDataset( file_path,compression_type='GZIP') #<TFRecordDatasetV2 element_spec=TensorSpec(shape=(), dtype=tf.string, name=None)>
          
          # 3. Iterate over the records in the current file
          file_records_processed = 0
          
          # Use .take() to limit inspection, or remove it to process all records
          for raw_record in raw_dataset:#.take(10):  # Limiting to 10 records per file for inspection
            try:
              record_bytes = raw_record.numpy()
              # Parse the raw bytes into the PredictionLog protobuf object
              prediction_log.ParseFromString(record_bytes)
              log_entry = prediction_log.predict_log.response.outputs
              outputs = log_entry['outputs']
              #print(f'outputs={outputs.float_val}')
              file_records_processed += 1
            
            except tf.errors.DataLossError as e:
              print(
                f"  DataLossError encountered in {os.path.basename(file_path)}: {e}")
              continue
          
          total_records_processed += file_records_processed
          print(
            f"  Finished file. Processed {file_records_processed} records.")
        
        print(f"\n--- DONE ---")
        print(
          f"Total inspected records across all files: {total_records_processed}")
        self.assertEqual(1000, total_records_processed)
      read_prediction_logs_from_directory(inference_result_uri)
  
  def test_metadata_model(self):
    
    test_num = "1"
    
    PIPELINE_NAME = 'TestMetadataPipelines'
    # output_data_dir = os.path.join(os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',self.get_temp_dir()),self._testMethodName)
    output_data_dir = os.path.join(get_bin_dir(), self._testMethodName, test_num)
    PIPELINE_ROOT = os.path.join(output_data_dir, PIPELINE_NAME)
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
    output_parquet_path = os.path.join(PIPELINE_ROOT, 'transformed_parquet_examples')
    os.makedirs(output_parquet_path, exist_ok=True)
    team_lead = "Nichole King"
    import subprocess
    process = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
    git_hash = process.communicate()[0].strip().decode()
    
    pipeline_factory = PipelineComponentsFactory(
      num_examples=self.num_examples,
      infiles_dict_ser=self.infiles_dict_ser,
      output_config_ser=self.output_config_ser,
      transform_dir=tr_dir, user_id_max=self.user_id_max,
      movie_id_max=self.movie_id_max,
      n_genres=self.n_genres, n_age_groups=self.n_age_groups,
      min_eval_size=self.MIN_EVAL_SIZE,
      batch_size=32, num_epochs=2, device="CPU",
      serving_model_dir=serving_model_dir,
      output_parquet_path=output_parquet_path, version= "1.0.0", git_hash=git_hash,
      team_lead=team_lead)
    
    SETUP_FILE_PATH = os.path.join(get_project_dir(), 'setup.py')
    
    beam_pipeline_args = [
      '--direct_running_mode=multi_processing',
      '--direct_num_workers=0',
      f'--setup_file={SETUP_FILE_PATH}',
    ]
    
    baseline_components = pipeline_factory.build_components_metadata_model(PIPELINE_TYPE.BASELINE)
    
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
    
    components = pipeline_factory.build_components_metadata_model(PIPELINE_TYPE.PRODUCTION)
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
    
    #TODO: add unit tests of the metadata model
