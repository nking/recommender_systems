
import shutil

from tfx.dsl.io import fileio
from tfx.orchestration import metadata

#import trainer_movie_lens

import tensorflow_transform as tft

from ml_metadata.proto import metadata_store_pb2
from ml_metadata.metadata_store import metadata_store
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
      infiles_dict_ser=self.infiles_dict_ser, output_config_ser=self.output_config_ser,
      transform_dir=tr_dir, user_id_max=self.user_id_max, movie_id_max=self.movie_id_max,
      n_genres=self.n_genres, n_age_groups=self.n_age_groups, min_eval_size=self.MIN_EVAL_SIZE,
      serving_model_dir=serving_model_dir,
    )
    
    beam_pipeline_args = [
      '--direct_running_mode=multi_processing',
      '--direct_num_workers=0'
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
    
