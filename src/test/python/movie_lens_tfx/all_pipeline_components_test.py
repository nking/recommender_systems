
import shutil

from tfx.dsl.io import fileio
from tfx.orchestration import metadata
from tfx.components import StatisticsGen, SchemaGen, ExampleValidator, Evaluator, Pusher
import tensorflow_model_analysis as tfma

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.tf_metadata import schema_utils
from google.protobuf import text_format
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing

from tfx.components import InfraValidator
from tfx.proto import infra_validator_pb2

from ingest_movie_lens_component import *
#import trainer_movie_lens

import tensorflow_transform as tft

from ml_metadata.proto import metadata_store_pb2
from ml_metadata.metadata_store import metadata_store
from tune_train_movie_lens import *

from helper import *

tf.get_logger().propagate = False
from absl import logging
logging.set_verbosity(logging.DEBUG)
logging.set_stderrthreshold(logging.DEBUG)

class AllPipelineComponentsTest(tf.test.TestCase):

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

  def test1(self):
    test_num = "test1"
    
    PIPELINE_NAME = 'TestAllPipelineComponents'
    # output_data_dir = os.path.join(os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',self.get_temp_dir()),self._testMethodName)
    output_data_dir = os.path.join(get_bin_dir(), test_num,
                                   self._testMethodName)
    PIPELINE_ROOT = os.path.join(output_data_dir, PIPELINE_NAME)
    # remove results from previous test runs:
    try:
      print(f"removing: {PIPELINE_ROOT}")
      shutil.rmtree(PIPELINE_ROOT)
    except OSError as e:
      pass
    METADATA_PATH = os.path.join(PIPELINE_ROOT, 'tfx_metadata',
                                 'metadata.db')
    os.makedirs(os.path.join(PIPELINE_ROOT, 'tfx_metadata'),
                exist_ok=True)
    
    ENABLE_CACHE = False
    
    # metadata_connection_config = metadata_store_pb2.ConnectionConfig()
    # metadata_connection_config.sqlite.SetInParent()
    # metadata_connection = metadata.Metadata(metadata_connection_config)
    metadata_connection_config = metadata.sqlite_metadata_connection_config(
      METADATA_PATH)
    
    beam_pipeline_args = [
      '--direct_running_mode=multi_processing',
      '--direct_num_workers=0'
    ]
    
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
      'incl_genres':True,
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
      hyperparameters=(tuner.outputs['best_hyperparameters']),
      train_args=tfx.proto.TrainArgs(num_steps=5),
      eval_args=tfx.proto.EvalArgs(num_steps=5),
      custom_config=trainer_custom_config,
    )
    
    # Get the latest blessed model for model validation.
    model_resolver = resolver.Resolver(
      strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel( type=ModelBlessing)).with_id('latest_blessed_model_resolver')
    
    # Uses TFMA to compute a evaluation statistics over features of a model and
    # perform quality validation of a candidate model (compared to a baseline).
    eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(signature_name='serving_default', label_key='rating')],
      slicing_specs=[
        tfma.SlicingSpec(),
        tfma.SlicingSpec(feature_keys=['hr_wk']) #range is [0, 167]
      ],
      metrics_specs=[
        tfma.MetricsSpec(
          # The metrics added here are in addition to those saved with the
          # model (assuming either a keras model or EvalSavedModel is used).
          # Any metrics added into the saved model (for example using
          # model.compile(..., metrics=[...]), etc) will be computed
          # automatically.
          metrics=[
            tfma.MetricConfig(
              class_name='MeanAbsoluteError',
              module='tensorflow.keras.metrics',
              threshold=tfma.MetricThreshold(
                change_threshold=tfma.GenericChangeThreshold(
                  direction=tfma.MetricDirection.LOWER_IS_BETTER,
                  #42 in range[0,167] is 4 hours
                  absolute={'value': 42}))) # v_candidate - v_baseline
                  #relative = {'value': -1e-10}))) # v_candidate / v_baseline
          ]
        )
      ])
    
    #can be used to supplement pre and post- transform data
    evaluator = Evaluator(
      examples=ratings_transform.outputs['transformed_examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      eval_config=eval_config)
  
    """
    infra_validator = InfraValidator(
      model=trainer.outputs['model'],
      serving_spec=infra_validator_pb2.ServingSpec(
        tensorflow_serving=infra_validator_pb2.TensorFlowServing(
          tags=['latest']
        ),
        #kubernets, or localDocker https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/proto/ServingSpec
        kubernetes=infra_validator_pb2.KubernetesConfig()
      ),
      validation_spec=infra_validator_pb2.ValidationSpec(
        max_loading_time_seconds=60,
        num_examples=1000
      )
    )
    """
    
    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    serving_model_dir = os.path.join(PIPELINE_ROOT, 'serving_model')
    
    pusher = Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(base_directory=serving_model_dir)))

    components = [ratings_example_gen, statistics_gen, schema_gen, example_validator,
                  ratings_transform, tuner, trainer, model_resolver, evaluator, pusher]
    
    # imple is tfx.v1.dsl.Pipeline  where tfx is aliased in import
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
    self.assertLessEqual(10, len(artifact_types))
    self.assertLessEqual(10, len(artifacts))
    self.assertLessEqual(10, len(executions))
    # executions has custom_properties.key: "infiles_dict_ser"
    #    and custom_properties.key: "output_config_ser"
    artifact_count = len(artifacts)
    execution_count = len(executions)
    self.assertGreaterEqual(artifact_count, execution_count)
    self.assertGreaterEqual(artifact_count, execution_count)
    
   
