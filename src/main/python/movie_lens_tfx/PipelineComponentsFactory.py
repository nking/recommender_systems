from tfx.dsl.components.base import base_beam_component

from tfx.components import StatisticsGen, SchemaGen, ExampleValidator, Evaluator, Pusher
import tensorflow_model_analysis as tfma

import enum

from tfx.proto import pusher_pb2
from movie_lens_tfx.ingest_pyfunc_component.ingest_movie_lens_component import *
from movie_lens_tfx.misc import tfrecord_to_parquet

class PIPELINE_TYPE(enum.Enum):
  PREPROCESSING = "preprocessing_data"
  BASELINE = "baseline"
  PRODUCTION = "production"
  BATCH_INFERENCE = "batch_inference"

class PipelineComponentsFactory():
  def __init__(self, num_examples:int, infiles_dict_ser:str, output_config_ser:str, transform_dir:str,
    user_id_max: int, movie_id_max:int, n_genres:int, n_age_groups:int,
    min_eval_size:int=100, batch_size:int=64, num_epochs:int=20, device:str="CPU",
    serving_model_dir:str=None, output_parquet_path:str=None):
    self.num_examples = num_examples
    self.infiles_dict_ser = infiles_dict_ser
    self.output_config_ser = output_config_ser
    self.transform_dir = transform_dir
    self.user_id_max = user_id_max
    self.movie_id_max = movie_id_max
    self.n_genres = n_genres
    self.n_age_groups = n_age_groups
    self.min_eval_size = min_eval_size
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.serving_model_dir = serving_model_dir
    self.output_parquet_path = output_parquet_path
    self.device = device
    
  def build_components(self, type: PIPELINE_TYPE, run_example_diff:bool=False, pre_transform_schema_dir_path:str=None,
    post_transform_schema_dir_path:str=None) -> List[base_beam_component.BaseBeamComponent]:
    
    if type == PIPELINE_TYPE.BATCH_INFERENCE:
      if self.serving_model_dir is None:
        raise ValueError(f"missing serving_model_dir.  location of Format-Serving directory is needed.")
      example_gen = MovieLensExampleGen(
        infiles_dict_ser=self.infiles_dict_ser,
        output_config_ser=self.output_config_ser)
      model_resolver = (tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(type=tfx.types.standard_artifacts.ModelBlessing))
          .with_id('latest_blessed_model_resolver'))
      """
      #BulkInferrer might work with keras3 but not keras2??  the rest of tfx 1.16.0 requires model.saved_model.save(serving_model_dir)
      # possibly works with modles saved as tf.keras.models.save_model(model, save_model_dir_multiply, signatures=signature)
      # but that is not compatible with tfx 1.16.0 which needs tf.saved_model.save
      # I checked out the source code and ran penguin example test
      # tfx/tfx/examples/penguin/penguin_pipeline_local_e2e_test.py which failed similarly
      # added an issue for it:  https://github.com/tensorflow/tfx/issues/7782
      ##
      ## tf.compat.v1.saved_model.tag_constants
      ## GPU	'gpu'
      ## SERVING	'serve'
      ## TPU	'tpu'
      ## TRAINING	'train'
      bulk_inferrer = tfx.components.BulkInferrer(
        examples=example_gen.outputs['output_examples'],
        model = model_resolver.outputs['model'],
        #model_spec type is bulk_inferrer_pb2.ModelSpec
        model_spec=tfx.proto.ModelSpec(
          model_signature_name=['serving_default'],
          tag=tf.saved_model.SERVING
        )
      )
      return [example_gen, model_resolver, bulk_inferrer]
      """
      from movie_lens_tfx.bulk_infer_component.BulkInferrerBeam import BulkInferrerBeam
      bulk_inferrer = BulkInferrerBeam(
        examples=example_gen.outputs['output_examples'],
        model=model_resolver.outputs['model'],
        # model_spec type is bulk_inferrer_pb2.ModelSpec
        model_spec=tfx.proto.ModelSpec(
          model_signature_name=['serving_default'],
          tag=[tf.saved_model.SERVING]
        )
      )
      return [example_gen, model_resolver, bulk_inferrer]
    
    tuner_custom_config = {
      'user_id_max': self.user_id_max,
      'movie_id_max': self.movie_id_max,
      'n_genres': self.n_genres,
      'n_age_groups': self.n_age_groups,
      'feature_acronym': "a",
      'run_eagerly': False,
      "use_bias_corr": False,
      'incl_genres': True,
      'BATCH_SIZE':self.batch_size,
      "NUM_EPOCHS":self.num_epochs,
      "device":self.device,
      "num_examples":self.num_examples,
    }
    
    #TODO: consider how to use tfx.dsl.Cond to check for existing output of example_gen for same inputs
    #  and resolve those instead of repeating work.  similarly a conditional for existing output from
    #  Transform if all other inputs are the same.
    #TODO: consider best integration of Tuning other models and selection of best overall model.
    
    if type == PIPELINE_TYPE.BASELINE:
      tuner_custom_config["feature_acronym"] = ""
      tuner_custom_config["include_genres"] = False
      if run_example_diff:
        logging.error("for BASELINE, cannot select run_example_diff=True, so setting that to False now")
        run_example_diff = False
      
    example_gen = (MovieLensExampleGen(
      infiles_dict_ser=self.infiles_dict_ser,
      output_config_ser=self.output_config_ser))
    
    statistics_gen = StatisticsGen(
      examples=example_gen.outputs['output_examples'])
    
    schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=True)
    
    pre_transform_schema_importer = None
    if pre_transform_schema_dir_path is not None:
      pre_transform_schema_importer = tfx.dsl.Importer(
        source_uri=pre_transform_schema_dir_path,
        artifact_type=tfx.types.standard_artifacts.Schema).with_id(
        'pre_transform_schema_importer')
      pre_transform_example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=pre_transform_schema_importer.outputs['result'])
    else:
     # Performs anomaly detection based on statistics and data schema of raw tf examples
      pre_transform_example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])
    
    example_resolver = None
    example_diff = None
    if run_example_diff:
      include_split_pairs = [('train', 'train'), ('train', 'eval')]
      #TODO: change as needed:
      example_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestArtifactStrategy,
        # Or SpanRangeStrategy
        config={},
        examples=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Examples,
          producer_component_id=example_gen.id
        )
      ).with_id('latest_examples_resolver')
      example_diff = tfx.components.ExampleDiff(
        examples_test=example_gen.outputs['output_examples'],
        examples_base=example_resolver.outputs['examples'],
        include_split_pairs=include_split_pairs
      )
    
    ratings_transform = tfx.components.Transform(
      examples=example_gen.outputs['output_examples'],
      schema=schema_gen.outputs['schema'],
      module_file=os.path.join(self.transform_dir, 'transform_movie_lens.py'))
    
    if type == PIPELINE_TYPE.PREPROCESSING:
      parquet_task = tfrecord_to_parquet.FromTFRecordToParquet(
        transform_graph=ratings_transform.outputs['transform_graph'],
        transformed_examples=ratings_transform.outputs['transformed_examples'],
        output_file_path=self.output_parquet_path
      )
      components = [example_gen, statistics_gen, schema_gen]
      if pre_transform_schema_importer is not None:
        components.append(pre_transform_schema_importer)
      components.append(pre_transform_example_validator)
      if example_resolver is not None:
        components.extend([example_resolver, example_diff])
      components.extend([ratings_transform, parquet_task])
      return components
    
    tuner = tfx.components.Tuner(
      module_file=os.path.join(self.transform_dir, 'tune_train_movie_lens.py'),
      examples=ratings_transform.outputs['transformed_examples'],
      # schema is already in the transform graph
      transform_graph=ratings_transform.outputs['transform_graph'],
      # args: splits, num_steps.  splits defaults are assumed if none given
      custom_config=tuner_custom_config,
    )
    
    # see https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local.py
    # trainer = trainer_movie_lens.MovieLensTrainer(
    trainer = tfx.components.Trainer(
      module_file=os.path.join(self.transform_dir, 'tune_train_movie_lens.py'),
      examples=ratings_transform.outputs['transformed_examples'],
      transform_graph=ratings_transform.outputs['transform_graph'],
      hyperparameters=(tuner.outputs['best_hyperparameters']),
    )
  
    # for the current trained model to be blessed,
    # - resolver does not have to find a baseline model, but it does have to provide at least one value
    #   threshold and no change thresholds.  none of the later because no baseline to compare to.
    
    # - there must be at least 1 threshold defined
    
    # Uses TFMA to compute a evaluation statistics over features of a model and
    # perform quality validation of a candidate model (compared to a baseline).
    if type == PIPELINE_TYPE.BASELINE:
      eval_config = tfma.EvalConfig(
        model_specs=[
          tfma.ModelSpec(signature_name='serving_default', label_key='rating',
            preprocessing_function_names=['transform_features']),
        ],
        slicing_specs=[
          tfma.SlicingSpec(),
          # tfma.SlicingSpec(feature_keys=['hr_wk'])
        ],
        metrics_specs=[
          tfma.MetricsSpec(
            # The metrics added here are in addition to those saved with the
            # model (assuming either a keras model or EvalSavedModel is used).
            # Any metrics added into the saved model (for example using
            # model.compile(..., metrics=[...]), etc) will be computed
            # automatically.
            metrics=[
              # start with simple to see blessing
              tfma.MetricConfig(
                class_name='ExampleCount',
                # Requires at least min_eval_size examples to be present in the evaluation data.
                # This is the simplest possible "blessing" check.
                threshold=tfma.MetricThreshold(
                  value_threshold=tfma.GenericValueThreshold(
                    lower_bound={'value': self.min_eval_size}
                  ))),
            ]
          )
        ])
    else:
      eval_config = tfma.EvalConfig(
        model_specs=[
          tfma.ModelSpec(name='candidate', signature_name='serving_default', label_key='rating', preprocessing_function_names=['transform_features']),
          tfma.ModelSpec(name='baseline', signature_name='serving_default', label_key='rating', preprocessing_function_names=['transform_features'], is_baseline=True)
        ],
        slicing_specs=[
          tfma.SlicingSpec(),
          #tfma.SlicingSpec(feature_keys=['hr_wk'])
        ],
        metrics_specs=[
          tfma.MetricsSpec(
            # The metrics added here are in addition to those saved with the
            # model (assuming either a keras model or EvalSavedModel is used).
            # Any metrics added into the saved model (for example using
            # model.compile(..., metrics=[...]), etc) will be computed
            # automatically.
            metrics=[
              #start with simple to see blessing
              tfma.MetricConfig(
                class_name='ExampleCount',
                # Requires at least min_eval_size examples to be present in the evaluation data.
                # This is the simplest possible "blessing" check.
                threshold=tfma.MetricThreshold(
                  value_threshold=tfma.GenericValueThreshold(
                    lower_bound={'value': self.min_eval_size}
                  ))),
              tfma.MetricConfig(class_name='MeanAbsoluteError',
                # rating scale 0:5 is 0.:1.0 so error of 1 in a rating is 0.20. fail for error of 2 in a rating = 0.4
                threshold=tfma.MetricThreshold(
                  change_threshold=tfma.GenericChangeThreshold(
                    direction=tfma.MetricDirection.LOWER_IS_BETTER,
                    absolute={'value': 0.4}
                  ))),
              #tfma.MetricConfig(class_name='MeanSquaredError'),
            ]
          )
        ])
  
    #is resolver.Resolver the same as tfx.dsl.Resolver?
    model_resolver = (tfx.dsl.Resolver(
      strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
      model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
      model_blessing=tfx.dsl.Channel(type=tfx.types.standard_artifacts.ModelBlessing))
    .with_id('latest_blessed_model_resolver'))
    
    # see https://www.tensorflow.org/tfx/guide/evaluator
    evaluator = Evaluator(
      examples=example_gen.outputs['output_examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      eval_config=eval_config)
    #outputs: evaluation, blessing
    # it creates an empty file called BLESSED or NOT_BLESSED in <pipeline_path>/Evaluator/blessing/9]<number>/
   
    if type == PIPELINE_TYPE.BASELINE:
      #TODO: save schema.pbtxt to version control when have a working version
      components = [example_gen, statistics_gen, schema_gen]
      if pre_transform_schema_importer is not None:
        components.append(pre_transform_schema_importer)
      components.extend([pre_transform_example_validator, ratings_transform,
        tuner, trainer, model_resolver, evaluator])
      return components
    
    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    pusher = None
    with tfx.dsl.Cond(
      evaluator.outputs['blessing'].future()[0].custom_property(
        'blessed') == 1
    ):
      """
      infra_validator = InfraValidator(
        model=trainer.outputs['model'],
        serving_spec=infra_validator_pb2.ServingSpec(
         low_serving=infra_validator_pb2.TensorFlowServing(
            tags=['latest']
          ),
          #kubernet tensorfs, or localDocker https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/proto/ServingSpec
          kubernetes=infra_validator_pb2.KubernetesConfig()
          #or LocalDockerConfig
        ),
        validation_spec=infra_validator_pb2.ValidationSpec(
          max_loading_time_seconds=60,
          num_examples=1000
        ),
        request_spec=tfx.proto.RequestSpec(
          tensorflow_serving=tfx.proto.TensorFlowServingRequestSpec(
            signature_names=['serving_default']
          ),
          num_examples=10
        )
      )
      """
      pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        #infra_blessing=infra_validator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory=self.serving_model_dir)))
    
    components = [example_gen, statistics_gen, schema_gen]
    if pre_transform_schema_importer is not None:
      components.append(pre_transform_schema_importer)
    components.append(pre_transform_example_validator)
    if example_resolver is not None:
      components.extend([example_resolver, example_diff])
    components.extend([ratings_transform, model_resolver, tuner, trainer,evaluator, pusher])
    
    return components