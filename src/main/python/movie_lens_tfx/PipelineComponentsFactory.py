import os
from typing import List

from tfx.dsl.components.base import base_beam_component

from tfx.components import StatisticsGen, SchemaGen, ExampleValidator, Evaluator, Pusher
import tensorflow_model_analysis as tfma

import enum

from tfx.dsl.components.common import resolver
from tfx.proto import pusher_pb2, range_config_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model

from movie_lens_tfx.ingest_pyfunc_component.ingest_movie_lens_component import *
from movie_lens_tfx.misc import tfrecord_to_parquet

class PIPELINE_TYPE(enum.Enum):
  PREPROCESSING = "preprocessing_data"
  BASELINE = "baseline"
  PRODUCTION = "production"

class PipelineComponentsFactory():
  def __init__(self, infiles_dict_ser:str, output_config_ser:str, transform_dir:str,
    user_id_max: int, movie_id_max:int, n_genres:int, n_age_groups:int,
    min_eval_size:int=100, serving_model_dir:str=None, output_parquet_path:str=None):
    self.infiles_dict_ser = infiles_dict_ser
    self.output_config_ser = output_config_ser
    self.transform_dir = transform_dir
    self.user_id_max = user_id_max
    self.movie_id_max = movie_id_max
    self.n_genres = n_genres
    self.n_age_groups = n_age_groups
    self.min_eval_size = min_eval_size
    self.serving_model_dir = serving_model_dir
    self.output_parquet_path = output_parquet_path
    
  def build_components(self, type: PIPELINE_TYPE, run_example_diff:bool=False) -> List[base_beam_component.BaseBeamComponent]:
    tuner_custom_config = {
      'user_id_max': self.user_id_max,
      'movie_id_max': self.movie_id_max,
      'n_genres': self.n_genres,
      'n_age_groups': self.n_age_groups,
      'feature_acronym': "h",
      'run_eagerly': False,
      "use_bias_corr": False,
      'incl_genres': True,
    }
    
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
    
    # Performs anomaly detection based on statistics and data schema.
    example_validator = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])
    
    if run_example_diff:
      example_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestArtifactStrategy,
        # Or SpanRangeStrategy
        config={},
        examples=tfx.dsl.Channel(
          type=tfx.types.standard_artifacts.Examples,
          producer_component_id=example_gen.id
        )
      ).with_id('latest_examples_resolver')
      example_diff = tfx.components.ExampleDiff(
        examples_test=example_gen.outputs['examples'],
        examples_base=example_resolver.outputs['examples'],
      )
    
    ratings_transform = tfx.components.Transform(
      examples=example_gen.outputs['output_examples'],
      schema=schema_gen.outputs['schema'],
      module_file=os.path.join(self.transform_dir, 'transform_movie_lens.py'))
    
    if type == PIPELINE_TYPE.PREPROCESSING:
      parquet_task = tfrecord_to_parquet.FromTFRecordToParquet(
        transform_graph=ratings_transform.outputs['transform_graph'],
        transformed_examples=ratings_transform.outputs[
          'transformed_examples'],
        output_file_path=self.output_parquet_path
      )
      if tuner_custom_config:
        return [example_gen, statistics_gen, schema_gen, example_resolver, example_diff,
                example_validator, ratings_transform, parquet_task]
      else:
        return [example_gen, statistics_gen, schema_gen,
              example_validator, ratings_transform, parquet_task]
    
    # resolver, if needing last trained model as baseline model, needs to be invoked before
    # Tuner and Trainer.
    # but if using last belssed model, it can be invoked after the Trainer
    
    # tfx.v1.dsl.experimental.LatestArtifactStrategy
    # tfx.v1.dsl.experimental.LatestBlessedModelStrategy
    # tfx.v1.dsl.experimental.SpanRangeStrategy
    # Get the latest blessed model for model validation.
    model_resolver = (resolver.Resolver(
      # strategy_class=tfx.dsl.experimental.latest_blessed_model_resolver.LatestBlessedModelResolver,
      strategy_class=tfx.dsl.experimental.LatestArtifactStrategy,
      model=Channel(type=Model),
      # model_blessing=Channel(type=ModelBlessing)
    ).with_id('latest_model_resolver'))
    # 'latest_blessed_model_resolver')
    
    tuner = tfx.components.Tuner(
      module_file=os.path.join(self.transform_dir, 'tune_train_movie_lens.py'),
      examples=ratings_transform.outputs['transformed_examples'],
      # schema is already in the transform graph
      transform_graph=ratings_transform.outputs['transform_graph'],
      # args: splits, num_steps.  splits defaults are assumed if none given
      train_args=tfx.proto.TrainArgs(num_steps=5),
      eval_args=tfx.proto.EvalArgs(num_steps=5),
      custom_config=tuner_custom_config,
    )
    
    trainer_custom_config = {
      'device': "CPU",
    }
    
    # see https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local.py
    # trainer = trainer_movie_lens.MovieLensTrainer(
    trainer = tfx.components.Trainer(
      module_file=os.path.join(self.transform_dir, 'tune_train_movie_lens.py'),
      examples=ratings_transform.outputs['transformed_examples'],
      transform_graph=ratings_transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      hyperparameters=(tuner.outputs['best_hyperparameters']),
      train_args=tfx.proto.TrainArgs(num_steps=5),
      eval_args=tfx.proto.EvalArgs(num_steps=5),
      custom_config=trainer_custom_config,
    )
  
    # for the current trained model to be blessed,
    # - resolver must find a baseline model
    # - there must be at least 1 threshold defined
    
    # Uses TFMA to compute a evaluation statistics over features of a model and
    # perform quality validation of a candidate model (compared to a baseline).
    eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(
        signature_name='serving_default', label_key='rating')],
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
              # Requires at least 1 example to be present in the evaluation data.
              # This is the simplest possible "blessing" check.
              threshold=tfma.MetricThreshold(
                value_threshold=tfma.GenericValueThreshold(
                  lower_bound={'value': self.min_eval_size}
                ))
            #tfma.MetricConfig(class_name='MeanAbsoluteError',
            #  threshold=tfma.MetricThreshold(change_threshold=tfma.GenericChangeThreshold(
            #    direction=tfma.MetricDirection.LOWER_IS_BETTER,
            #    #MAE is 0.26, diff is 0.02.  next_mae must be <= 026 + 0.08 for value=-0.08
            #    absolute={'value': -0.1}
            #  ))
            ),
            #tfma.MetricConfig(class_name='MeanSquaredError'),
            #tfma.MetricConfig(class_name='ExampleCount')
          ]
        )
      ])
    
    # can be used to supplement pre- and post- transform evaluation
    evaluator = Evaluator(
      examples=ratings_transform.outputs['transformed_examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      eval_config=eval_config)
    #outputs: evaluation, blessing
    # it creates an empty file called BLESSED or NOT_BLESSED in <pipeline_path>/Evaluator/blessing/9]<number>/
    # TFMA: EvalResult?  ValidationResult?
    
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
    
    if type == PIPELINE_TYPE.BASELINE:
      return [example_gen, statistics_gen, schema_gen,
              example_validator, ratings_transform, tuner, trainer, model_resolver, evaluator]
    
    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    pusher = Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
          base_directory=self.serving_model_dir)))
    
    if run_example_diff:
      return [example_gen, statistics_gen, schema_gen, example_resolver, example_diff,
              example_validator,
              ratings_transform, model_resolver, tuner, trainer,
              evaluator, pusher]
    else:
      return [example_gen, statistics_gen, schema_gen,
                  example_validator,
                  ratings_transform, model_resolver, tuner, trainer,
                  evaluator, pusher]