import os
from ingest_movie_lens_1m_tfx import *
from stringify_ingest_params import stringify_ingest_params

from tfx import v1 as tfx

@dsl.pipeline(
    name="ingest, transform  pipeline",
    description="a pipeline to ingest and transform the data",
)
def ingest_and_transform(ratings_uri : str, movies_uri : str, users_uri : str, \
  ratings_key_col_dict : dict[str, int], \
  movies_key_col_dict : dict[str, int], \
  users_key_col_dict : dict[str, int], \
  partitions : list[int]) -> None:

  input_dict_ser = stringify_ingest_params(ratings_uri, movies_uri, users_uri, \
    ratings_key_col_dict, movies_key_col_dict, users_key_col_dict, \
    partitions)

  ingest_task = ReadMergeAndSplitComponent(input_dict_ser=input_dict_ser)

  for i, part in enumerate(ingest_task.outputs['examples']):
    print(f'part_{i}:\n{part}')

  partition_0 = ingest_task.outputs['examples'].get()[0]

  #stats_gen = tfx.components.StatisticsGen(examples=ingest_task.outputs['examples'])


  '''
  #in a notebook, interactive:
  # see https://www.tensorflow.org/tfx/tutorials/tfx/recommenders
  from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
  context = InteractiveContext()
  context.run(stats_gen, enable_cache=True)
  context.show(stats_gen.outputs['statistics'])

  ratings_schema_gen = tfx.components.SchemaGen(
    statistics=stats_gen.outputs['statistics'],
    infer_feature_shape=False)
  context.run(ratings_schema_gen, enable_cache=True)
  context.show(ratings_schema_gen.outputs['schema'])
  '''

  transform_task = TransformRatingsComponent(\
    examples=ingest_task.outputs['examples'],\
    transformed_examples=transformed_examples_channel,
    # A new Channel for output
    transform_graph=transform_graph_channel  # A new Channel for output
  )

'''
#or instead of using in a component, could test locally with:
#from tfx.orchestration import pipeline
#from tfx.orchestration.local.local_dag_runner import LocalDagRunner
my_pipeline = pipeline.Pipeline(
  pipeline_name='my_tfx_pipeline',
  components=[
    # ... other components
    my_transform_component_instance
  ],
  pipeline_root='./tfx_pipeline_output',
  metadata_connection_config=None # For local execution
)

LocalDagRunner().run(my_pipeline)
'''
