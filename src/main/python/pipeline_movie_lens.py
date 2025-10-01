import os
from ingest_movie_lens_tfx import *
from stringify_ingest_params import stringify_ingest_params
from tfx import v1 as tfx

@tfx.dsl.pipeline(
    name="ingest, transform  pipeline",
    description="a pipeline to ingest and transform the data",
)
def ingest_and_transform_pipeline(ratings_uri : str, movies_uri : str, users_uri : str, \
  ratings_key_col_dict : dict[str, int], \
  movies_key_col_dict : dict[str, int], \
  users_key_col_dict : dict[str, int], \
  partitions : list[int]) -> None:

  input_dict_ser = stringify_ingest_params(ratings_uri, movies_uri, users_uri,
    ratings_key_col_dict, movies_key_col_dict, users_key_col_dict, \
    partitions)

  ingest_task = ReadMergeAndSplitComponent(input_dict_ser=input_dict_ser)

  for i, part in enumerate(ingest_task.outputs['examples']):
    print(f'part_{i}:\n{part}')

  part_0 = ingest_task.outputs['examples'].get()[0]

  stats_gen = tfx.components.StatisticsGen(examples=ingest_task.outputs['examples'])

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

  '''
  transform_task = TransformRatingsComponent(\
    examples=ingest_task.outputs['examples'],\
    transformed_examples=transformed_examples_channel,
    # A new Channel for output
    transform_graph=transform_graph_channel  # A new Channel for output
  )
  '''

if __name__ == "__main__":
  from tfx import v1 as tfx

  ratings_uri = "../resources/ml-1m/ratings.dat"
  movies_uri = "../resources/ml-1m/movies.dat"
  users_uri = "../resources/ml-1m/users.dat"
  ratings_key_col_dict = {"user_id": 0, "movie_id": 1, "rating": 2, "timestamp": 3}
  movies_key_col_dict = {"movie_id": 0, "title": 1, "genres": 2}
  users_key_col_dict = {"user_id": 0, "gender": 1, "age": 2, \
                         "occupation": 3, "zipcode": 4}
  partitions = [80, 10, 10]

  #replace LocalDagRunner with apache-beam, or airflow or kubeflow pipelines or vertex ai, ...
  my_pipeline = ingest_and_transform_pipeline(\
    ratings_uri, movies_uri, users_uri, \
    ratings_key_col_dict, \
    movies_key_col_dict, \
    users_key_col_dict, \
    partitions)

  tfx.orchestration.LocalDagRunner().run(my_pipeline)
