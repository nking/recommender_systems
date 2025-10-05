import os
import absl
from absl import logging
import pprint
import time
import pickle

import apache_beam as beam
from tfx.types import standard_artifacts
from tfx.dsl.component.experimental import annotations
from tfx.dsl.component.experimental.decorators import component
#from tfx.dsl.components.component import component

from movie_lens_utilss import *
from ingest_movie_lens_beam import ingest_and_join
from tfx.proto import example_gen_pb2

from tfx import v1 as tfx

import tensorflow as tf

tf.get_logger().propagate = False
absl.logging.set_verbosity(absl.logging.DEBUG)
pp = pprint.PrettyPrinter()

print(f"TensorFlow version: {tf.__version__}")
print(f"TFX version: {tfx.__version__}")

#when use_beam=True, when the TFX pipeline is compiled and run with a
# Beam orchestrator, TFX automatically injects a Beam pipeline into
# that argument, there is no need to supply it directly.

@component(use_beam=True)
def ingest_movie_lens_component( \
  #name: tfx.dsl.components.Parameter[str], \
  infiles_dict_ser: tfx.dsl.components.Parameter[str], \
  bucket_names: tfx.dsl.components.Parameter[str], \
  buckets: tfx.dsl.components.Parameter[str], \
  output_examples: tfx.dsl.components.OutputArtifact[standard_artifacts.Examples], \
  #output_split_config: tfx.dsl.components.OutputArtifact[standard_artifacts.SplitConfig],\
  beam_pipeline: annotations.BeamComponentParameter[beam.Pipeline]=None) -> None:
  """
  ingest the ratings, movies, and users files, left join them on ratings,
  and split them into the given buckets by percentange and bucket_name.
  
  Args:
    :param infiles_dict_ser: a string created from using pickle.dumps()
      on the infiles_dict created with 
      movie_lens_utils.create_infiles_dict where its input arguments are made
      from movie_lens_utils.create_infile_dict
    :param buckets: list of partitions in percent.
      this is a string from pickle.dumps(the_list)
    :param bucket_names: list of partitions names corresponding to 
      buckets.  this is a string from pickle.dumps(the_list)
    :param output_examples: 
      ChannelParameter(type=standard_artifacts.Examples),
    :param beam_pipeline: injected into method by TFX.  do not supply
      this value
  """
  logging.info("ingest_movie_lens_component")

  try:
    infiles_dict = pickle.loads(infiles_dict_ser)
  except Exception as ex:
    err = f"error using pickle.loads(infiles_dict_ser)"
    logging.error(f'{err} : {ex}')
    raise ValueError(f'{err} : {ex}')

  err = infiles_dict_formedness_error(infiles_dict)
  if err:
    logging.error(err)
    raise ValueError(err)
    
  try:
    buckets = pickle.loads(buckets)
  except Exception as ex:
    err = f"error using pickle.loads(buckets), {ex}"
    logging.error(err)
    raise ValueError(err)

  try:
    bucket_names = pickle.loads(bucket_names)
  except Exception as ex:
    err = f"error using pickle.loads(bucket_namess), {ex}"
    logging.error(err)
    raise ValueError(err)

  if len(buckets) != len(bucket_names):
    err = (
      f'deserialized len(buckets) != deserialized len(bucket_names)'
      f' buckets={buckets}, bucket_names={bucket_names}')
    logging.error(f'ERROR: {err}')
    raise ValueError(f'ERROR: {err}')

  # see https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/proto/example_gen.proto#L146
  splits = [example_gen_pb2.SplitConfig.Split(name=n, hash_buckets=b) \
    for n, b in zip(bucket_names, buckets)]

  output_config = example_gen_pb2.Output(
    split_config=example_gen_pb2.SplitConfig(
      splits=splits
    )
  )

  with beam_pipeline as pipeline:
    #beam.pvalue.PCollection, List[Tuple[str, Any]
    ratings, column_name_type_list = \
      pipeline | f'ingest_and_join_{time.time_ns()}'>> \
      ingest_and_join(infiles_dict = infiles_dict)

    #perform partitions
    total = sum(buckets)
    s = 0
    cumulative_buckets = []
    for b in buckets:
      s += int(100*(b/total))
      cumulative_buckets.append(s)

    #type: apache_beam.pvalue.DoOutputsTuple
    ratings_tuple = ratings | f'split_{time.time_ns()}' >> beam.Partition( \
      partition_fn, len(buckets), cumulative_buckets, \
      output_config.split_config)

    logging.debug(f"have ratings_tuple.  type={type(ratings_tuple)}")

    logging.debug(f'output_examples.uri={output_examples.uri}')

    write_to_tfrecords = True

    output_examples.set_string_custom_property('description',\
      'ratings file created from left join of ratings, users, movies')

    if not write_to_tfrecords:
      #write to csv
      column_names = ",".join([t[0] for t in column_name_type_list])
      for i, part in enumerate(ratings_tuple):
        prefix_path = f'{output_examples.uri}/{bucket_names[i]}'
        write_to_csv(pcollection=part, \
          column_names=column_names, prefix_path=prefix_path, delim='_')
    else:
      #write to TFRecords
      for i, part in enumerate(ratings_tuple):
          prefix_path = f'{output_examples.uri}/{bucket_names[i]}'
          convert_to_tf_example(part, column_name_type_list) \
            | f"Serialize {time.time_ns()}" >> beam.Map(lambda x: x.SerializeToString()) \
            | f"write_to_tf {time.time_ns()}" >> beam.io.tfrecordio.WriteToTFRecord(\
            file_path_prefix=prefix_path, file_name_suffix='.tfrecord')


if __name__ == "__main__":

  from tfx.orchestration.local.local_dag_runner import LocalDagRunner

  #TODO: move this out of source code and into test code
  #  ... unittest.mock in Python
  # see https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/base_example_gen_executor_test.py

  print(f'begin test')

  kaggle = True
  if kaggle:
    prefix = '/kaggle/working/ml-1m/'
    output_dir = '/kaggle/working/bin'
  else:
    prefix = "../resources/ml-1m/"
    output_dir = '../../bin'
  ratings_uri = f"{prefix}ratings.dat"
  movies_uri = f"{prefix}movies.dat"
  users_uri = f"{prefix}users.dat"

  ratings_col_names = ["user_id", "movie_id", "rating"]
  ratings_col_types = [int, int,
                       int]  # for some files, ratings are floats
  movies_col_names = ["movie_id", "title", "genres"]
  movies_col_types = [int, str, str]
  users_col_names = ["user_id", "gender", "age", "occupation",
                     "zipcode"]
  users_col_types = [int, str, int, int, str]

  expected_schema_cols = [ \
    ("user_id", int), ("movie_id", int), ("rating", int), \
    ("gender", str), ("age", int), ("occupation", int), \
    ("genres", str)]

  ratings_dict = create_infile_dict(for_file='ratings', \
                                uri=ratings_uri,
                                col_names=ratings_col_names, \
                                col_types=ratings_col_types,
                                headers_present=False, delim="::")

  movies_dict = create_infile_dict(for_file='movies', \
                               uri=movies_uri,
                               col_names=movies_col_names, \
                               col_types=movies_col_types,
                               headers_present=False, delim="::")

  users_dict = create_infile_dict(for_file='users', \
                              uri=users_uri, col_names=users_col_names, \
                              col_types=users_col_types,
                              headers_present=False, delim="::")

  infiles_dict = create_infiles_dict(ratings_dict=ratings_dict, \
                             movies_dict=movies_dict, \
                             users_dict=users_dict)

  infiles_dict_ser = pickle.dumps(infiles_dict)
  buckets = [80, 10, 10]
  buckets = pickle.dumps(buckets)
  bucket_names = ['train', 'eval', 'test']
  bucket_names = pickle.dumps(bucket_names)

  name = "test_tfx_component"

  #exec_properties = {ratings_uri:ratings_uri, movies_uri:movies_uri, \
  #  users_uri:users_uri, ratings_key_col_dict:ratings_key_col_dict, \
  #  movies_key_col_dict:movies_key_col_dict, \
  #  users_key_col_dict:users_key_col_dict, \
  #  headers_present:headers_present, delim:delim, buckets:buckets \
  #}
  #exec_properties = json.dumps(exec_properties)

  #context = InteractiveContext()

  ratings_example_gen = ingest_movie_lens_component( \
    infiles_dict_ser=infiles_dict_ser, \
    bucket_names=bucket_names, buckets=buckets \
  )

  #statistics_gen = tfx.components.StatisticsGen(\
  #  examples=ratings_example_gen.outputs['output_examples'])

  PIPELINE_NAME = "movie_lens_ingest_test"
  PIPELINE_ROOT = os.path.join(output_dir, 'pipelines', PIPELINE_NAME)
  # Path to a SQLite DB file to use as an MLMD storage.
  METADATA_PATH = os.path.join(output_dir, 'metadata', PIPELINE_NAME, \
    'metadata.db')

  os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)
  os.makedirs(PIPELINE_ROOT, exist_ok=True)

  my_pipeline = tfx.dsl.Pipeline(\
    pipeline_name=PIPELINE_NAME, \
    pipeline_root=PIPELINE_ROOT,\
    metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH),\
    components=[ratings_example_gen], enable_cache=True)

  LocalDagRunner().run(my_pipeline)

  print("tests done")
