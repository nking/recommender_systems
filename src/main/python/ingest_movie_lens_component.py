import os
import absl
from absl import logging
import json
import pprint

import apache_beam as beam
from tfx.types import standard_artifacts
from tfx.dsl.component.experimental import annotations
from tfx.dsl.component.experimental.decorators import component
#from tfx.dsl.components.component import component

from ingest_movie_lens_beam import ingest_and_join
from partition_funcs import *

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
  ratings_uri: tfx.dsl.components.Parameter[str], \
  movies_uri: tfx.dsl.components.Parameter[str], \
  users_uri: tfx.dsl.components.Parameter[str], \
  headers_present: tfx.dsl.components.Parameter[bool], \
  delim: tfx.dsl.components.Parameter[str], \
  ratings_key_col_dict: tfx.dsl.components.Parameter[str], \
  movies_key_col_dict: tfx.dsl.components.Parameter[str], \
  users_key_col_dict: tfx.dsl.components.Parameter[str], \
  bucket_names: tfx.dsl.components.Parameter[str], \
  buckets: tfx.dsl.components.Parameter[str], \
  output_examples: tfx.dsl.components.OutputArtifact[standard_artifacts.Examples],\
  beam_pipeline: annotations.BeamComponentParameter[beam.Pipeline] = None) -> None:
  """
  ingest the ratings, movies, and users files, left join them on ratings,
  and split them into the given buckets by percentange and bucket_name.
  
  Args:
    :param ratings_uri: uri of ratings.dat or equivalent
    :param movies_uri: uri of movies.dat or equiv
    :param users_uri: uri of users.dat or equiv
    :param headers_present: whether or not a header line is present as first line
    :param delim: the delimeter between columns in a row
    :param ratings_key_col_dict: for ratings file, a dictionary with key:values
      being header_column_name:column number.  
      this is a string from json.dumps(the_dict)
    :param movies_key_col_dict: for movies file, a dictionary with key:values
      being header_column_name:column number.
      this is a string from json.dumps(the_dict)
    :param users_key_col_dict: for users file, a dictionary with key:values
      being header_column_name:column number.
      this is a string from json.dumps(the_dict)
    :param buckets: list of partitions in percent.
      this is a string from json.dumps(the_list)
    :param bucket_names: list of partitions names corresponding to buckets.
      this is a string from json.dumps(the_list)
    :param output_examples: ChannelParameter(type=standard_artifacts.Examples),
    :param beam_pipeline: injected into method by TFX.  do not supply
       this value
  """
  logging.info("ingest_movie_lens_component")

  ratings_key_col_dict = json.loads(ratings_key_col_dict)
  movies_key_col_dict = json.loads(movies_key_col_dict)
  users_key_col_dict = json.loads(users_key_col_dict)
  buckets = json.loads(buckets)
  bucket_names = json.loads(bucket_names)

  if len(buckets) != len(bucket_names):
    err = (
      f'deserialized buckets must be same length as deserialized bucket_names'
      f' buckets={buckets}, bucket_names={bucket_names}')
    logger.error(f'ERROR: {err}: {ex}')
    raise ValueError(f'ERROR: {err}: {ex}')

  # see https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/proto/example_gen.proto#L146
  splits = [example_gen_pb2.SplitConfig.Split(name=n, hash_buckets=b) \
    for n, b in zip(bucket_names, buckets)]

  output_config = example_gen_pb2.Output(
    split_config=example_gen_pb2.SplitConfig(
      splits=splits
    )
  )

  with beam_pipeline as pipeline:
    ratings = ingest_join_and_split(pipeline=pipeline, \
      ratings_uri=ratings_uri, movies_uri=movies_uri, \
      users_uri=users_uri, headers_present=headers_present, delim=delim, \
      ratings_key_dict=ratings_key_col_dict, \
      users_key_dict=users_key_col_dict, \
      movies_key_dict=movies_key_col_dict, \
      buckets=partitions)

    total = sum(a)
    s = 0
    cumulative_buckets = []
    for b in buckets:
      s += int(100*(b/total))
      cumulative_buckets.append()

    #type: apache_beam.pvalue.DoOutputsTuple
    ratings_tuple = ratings \
      | f'split_{time.time_ns()}' >> beam.Partition( \
      _PartitionFn, len(buckets), cumulative_buckets, output_config.split_config)

    logging.debug(f"have ratings_tuple.  type={type(ratings_tuple)}")

    logging.debug(f'output_examples.uri={output_examples.uri}')

    """
    # Create a TFRecord file path within the output artifact's URI
    output_path = os.path.join(output_examples.uri, 'data.tfrecord')
    # Write the data as TF Examples to the specified path
    with tf.io.TFRecordWriter(output_path) as writer:
      edit this for ratings
      for i in range(len(data['feature_a'])):
        feature = {
          'feature_a': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[data['feature_a'][i]])),
          'feature_b': tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[data['feature_b'][i].encode('utf-8')]))
        }
        example = tf.train.Example(
          features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
  
    logging.info(f"Generated Examples written to: {output_path}")
  
    # You can also set custom properties on the output artifact if needed
    # output_examples.set_string_custom_property('description', 'Dummy examples generated by MyExampleGen')
    """

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

  ratings_key_col_dict = {"user_id": 0, "movie_id": 1, "rating": 2,\
                          "timestamp": 3}
  ratings_key_col_dict = json.dumps(ratings_key_col_dict)
  movies_key_col_dict = {"movie_id": 0, "title": 1, "genres": 2}
  movies_key_col_dict = json.dumps(movies_key_col_dict)
  users_key_col_dict = {"user_id": 0, "gender": 1, "age": 2, \
                        "occupation": 3, "zipcode": 4}
  users_key_col_dict = json.dumps(users_key_col_dict)
  delim = "::"

  #these might need to be serialized into strings for tfx,
  # use json.dumps
  headers_present = False
  buckets = [80, 10, 10]
  buckets = json.dumps(buckets)
  bucket_names = ['train', 'eval', 'test']
  bucket_names = json.dumps(bucket_names)

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
    ratings_uri=ratings_uri, movies_uri=movies_uri, \
    users_uri=users_uri, headers_present=headers_present, \
    delim=delim, ratings_key_col_dict=ratings_key_col_dict, \
    users_key_col_dict=users_key_col_dict, \
    movies_key_col_dict=movies_key_col_dict, \
    bucket_names=bucket_names, buckets=buckets \
  )

  #statistics_gen = tfx.components.StatisticsGen(\
  #  examples=ratings_example_gen.outputs['output_examples'])

  PIPELINE_NAME = "movie_ens_ingest_test"
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
