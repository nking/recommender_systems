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

from movie_lens_utils import *
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
      ingest_and_join(pipeline=pipeline, infiles_dict = infiles_dict)

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
