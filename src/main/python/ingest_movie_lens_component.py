import absl
from absl import logging
import pprint
import time
import pickle
import base64
import random

import apache_beam as beam
from tfx.types import standard_artifacts, artifact_utils, standard_component_specs
from tfx.dsl.component.experimental import annotations
from tfx.dsl.component.experimental.decorators import component
from tfx.components.example_gen import write_split
#from tfx.dsl.components.component import component

from movie_lens_utils import *
from ingest_movie_lens_beam import *
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

#A parameter is an argument (int, float, bytes, or unicode string)

@component(use_beam=True)
def ingest_movie_lens_component( \
  #name: tfx.dsl.components.Parameter[str], \
  infiles_dict_ser: tfx.dsl.components.Parameter[str], \
  output_config_ser: tfx.dsl.components.Parameter[str], \
  output_examples: tfx.dsl.components.OutputArtifact[standard_artifacts.Examples], \
  #output_split_config: tfx.dsl.components.OutputArtifact[standard_artifacts.SplitConfig],\
  beam_pipeline: annotations.BeamComponentParameter[beam.Pipeline]=None):
  """
  ingest the ratings, movies, and users files, left join them on ratings,
  and split them into the given buckets in output_config.
  
  Args:
    :param infiles_dict_ser: a string created from using base64 and pickle on the infiles_dict created with
      movie_lens_utils.create_infiles_dict where its input arguments are made
      from movie_lens_utils.create_infile_dict
    :param output_config_ser: string serialized example_gen_pb2.Output which
      must include split_config
    :param output_examples: 
      ChannelParameter(type=standard_artifacts.Examples),
    :param beam_pipeline: injected into method by TFX.  do not supply
      this value
  """
  logging.info("ingest_movie_lens_component")

  try:
    infiles_dict = deserialize(infiles_dict_ser)
  except Exception as ex:
    err = f"error using pickle and base64"
    logging.error(f'{err} : {ex}')
    raise ValueError(f'{err} : {ex}')

  err = infiles_dict_formedness_error(infiles_dict)
  if err:
    logging.error(err)
    raise ValueError(err)
    
  try:
    output_config = deserialize_to_proto(output_config_ser)
  except Exception as ex:
    err = f"error decoding, {ex}"
    logging.error(err)
    raise ValueError(err)

  split_names = [split.name for split in
                 output_config.split_config.splits]

  if not output_examples:
    examples_artifact = standard_artifacts.Examples()
    examples_artifact.splits = split_names
    output_examples = channel_utils.as_channel([examples_artifact])
  else:
    logging.debug(f"output_examples was passed in to component")
    #TODO: consider if need to check for splits

  if isinstance(output_examples, list):
    output_uri = artifact_utils.get_single_instance(output_examples).uri
  else:
    output_uri = output_examples.uri

  logging.debug(f"output_examples TYPE={type(output_examples)}")
  logging.debug(f"output_examples={output_examples}")
  logging.debug(f"split_names={split_names}")

  with beam_pipeline as pipeline:
    #beam.PCollection, List[Tuple[str, Any]
    ratings, column_name_type_list = \
      pipeline | f"IngestAndJoin_{random.randint(0,1000000000)}" \
      >> IngestAndJoin(infiles_dict = infiles_dict)

    total = sum([split.hash_buckets for split in output_config.split_config.splits])
    s = 0
    cumulative_buckets = []
    for split in output_config.split_config.splits:
      s += int(100 * (split.hash_buckets / total))
    cumulative_buckets.append(s)

    #type: apache_beam.DoOutputsTuple
    ratings_tuple = ratings | f'split_{random.randint(0, 1000000000000)}' >> beam.Partition( \
      partition_fn, len(cumulative_buckets), cumulative_buckets, \
      output_config.split_config)

    logging.debug(f"have ratings_tuple.  type={type(ratings_tuple)}")

    logging.debug(f'output_examples.uri={output_uri}')

    write_to_tfrecords = True

    # https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/types/standard_artifacts/Examples
    # files should be written as {uri}/Split-{split_name1}

    # https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/proto/example_gen.proto#L44
    # If VERSION is specified, but not SPAN, an error will be thrown.
    # output_examples.version = infiles_dict['version']

    #output_examples.set_string_custom_property('description',\
    #  'ratings file created from left join of ratings, users, movies')

    if not write_to_tfrecords:
      #write to csv
      column_names = ",".join([t[0] for t in column_name_type_list])
      for i, part in enumerate(ratings_tuple):
        prefix_path = f'{output_uri}/Split-{split_names[i]}'
        write_to_csv(pcollection=part, \
          column_names=column_names, prefix_path=prefix_path, delim='_')
      logging.info(
        f'Examples written to output_examples as CSV to {output_uri}')
    else:
      #write to TFRecords.  by default it uses coder=coders.BytesCoder()
      for i, example_split in enumerate(ratings_tuple):
        split_name = split_names[i]
        prefix_path = f'{output_uri}/Split-{split_name}'
        example_split | f"pcoll_to_tf_{random.randint(0, 1000000000000)}" \
          >> beam.Map(create_example, column_name_type_list) \
          | f"Serialize_{random.randint(0, 1000000000000)}" \
          >> beam.Map(lambda x: x.SerializeToString()) \
          | f"write_to_tfrecord_{random.randint(0, 1000000000000)}" \
          >> beam.io.tfrecordio.WriteToTFRecord(\
            file_path_prefix=prefix_path, file_name_suffix='.tfrecord')
        logging.debug(f"prefix_path={prefix_path}")
    logging.info(
      f'Examples written to output_examples as TFRecords to {output_uri}')