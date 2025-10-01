import apache_beam as beam
from ingest_movie_lens_1m_beam import ingest_and_join

import os
import absl
import json
import pprint
import tempfile

from typing import Any, Dict, List, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import apache_beam as beam

from absl import logging

from tfx.components.example_gen.base_example_gen_executor import BaseExampleGenExecutor
from tfx.components.example_gen.component import FileBasedExampleGen
from tfx.components.example_gen import utils
from tfx.dsl.components.base import executor_spec

from tfx.types import artifact
from tfx.types import artifact_utils
from tfx.types import channel
from tfx.types import standard_artifacts
from tfx.types.standard_artifacts import Examples

from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.decorators import component
from tfx.types.experimental.simple_artifacts import Dataset

from tfx import v1 as tfx
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

# Set up logging.
tf.get_logger().propagate = False
absl.logging.set_verbosity(absl.logging.INFO)
pp = pprint.PrettyPrinter()

print(f"TensorFlow version: {tf.__version__}")
print(f"TFX version: {tfx.__version__}")

import json
from tfx.utils import json_utils

#tf.train.Example is for independent, fixed-size examples
#tf.train.SequenceExample is for variable-length sequential data,
#    such as sentences, time series, or videos.

# for a TFX pipeline, we want the ingestion to be performed by
# an ExampleGen component that accepts input data and formats it as tf.Examples
class MovieLens1mExecutor(BaseExampleGenExecutor):
  def GenerateExamplesByBeam(self, \
    pipeline: beam.Pipeline, \
    exec_properties: Dict[str, Any]\
  ) -> Dict[str, beam.pvalue.PCollection]:
    '''
    :param pipeline:
    :param exec_properties: is a dictionary holding:
      key = "ratings_uri", value = uri for ratings file in ml1m format,
      key = "movies_uri", value = uri for movies file in ml1m format,
      key = "users_uri", value = uri for users file in ml1m format,
      key = headers_present, value = Bool for whether of not the first
        line in the ratings, movies, and users files are headers.
      key = delim, value = the delimiter used between columns in a row.
        e.g. "::"
      key = "ratings_key_col_dict", value = dictionary for ratings_uri of
        keys=coulmn names and values = the column numbers,
      key = "movies_key_col_dict", value = dictionary for ratings_uri of
        keys=coulmn names and values = the column numbers,
      key = "users_key_col_dict", value = dictionary for ratings_uri of
        keys=coulmn names and values = the column numbers,
      key = buckets, value = a list of integers as percents of the whole,

    :return:  a dictionary of the merged, split data as keys=name, values
      = PCollection for a partition
    '''

    # Get input path from artifacts

    #apache_beam.pvalue.DoOutputsTuple
    ratings_tuple = ingest_and_join(pipeline=pipeline, \
      ratings_uri=exec_properties['ratings_uri'], \
      movies_uri=exec_properties['movies_uri'], \
      users_uri=exec_properties['users_uri'], \
      headers_present=exec_properties['headers_present']\,
      delim=exec_properties['delim'],\
      ratings_key_dict=exec_properties['ratings_key_col_dict'], \
      users_key_dict=exec_properties['users_key_col_dict'], \
      movies_key_dict=exec_properties['movies_key_col_dict'])

    buckets=exec_properties['buckets']

    raise RuntimeError('not yet finished')

if __name__ == "__main__":

  #TODO: move this out of source code and into test code

  ratings_uri = "../resources/ml-1m/ratings.dat"
  movies_uri = "../resources/ml-1m/movies.dat"
  users_uri = "../resources/ml-1m/users.dat"
  ratings_key_col_dict = {"user_id": 0, "movie_id": 1, "rating": 2,
                          "timestamp": 3}
  movies_key_col_dict = {"movie_id": 0, "title": 1, "genres": 2}
  users_key_col_dict = {"user_id": 0, "gender": 1, "age": 2, \
                        "occupation": 3, "zipcode": 4}
  headers_present = False
  delim = "::"
  buckets = [80, 10, 10]

  exec_properties = {ratings_uri:ratings_uri, movies_uri:movies_uri, \
    users_uri:users_uri, ratings_key_col_dict:ratings_key_col_dict, \
    movies_key_col_dict:movies_key_col_dict, \
    users_key_col_dict:users_key_col_dict, \
    headers_present:headers_present, delim=delim, buckets=buckets \
  }
