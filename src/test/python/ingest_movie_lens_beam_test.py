from unittest import mock
import apache_beam as beam
from apache_beam.testing.util import assert_that, is_not_empty, equal_to

import tensorflow as tf
from tfx.dsl.components.base import executor_spec
from tfx.dsl.io import fileio
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import publisher
from tfx.orchestration.launcher import in_process_component_launcher
from tfx.proto import example_gen_pb2
from tfx.utils import name_utils

from ingest_movie_lens_beam import *
from movie_lens_utils import *

import random

from ml_metadata.proto import metadata_store_pb2

import absl
from absl import logging
import pprint

absl.logging.set_verbosity(absl.logging.DEBUG)
pp = pprint.PrettyPrinter()


class IngestMovieLensBeamTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    kaggle = True
    if kaggle:
      prefix = '/kaggle/working/ml-1m/'
    else:
      prefix = "../resources/ml-1m/"
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
                                    uri=users_uri,
                                    col_names=users_col_names, \
                                    col_types=users_col_types,
                                    headers_present=False, delim="::")

    self.infiles_dict = create_infiles_dict(ratings_dict=ratings_dict, \
                                      movies_dict=movies_dict, \
                                      users_dict=users_dict, version=1)

    self.buckets = [80, 10, 10]
    self.bucket_names = ['train', 'eval', 'test']

    self.name = 'test run of ingest with tfx'

  @mock.patch.object(publisher, 'Publisher')
  def testRun2(self, mock_publisher):

    expected_schema_cols = [ \
      ("user_id", int), ("movie_id", int), ("rating", int), \
      ("gender", str), ("age", int), ("occupation", int), \
      ("genres", str)]

    #DirectRunner is default pipeline if options is not specified
    from apache_beam.options.pipeline_options import PipelineOptions

    #apache-beam 2.59.0 - 2.68.0 with SparkRunner supports pyspark 3.2.x
    #but not 4.0.0
    #pyspark 3.2.4 is compatible with java >= 8 and <= 11 and python >= 3.6 and <= 3.9
    # start Docker, then use portable SparkRunner
    # https://beam.apache.org/documentation/runners/spark/
    #from pyspark import SparkConf
    options = PipelineOptions(\
      #runner='SparkRunner',\
      #runner='PortableRunner',\
      runner='DirectRunner',\
      #spark_conf=spark_conf_list,\
    )

    with beam.Pipeline(options=options) as pipeline:

      #test read files
      pc = pipeline | f"read_{time.time_ns()}" >> ReadFiles(self.infiles_dict)

      #pc['ratings'] | f'ratings: {time.time_ns()}' >> \
      #  beam.Map(lambda x: print(f'ratings={x}'))
      ratings_pc = pc['ratings']

      r_count = ratings_pc  | f'ratings_count_{random.randint(0, 1000000000000)}' >> beam.combiners.Count.Globally()
      #r_count | 'count ratings' >> beam.Map(lambda x: print(f'len={x}'))
      assert_that(r_count, equal_to([1000209]), label=f"assert_that_{random.randint(0, 1000000000000)}")

      assert_that(pc['movies']  | f'movies_count_{random.randint(0, 1000000000000)}' >> beam.combiners.Count.Globally(), \
        equal_to([3883]), label=f"assert_that_{random.randint(0, 1000000000000)}")
      assert_that(pc['users'] | f'users_count_{random.randint(0, 1000000000000)}' >> beam.combiners.Count.Globally(), \
        equal_to([6040]), label=f"assert_that_{random.randint(0, 1000000000000)}")

      #beam.pvalue.PCollection, List[Tuple[str, Any]]
      ratings, column_name_type_list = \
        pipeline | f"IngestAndJoin_{random.randint(0,1000000000)}" \
        >> IngestAndJoin(infiles_dict = self.infiles_dict)

      assert expected_schema_cols == column_name_type_list

      assert_that(ratings, is_not_empty(), label=f'assert_that_{random.randint(0, 1000000000000)}')

      #retrieved_root = pipeline.pipeline_info.pipeline_root
      #print(f"The pipeline root is: {retrieved_root}")