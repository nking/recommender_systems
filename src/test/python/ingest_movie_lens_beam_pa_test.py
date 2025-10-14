import apache_beam as beam
from apache_beam.testing.util import assert_that, is_not_empty, equal_to

import tensorflow as tf

from ingest_movie_lens_beam_pa import *
from movie_lens_utils import *
from helper import *

import time
import random

import absl
from absl import logging
import pprint

from apache_beam.options.pipeline_options import PipelineOptions

import os
import glob

absl.logging.set_verbosity(absl.logging.INFO)
pp = pprint.PrettyPrinter()

class IngestMovieLensBeamPATest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.infiles_dict_ser, self.output_config_ser, self.split_names = \
      get_test_data()
    self.name = 'test pyarrow PTransforms'

  def testReadFiles(self):
    # DirectRunner is default pipeline if options is not specified
    # apache-beam 2.59.0 - 2.68.0 with SparkRunner supports pyspark 3.2.x
    # but not 4.0.0
    # pyspark 3.2.4 is compatible with java >= 8 and <= 11 and python >= 3.6 and <= 3.9
    # start Docker, then use portable SparkRunner
    # https://beam.apache.org/documentation/runners/spark/
    # from pyspark import SparkConf
    options = PipelineOptions( \
      runner='DirectRunner', \
      direct_num_workers=0, \
      direct_running_mode='multi_processing', \
      # direct_running_mode='multi_threading', \
    )

    infiles_dict = deserialize(self.infiles_dict_ser)

    with beam.Pipeline(options=options) as pipeline:
      # test read files
      pc = pipeline | f"read_{time.time_ns()}" >> ReadFiles(
        infiles_dict)

      # pc['ratings'] | f'ratings: {time.time_ns()}' >> \
      #  beam.Map(lambda x: print(f'ratings={x}'))
      ratings_pc = pc['ratings']

      r_count = ratings_pc | f'ratings_count_{random.randint(0, 1000000000000)}' >> beam.combiners.Count.Globally()
      # r_count | 'count ratings' >> beam.Map(lambda x: print(f'len={x}'))
      assert_that(r_count, equal_to([1000]),
                  label=f"assert_that_{random.randint(0, 1000000000000)}")

      assert_that(pc['movies'] | f'movies_count_{random.randint(0, 1000000000000)}' >> beam.combiners.Count.Globally(), \
                  equal_to([3883]),
                  label=f"assert_that_{random.randint(0, 1000000000000)}")
      assert_that(pc['users'] | f'users_count_{random.randint(0, 1000000000000)}' >> beam.combiners.Count.Globally(), \
                  equal_to([100]),
                  label=f"assert_that_{random.randint(0, 1000000000000)}")

  def testIngestAndJoin(self):
    # DirectRunner is default pipeline if options is not specified

    infiles_dict = deserialize(self.infiles_dict_ser)

    options = PipelineOptions( \
      runner='DirectRunner', \
      direct_num_workers=0, \
      direct_running_mode='multi_processing', \
      # direct_running_mode='multi_threading', \
    )

    # write Parquet files if they do not already exist
    dir_path = os.path.dirname(
      os.path.abspath(infiles_dict['ratings']['uri']))
    if not glob.glob(os.path.join(dir_path, "movies*.parquet")):
      with beam.Pipeline(options=options) as pipeline:

        ratings_dict_records = pipeline \
          | f"read_{random.randint(0, 1000000000000)}" \
          >> ReadCSVToRecords(infiles_dict) \

        for key, pa_record in ratings_dict_records.items():
          pa_record \
            | f"write_{random.randint(0, 1000000000000)}" \
            >> WriteParquet(infiles_dict[key], \
            file_path_prefix=f'{dir_path}/{key}')

    ##IngestAndJoin reads those Parquet files

    expected_schema_cols = [ \
      ("user_id", int), ("movie_id", int), ("rating", int),\
      ("timestamp", int), \
      ("gender", str), ("age", int), ("occupation", int), \
      ("genres", str)]

    with beam.Pipeline(options=options) as pipeline:
      # beam.pvalue.PCollection, List[Tuple[str, Any]]
      ratings, column_name_type_list = \
        pipeline | f"IngestAndJoin_{random.randint(0, 1000000000)}" \
        >> IngestAndJoin(infiles_dict=infiles_dict)

      assert expected_schema_cols == column_name_type_list

      assert_that(ratings, is_not_empty(),
                  label=f'assert_that_{random.randint(0, 1000000000000)}')

      file_path_prefix = f'{dir_path}/ratings_joined'
      print(f'file_path_prefix={file_path_prefix}')

      ratings | WriteJoinedRatingsParquet( \
        file_path_prefix=file_path_prefix, \
        column_name_type_list=column_name_type_list)

      found = glob.glob(os.path.join(dir_path, "ratings_joined*.parquet"))
      self.assertIsNotNone(found)
