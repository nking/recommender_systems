import pprint
import time

from apache_beam.testing.util import assert_that, is_not_empty, equal_to

from ingest_movie_lens_beam import *

logging.set_verbosity(logging.WARNING)
logging.set_stderrthreshold(logging.WARNING)

pp = pprint.PrettyPrinter()

from helper import *

class IngestMovieLensBeamTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.infiles_dict_ser, self.output_config_ser, self.split_names = get_test_data()
    try:
      self.infiles_dict = deserialize(self.infiles_dict_ser)
    except Exception as ex:
      err = f"error with deserialize(infiles_dict_ser)"
      logging.error(f'{err} : {ex}')
      raise ValueError(f'{err} : {ex}')
    self.name = 'test run of ingest with tfx'


  def testRun2(self):

    expected_schema_cols = [ \
      ("user_id", int), ("movie_id", int), ("rating", int), ("timestamp", int),\
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
      runner='DirectRunner', \
      direct_num_workers = 0, \
      direct_running_mode='multi_processing', \
      #direct_running_mode='multi_threading', \
    )

    with beam.Pipeline(options=options) as pipeline:

      #test read files
      pc = pipeline | f"read_{time.time_ns()}" >> ReadFiles(self.infiles_dict)

      #pc['ratings'] | f'ratings: {time.time_ns()}' >> \
      #  beam.Map(lambda x: print(f'ratings={x}'))
      ratings_pc = pc['ratings']

      r_count = ratings_pc  | f'ratings_count_{random.randint(0, 1000000000000)}' >> beam.combiners.Count.Globally()

      u_count = pc['users']  | beam.combiners.Count.Globally()

      assert_that(pc['movies']  | f'movies_count_{random.randint(0, 1000000000000)}' >> beam.combiners.Count.Globally(), \
        equal_to([3883]), label=f"assert_that_{random.randint(0, 1000000000000)}")
      assert_that(u_count, \
        equal_to([100]), label=f"assert_that_{random.randint(0, 1000000000000)}")
      assert_that(r_count, \
                  equal_to([1000]), label=f"assert_that_{random.randint(0, 1000000000000)}")

      #beam.pvalue.PCollection, List[Tuple[str, Any]]
      ratings, column_name_type_list = \
        pipeline | f"IngestAndJoin_{random.randint(0,1000000000)}" \
        >> IngestAndJoin(infiles_dict = self.infiles_dict)

      assert expected_schema_cols == column_name_type_list

      assert_that(ratings, is_not_empty(), label=f'assert_that_{random.randint(0, 1000000000000)}')

      #retrieved_root = pipeline.pipeline_info.pipeline_root
      #print(f"The pipeline root is: {retrieved_root}")