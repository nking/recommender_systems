import shutil

from apache_beam.options.pipeline_options import PipelineOptions

from helper import *
from misc.ingest_movie_lens_beam_pa import *

tf.get_logger().propagate = False
logging.set_verbosity(logging.WARNING)
logging.set_stderrthreshold(logging.WARNING)

class IngestMovieLensBeamPATest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.infiles_dict_ser, self.output_config_ser, self.split_names = get_test_data()
    self.name = 'test pyarrow PTransforms'

  def testIngestAndJoin(self):
    # DirectRunner is default pipeline if options is not specified

    infiles_dict = deserialize(self.infiles_dict_ser)

    test_num = "1"

    PIPELINE_NAME = 'TestIngestAndTransformPA'
    # output_data_dir = os.path.join(os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',self.get_temp_dir()),self._testMethodName)
    PIPELINE_ROOT = os.path.join(get_bin_dir(), test_num, self._testMethodName, PIPELINE_NAME)
    # remove results from previous test runs:
    try:
      shutil.rmtree(PIPELINE_ROOT)
    except OSError as e:
      pass
    os.makedirs(PIPELINE_ROOT, exist_ok=True)

    options = PipelineOptions(
      runner='DirectRunner',
      direct_num_workers=0,
      direct_running_mode='multi_processing',
      # direct_running_mode='multi_threading',
    )

    with beam.Pipeline(options=options) as pipeline:

      ratings_dict_records = pipeline \
        | f"read_{random.randint(0, 1000000000000)}" \
        >> ReadCSVToRecords(infiles_dict) \

      for key, pa_record in ratings_dict_records.items():
        pa_record \
          | f"write_{random.randint(0, 1000000000000)}" \
          >> WriteParquet(infiles_dict[key],  file_path_prefix=f'{PIPELINE_ROOT}/{key}')

    ##IngestAndJoin reads those Parquet files
    """
    expected_schema_cols = [
      ("user_id", int), ("movie_id", int), ("rating", int),
      ("timestamp", int),
      ("gender", str), ("age", int), ("occupation", int),
      ("genres", str)]

    with beam.Pipeline(options=options) as pipeline:
      for key in ["ratings", "users", "movies"]:
        infiles_dict[key]["uri"] = f"{PIPELINE_ROOT}/"
      # beam.pvalue.PCollection, List[Tuple[str, Any]]
      ratings, column_name_type_list = \
        pipeline | f"IngestAndJoin_{random.randint(0, 1000000000)}" \
        >> IngestAndJoin(infiles_dict=infiles_dict)

      assert expected_schema_cols == column_name_type_list

      assert_that(ratings, is_not_empty(),
                  label=f'assert_that_{random.randint(0, 1000000000000)}')

      file_path_prefix = f'{PIPELINE_ROOT}/ratings_joined'

      ratings | WriteJoinedRatingsParquet(
        file_path_prefix=file_path_prefix,
        column_name_type_list=column_name_type_list)

      found = glob.glob(os.path.join(PIPELINE_ROOT, "ratings_joined*.parquet"))
      self.assertIsNotNone(found)

      from apache_beam.io import parquetio
      with beam.Pipeline(options=options) as pipeline:
        pc = {}
        for key in ["ratings", "users", "movies"]:
          infiles_dict[key]["uri"] = PIPELINE_ROOT
          file_path_pattern = f"{PIPELINE_ROOT}/{key}*.parquet"
          pc[key] = pipeline | f'read_parquet_{random.randint(0, 1000000000000)}'\
            >> parquetio.ReadFromParquetBatched(file_path_pattern)

        pc['ratings'] | f'ratings: {time.time_ns()}' >> \
          beam.Map(lambda x: print(f'ratings={x}'))

        for key in ["ratings", "users", "movies"]:
            r_count =  pc[key] | f'{key}_count_{random.randint(0, 1000000000000)}' >> beam.combiners.Count.Globally()
            r_count | f'count {key}' >> beam.Map(lambda x: print(f'len={x}'))
      """