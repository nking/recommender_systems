import shutil

from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions

from helper import *
from movie_lens_tfx.misc.ingest_movie_lens_beam_pa import *
from movie_lens_tfx.utils.ingest_movie_lens_beam import IngestAndJoin as IngestAndJoin0

tf.get_logger().propagate = False
logging.set_verbosity(logging.DEBUG)
logging.set_stderrthreshold(logging.DEBUG)

class IngestMovieLensBeamPATest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.infiles_dict_ser, self.output_config_ser, self.split_names = get_test_data()
    self.name = 'test pyarrow PTransforms'

  def testIngestAndJoin(self):
    # DirectRunner is default pipeline if options is not specified

    #Using the full files and 2 ratings partions

    infiles_dict = deserialize(self.infiles_dict_ser)
    infiles_dict["users"]["uri"] = os.path.join(get_project_dir(),"src/main/resources/ml-1m/users.dat")
    infiles_dict["movies"]["uri"] = os.path.join(get_project_dir(), "src/main/resources/ml-1m/movies.dat")

    PIPELINE_NAME = 'TestIngestAndTransformPA2'
    # output_data_dir = os.path.join(os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',self.get_temp_dir()),self._testMethodName)
    PIPELINE_ROOT = os.path.join(get_bin_dir(),  PIPELINE_NAME, self._testMethodName)
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
    
    from apache_beam.metrics import Metrics
    class RowCounter(beam.DoFn):
      def __init__(self, metric_name):
        self.counter = Metrics.counter(self.__class__, metric_name)
      def process(self, element):
        self.counter.inc()
        yield element
        
    #from apache_beam.testing.test_pipeline import TestPipeline
    #from apache_beam.testing.util import assert_that, equal_to
    import glob
    #with TestPipeline(options=options) as pipeline:
    step_names = []
    pipeline =  beam.Pipeline(options=options)
    for r_number in [1, 2]:
      for key in ["ratings", "users", "movies"]:
        if key == "ratings":
          infiles_dict[key]["uri"] = os.path.join(get_project_dir(),
            f"src/main/resources/ml-1m/ratings_timestamp_sorted_part_{r_number}.dat")
        
        ratings, column_name_type_list =\
          pipeline | f"IngestAndJoin_{random.randint(0, 1000000000)}"\
          >> IngestAndJoin0(infiles_dict=infiles_dict)
        
        """
        ##DEBUG
        sampled_data = ratings | f'Sample Elements_{random.randint(0, 1000000000)}' >> beam.combiners.Sample.FixedSizeGlobally(5)
        (
          sampled_data
          | f'Flatten List_{random.randint(0, 1000000000)}' >> beam.FlatMap(lambda x: x)
          | f'Print Rows_{random.randint(0, 1000000000)}' >> beam.Map(print)
        )
        """
          
        file_path_prefix = os.path.join(PIPELINE_ROOT, f'ratings_sorted_{r_number}_joined')
        
        ratings | f"WriteJoinedRatingsParquet_{random.randint(0, 1000000000)}" \
          >>  WriteJoinedRatingsParquet(
          file_path_prefix=file_path_prefix,
          column_name_type_list=column_name_type_list)
        
        step_name = f'Count Rows_{random.randint(0, 1000000000)}'
        
        ratings | step_name >> beam.ParDo(RowCounter('total_rows'))
        step_names.append(step_name)
    
    result = pipeline.run()
    result.wait_until_finish()
    final_count_integer = 0
    for step_name in step_names:
      query_result = result.metrics().query(
        # Filter for the counter we defined
        beam.metrics.MetricsFilter().with_name('total_rows').with_step(step_name)
          #.with_result_type('COMMITTED')
      )
      if query_result['counters']:
        count = query_result['counters'][0].committed
        print(f"✅ Extracted Integer Count: {count}")
        final_count_integer += count
      else:
        print("❌ Metrics not found.")
    
    pipeline = beam.Pipeline(options=options)
    file_pattern = os.path.join(PIPELINE_ROOT,'ratings_sorted*.parquet')
    pc_parquet = (pipeline
      | f'Read Parquet_{random.randint(0, 1000000000)}' >> parquetio.ReadFromParquet(file_pattern)
    )
    step_name = f'Count_parquet_Rows_{random.randint(0, 1000000000)}'
    pc_parquet | step_name >> beam.ParDo(RowCounter('total_rows'))
    result = pipeline.run()
    result.wait_until_finish()
    query_result = result.metrics().query(
      # Filter for the counter we defined
      beam.metrics.MetricsFilter().with_name('total_rows').with_step(
        step_name)
    )
    if query_result['counters']:
      final_count_integer2 = query_result['counters'][0].committed
      print(f"✅ Extracted Integer Count2: {final_count_integer2}")
    else:
      print("❌ Metrics2 not found.")
      
    #self.assertEqual(final_count_integer2, final_count_integer)
    self.assertEqual(final_count_integer2, 1000209)
    
  def _estWriteRawToParquet(self):
    # DirectRunner is default pipeline if options is not specified

    infiles_dict = deserialize(self.infiles_dict_ser)

    test_num = "1"

    PIPELINE_NAME = 'TestIngestAndTransformPA'
    # output_data_dir = os.path.join(os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',self.get_temp_dir()),self._testMethodName)
    PIPELINE_ROOT = os.path.join(get_bin_dir(), PIPELINE_NAME, self._testMethodName, test_num)
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
