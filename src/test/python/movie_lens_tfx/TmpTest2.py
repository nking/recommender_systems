import os.path
import shutil

from apache_beam.options.pipeline_options import PipelineOptions

from movie_lens_tfx.ingest_pyfunc_component.ingest_movie_lens_component import *

from movie_lens_tfx.tune_train_movie_lens import *

from helper import *

tf.get_logger().propagate = False
from absl import logging
logging.set_verbosity(logging.DEBUG)
logging.set_stderrthreshold(logging.DEBUG)

class TmpTest2(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    
  def test_write_movie_tfrecords(self):
    
    options = PipelineOptions(
      runner='DirectRunner',
      direct_num_workers=0,
      direct_running_mode='multi_processing',
      # direct_running_mode='multi_threading',
    )
    
    file_path = os.path.join(get_project_dir(), 'src/main/resources/ml-1m/movies.dat')
    output_uri = os.path.join(get_bin_dir(), "movies_tfrecords")
    os.makedirs(output_uri, exist_ok=True)
    
    column_name_type_list = [('movie_id', int), ('movie_title', str), ('genres', str) ]
    
    with beam.Pipeline(options=options) as pipeline:
  
      pc = pipeline | f"r{random.randint(0, 10000000000)}" >> \
        beam.io.ReadFromText(file_path, skip_header_lines=0, coder=CustomUTF8Coder()) \
        | f'parse_movie_{random.randint(0, 1000000000000)}' >> \
        beam.Map(lambda line: line.split("::"))
      
      examples = (pc | f'movie_ToTFExample_{random.randint(0, 1000000000000)}'
        >> beam.Map(create_example,column_name_type_list))
    
      prefix_path = f'{output_uri}/tfrecords'
      
      examples | f"Serialize_{random.randint(0, 1000000000000)}" \
      >> beam.Map(lambda x: x.SerializeToString()) \
      | f"write_to_tfrecord_{random.randint(0, 1000000000000)}" \
      >> beam.io.tfrecordio.WriteToTFRecord( \
        file_path_prefix=prefix_path, file_name_suffix='.gz')
    