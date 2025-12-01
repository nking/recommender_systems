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

#TODO: write a component for making the user and movie tfrecords needed for inputs to
# the embeddings in the Retrieval project.  for now, hard-wiring the columns instead of using
# pre-transformed schema
class WriteEmbInpToTFRecords(tf.test.TestCase):

  def setUp(self):
    super().setUp()
  
  def create_example_with_fake_for_missing(row, inp_column_name_type_list: List[Tuple[str, Any]],
    outp_column_name_type_list: List[Tuple[str, Any]]):
    """Creates a tf.train.Example from given feature values.
    row were created from beam.io.ReadFromText so are all strings.
    """
    final_keys = set([key for key, type in outp_column_name_type_list])
    feature_map = {}
    for i, value in enumerate(row):
      try:
        element_type = inp_column_name_type_list[i][1]
        name = inp_column_name_type_list[i][0]
        if name not in final_keys:
          continue
        if element_type == float:
          f = tf.train.Feature(
            float_list=tf.train.FloatList(value=[float(value)]))
        elif element_type == int or element_type == bool:
          f = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[int(value)]))
        elif element_type == str:
          f = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[value.encode('utf-8')]))
        else:
          raise ValueError(
            f"element_type={element_type}, but only float, int, and str classes are handled.")
        feature_map[name] = f
      except Exception as ex:
        logging.error(f"ERROR: {ex}\nrow={row}, name={name}, element_type={element_type}"
          f"\ni={i}\ncolumn_name_type_list={inp_column_name_type_list}")
        raise ex
    #add fake entries to make consistent with the joined ratings file columns
    for out_name, out_type in outp_column_name_type_list:
      if out_name in feature_map:
        continue
      if element_type == float:
        f = tf.train.Feature(float_list=tf.train.FloatList(value=[0.0]))
      elif element_type == int or element_type == bool:
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=[0]))
      elif element_type == str:
        f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[b""]))
      else:
        raise ValueError(
          f"element_type={element_type}, but only float, int, and str classes are handled.")
      feature_map[name] = f
    return tf.train.Example( features=tf.train.Features(feature=feature_map))
    
  def test_write_movie_tfrecords(self):
    
    options = PipelineOptions(
      runner='DirectRunner',
      direct_num_workers=0,
      direct_running_mode='multi_processing',
      # direct_running_mode='multi_threading',
    )
    
    ## write movies tfrecord for embeddings input
    file_path = os.path.join(get_project_dir(), 'src/main/resources/ml-1m/movies.dat')
    output_uri = os.path.join(get_bin_dir(), "movie_emb_inp")
    os.makedirs(output_uri, exist_ok=True)
    try:
      shutil.rmtree(output_uri)
    except Exception as ex:
      pass
    
    input_column_name_type_list = [('movie_id', int), ('movie_title', str), ('genres', str) ]
    output_column_name_type_list = [('genres', str), ('age', int), ('gender', str),
      ('movie_id', int), ('occupation', int), ('rating', int), ('timestamp', int),('user_id', int)]
    
    with beam.Pipeline(options=options) as pipeline:
  
      pc = pipeline | f"r{random.randint(0, 10000000000)}" >> \
        beam.io.ReadFromText(file_path, skip_header_lines=0, coder=CustomUTF8Coder()) \
        | f'parse_movie_{random.randint(0, 1000000000000)}' >> \
        beam.Map(lambda line: line.split("::"))
        
      examples = (pc | f'movie_ToTFExample_{random.randint(0, 1000000000000)}'
                  >> beam.Map(WriteEmbInpToTFRecords.create_example_with_fake_for_missing, input_column_name_type_list, output_column_name_type_list))
    
      prefix_path = f'{output_uri}/tfrecords'
      
      examples | f"Serialize_{random.randint(0, 1000000000000)}" \
      >> beam.Map(lambda x: x.SerializeToString()) \
      | f"write_to_tfrecord_{random.randint(0, 1000000000000)}" \
      >> beam.io.tfrecordio.WriteToTFRecord( \
        file_path_prefix=prefix_path, file_name_suffix='.gz')
      
    ## write users tfrecord for embeddings input
    file_path = os.path.join(get_project_dir(), 'src/main/resources/ml-1m/users.dat')
    output_uri = os.path.join(get_bin_dir(), "user_emb_inp")
    os.makedirs(output_uri, exist_ok=True)
    try:
      shutil.rmtree(output_uri)
    except Exception as ex:
      pass

    input_column_name_type_list = [('user_id', int), ('gender', str), ('age', int), ('occupation', int), ('zipcode', str)]
    output_column_name_type_list = [('genres', str), ('age', int), ('gender', str),
      ('movie_id', int), ('occupation', int), ('rating', int), ('timestamp', int),('user_id', int)]
    
    with beam.Pipeline(options=options) as pipeline:
  
      pc = pipeline | f"r{random.randint(0, 10000000000)}" >> \
        beam.io.ReadFromText(file_path, skip_header_lines=0, coder=CustomUTF8Coder()) \
        | f'parse_movie_{random.randint(0, 1000000000000)}' >> \
        beam.Map(lambda line: line.split("::"))
      
      examples = (pc | f'user_ToTFExample_{random.randint(0, 1000000000000)}'
                  >> beam.Map(WriteEmbInpToTFRecords.create_example_with_fake_for_missing, input_column_name_type_list, output_column_name_type_list))
    
      prefix_path = f'{output_uri}/tfrecords'
      
      examples | f"Serialize_{random.randint(0, 1000000000000)}" \
      >> beam.Map(lambda x: x.SerializeToString()) \
      | f"write_to_tfrecord_{random.randint(0, 1000000000000)}" \
      >> beam.io.tfrecordio.WriteToTFRecord( \
        file_path_prefix=prefix_path, file_name_suffix='.gz')
