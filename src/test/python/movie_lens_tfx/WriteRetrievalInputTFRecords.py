import os.path
import shutil
import numpy as np

"""
This writes various files that I need for the retrieval project.
It really should be refactored into components and unit tests...
"""

from apache_beam.options.pipeline_options import PipelineOptions
from tensorflow_metadata.proto.v0 import schema_pb2

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
class WriteRetrievalInputTFRecords(tf.test.TestCase):
  """
  Creates from user.dat, tfrecords having the columns needed for inputs to data models, that is, columns that are the
  same as the ratings_joined columns.  Columns not present in user.dat are filled in with fake numbers.
  
  Create a similar file from movies.data.
  """
  def setUp(self):
    super().setUp()
    
    self.rewrite_all = False
    
    self.pipeline_options = PipelineOptions(
      runner='DirectRunner',
      direct_num_workers=0,
      direct_running_mode='multi_processing',
      # direct_running_mode='multi_threading',
    )
    
    self.input_path0 = os.path.join(get_project_dir(), 'src/main/resources/ml-1m/ratings_timestamp_sorted_part_1.dat')
    self.output_uri0 = os.path.join(get_bin_dir(), "ratings_pivot")
    
    self.input_path1 = os.path.join(get_project_dir(), 'src/main/resources/ml-1m/movies.dat')
    self.output_uri1 = os.path.join(get_bin_dir(), "movie_emb_inp")
    self.input_path2 = os.path.join(get_project_dir(), 'src/main/resources/ml-1m/users.dat')
    self.output_uri2 = os.path.join(get_bin_dir(), "user_emb_inp")
    self.output_movie_mm_preds_uri = os.path.join(get_bin_dir(),
       "metadata_model_predictions")
    
    self.schema_path = os.path.join(get_project_dir(),
      'src/main/resources/pre_transform/schema.pbtxt')
    self.saved_model_path = os.path.join(get_project_dir(),
      'src/test/resources/serving_model/1763513411')
    
    self.output_pivot_uri = os.path.join(get_bin_dir(), "ratings_and_predictions_pivot")
  
  def test_0(self):
    pipeline0_handle = self._write_train_ratings_pivot_table()
    pipeline1_handle = self._write_movie_tfrecords()
    if pipeline1_handle is not None:
      pipeline1_handle.wait_until_finish()
    if pipeline0_handle is not None:
      pipeline0_handle.wait_until_finish()
    pipeline3_handle = self._left_outer_join_predictions_and_pivot()
    pipeline3_handle.wait_until_finish()
    
  def _write_movie_tfrecords(self):
    """
    1) reads movies.dat and formats it into the joined ratings format of columns, filling in the missing values
    with 0's etc.
    2) create tfexamples from that and writes to tfrecord files
    3) creates movie predictions from the metadata movie model using the PCollection just created
    4) reads users.dat and formats it into the joined ratings format of columns
    5) create tfexamples from that and writes to tfrecord files
    
    inputs are input_path1, input_path2
    outputs are output_uri1, output_uri2, output_movie_mm_preds_uri
    
    Returns:
      pipeline handle for use in waiting until finish etc.
      else None if rewrite_all=False and files are already written
    """
    
    if not self.rewrite_all:
      movie_tf_exists = os.path.exists(self.output_uri1) and bool(os.listdir(self.output_uri1))
      user_tf_exists = os.path.exists(self.output_uri2) and bool(os.listdir(self.output_uri2))
      print(f'files exist: {movie_tf_exists, user_tf_exists}')
      if movie_tf_exists and user_tf_exists:
        return None
        
    try:
      shutil.rmtree(self.output_uri1)
      shutil.rmtree(self.output_movie_mm_preds_uri)
    except Exception as ex:
      pass
    os.makedirs(self.output_uri1, exist_ok=True)
    os.makedirs(self.output_movie_mm_preds_uri, exist_ok=True)
    
    input_column_name_type_list = [('movie_id', int), ('movie_title', str), ('genres', str) ]
    output_column_name_type_list = [('genres', str), ('age', int), ('gender', str),
      ('movie_id', int), ('occupation', int), ('rating', int), ('timestamp', int),('user_id', int)]
    
    pipeline1 = beam.Pipeline(options=self.pipeline_options)
  
    pc = pipeline1 | f"r{random.randint(0, 10000000000)}" >> \
      beam.io.ReadFromText(self.input_path1, skip_header_lines=0, coder=CustomUTF8Coder()) \
      | f'parse_movie_{random.randint(0, 1000000000000)}' >> \
      beam.Map(lambda line: line.split("::"))
    
    #each row in pc is like: ['7', 'Sabrina (1995)', 'Comedy|Romance']

    examples = (pc 
      | f'movie_ToTFExample_{random.randint(0, 1000000000000)}'
      >> beam.Map(create_example_with_fake_for_missing,
      input_column_name_type_list, output_column_name_type_list))
      
    #each row in examples is like: features {feature{key:"age" value{int64_list{value:0}}} feature...}
    
    examples_ser = (examples
      | f"Serialize_{random.randint(0, 1000000000000)}"
      >> beam.Map(lambda x: x.SerializeToString()))
      
    #write to tfrecords
    #these can be used as inputs for the metadata model
    (examples_ser | f'write_movies_fmt_tfrecord{random.randint(0, 1000000000)}'
      >> beam.io.tfrecordio.WriteToTFRecord(file_path_prefix=f'{self.output_uri1}/tfrecords',
      file_name_suffix='.gz'))
    
    #calculate predictions from metadata model
    movie_id_and_preds = (examples_ser
      | f'predict_movies_{random.randint(0, 1000000000)}'
      >> beam.ParDo(_CalcMetadataModelPredictions(
      saved_model_path = self.saved_model_path, schema_path=self.schema_path)))
    
    #each row is a tuple like:
    # (parsed_features['movie_id'] is a tensor like: < tf.Tensor: shape=(1,), dtype = int64, numpy = array([7]),
    # emb is tensor like: tf.Tensor: shape = (1, 32), dtype = float32, numpy = array([[ 0.08...)
    
    # write the movie_id, prediction to file to combine later with pivot of ratings

    #create to tfexamples (tffeatures)
    movie_id_and_preds_examples = (
        movie_id_and_preds | f'preds_ToTFExample_{random.randint(0, 1000000000000)}'
        >> beam.Map(
        WriteRetrievalInputTFRecords.create_example_movie_id_prediction))
    
    #write predictions to tfrecord files
    (movie_id_and_preds_examples | f"Serialize_movie_pred_{random.randint(0, 1000000000000)}"
      >> beam.Map(lambda x: x.SerializeToString())
      | f'write_predictions_tfrecord{random.randint(0, 1000000000)}'
      >> beam.io.tfrecordio.WriteToTFRecord(
      file_path_prefix=f'{self.output_movie_mm_preds_uri}/tfrecords',
      file_name_suffix='.gz'))
    
    result1 = pipeline1.run()
    
    ####### write users tfrecord for embeddings input ######
    
    try:
      shutil.rmtree(self.output_uri2)
    except Exception as ex:
      pass
    os.makedirs(self.output_uri2, exist_ok=True)

    input_column_name_type_list2 = [('user_id', int), ('gender', str), ('age', int), ('occupation', int), ('zipcode', str)]
    pipeline2 = beam.Pipeline(options=self.pipeline_options)
  
    pc2 = pipeline2 | f"r{random.randint(0, 10000000000)}" >> \
      beam.io.ReadFromText(self.input_path2, skip_header_lines=0, coder=CustomUTF8Coder()) \
      | f'parse_movie_{random.randint(0, 1000000000000)}' >> \
      beam.Map(lambda line: line.split("::"))
    
    examples2 = (pc2 | f'user_ToTFExample_{random.randint(0, 1000000000000)}'
      >> beam.Map(create_example_with_fake_for_missing,
      input_column_name_type_list2, output_column_name_type_list))
    
    (examples2 | f"Serialize_{random.randint(0, 1000000000000)}"
      >> beam.Map(lambda x: x.SerializeToString())
      | f"write_to_tfrecord_{random.randint(0, 1000000000000)}"
      >> beam.io.tfrecordio.WriteToTFRecord(
        file_path_prefix=f'{self.output_uri2}/tfrecords', file_name_suffix='.gz'))
      
    result2 = pipeline2.run()
    
    return result1

  def _write_train_ratings_pivot_table(self):
    if not self.rewrite_all:
      ratings_pivot_exists = os.path.exists(self.output_uri0) and bool(os.listdir(self.output_uri0))
      print(f'ratings_pivot_exists files exist: {ratings_pivot_exists}')
      if ratings_pivot_exists:
        return None
      
    #read in ratings: user_id::movie_id::rating::timestamp
    
    try:
      shutil.rmtree(self.output_uri0)
    except Exception as ex:
      pass
    os.makedirs(self.output_uri0, exist_ok=True)
    
    input_column_name_type_list = [('user_id', int), ('movie_id', int), ('rating', int), ('timestamp', int)]
    
    pipeline0 = beam.Pipeline(options=self.pipeline_options)
  
    pc = pipeline0 | f"read_ratings_raw_train_{random.randint(0, 10000000000)}" >> \
      beam.io.ReadFromText(self.input_path0, skip_header_lines=0, coder=CustomUTF8Coder()) \
      | f'parse_ratings_raw_train_{random.randint(0, 1000000000000)}' >> \
      beam.Map(lambda line: line.split("::"))
  
    #create PC of "movie_id", "1", "2", "3", "4", "5"
    pivoted =  WriteRetrievalInputTFRecords._create_pivot(pipeline0, pc)
    
    column_name_type_dict = {'movie_id': int, '1': int, '2': int,
      '3':int, '4': int, '5': int}
      
    #write to file
    examples = (pivoted | f'pivot_ToTFExample_{random.randint(0, 1000000000000)}'
       >> beam.Map(WriteRetrievalInputTFRecords.create_pivot_example, column_name_type_dict))
      
    (examples | f"Serialize_{random.randint(0, 1000000000000)}"
      >> beam.Map(lambda x: x.SerializeToString())
      | f"write_to_tfrecord_{random.randint(0, 1000000000000)}"
      >> beam.io.tfrecordio.WriteToTFRecord(
        file_path_prefix=f'{self.output_uri0}/tfrecords', file_name_suffix='.gz'))
    
    #<apache_beam.runners.portability.fn_api_runner.fn_runner.RunnerResult object
    result0 = pipeline0.run()
    return result0
    
  def _create_pivot(pipeline3, pc):
    """
    from the ratings file which has columns user_id, movie_id, rating, timestamp, create a pivot table with
    columns movie_id, "1", "2", "3", "4", "5" where the numeric columns hold the counts of ratings of that value for that movie.
    NOTE: this method was generated by gemini.google.com and modified where necessary
    """
    movie_key = 1
    rating_key = 2
    keyed_pcollection = pc | f'MapToKeyedRatings_{random.randint(0, 1000000000000)}' >> beam.Map(
      lambda element: (
        int(element[movie_key]),
        (1, 0, 0, 0, 0) if element[rating_key] == "1" else
        (0, 1, 0, 0, 0) if element[rating_key] == "2" else
        (0, 0, 1, 0, 0) if element[rating_key] == "3" else
        (0, 0, 0, 1, 0) if element[rating_key] == "4" else
        (0, 0, 0, 0, 1) if element[rating_key] == "5" else
        (0, 0, 0, 0, 0) # Handle other ratings safely
        )
      )
    #each keyed element is like: keyed=(3430, (0, 0, 0, 1, 0))
    
    combined_pcollection = (keyed_pcollection | f'CombineByMovie_{random.randint(0, 1000000000000)}'
      >> beam.CombinePerKey(_PivotCombineFn()))
    
    #each combined_pcolliection is like: comb=(529, (2, 15, 149, 260, 160))
    
    pivot_pcollection = combined_pcollection | f'MapToPivotSchema_{random.randint(0, 1000000000000)}' >> beam.Map(
      lambda kv: {
        'movie_id': kv[0], '1': kv[1][0], '2': kv[1][1], '3': kv[1][2], '4': kv[1][3], '5': kv[1][4],
      }
    )
    #each pivot is like: pivot={'movie_id': 3418, '1': 30, '2': 89, '3': 346, '4': 478, '5': 242}

    return pivot_pcollection
  
  def create_example_movie_id_prediction(row):
    # each row is a tuple like:
    # (parsed_features['movie_id'] is a tensor like: < tf.Tensor: shape=(1,), dtype = int64, numpy = array([7]),
    # emb is tensor like: tf.Tensor: shape = (1, 32), dtype = float32, numpy = array([[ 0.08...)
    m_id = row[0]
    emb = row[1]
    #print(f'm_id={m_id}; emb={emb}')
    feature_map = {
      'movie_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[m_id])),
      'prediction_mm' : tf.train.Feature(float_list=tf.train.FloatList(value=emb))}
    return tf.train.Example(features=tf.train.Features(feature=feature_map))
  
  def create_pivot_example(row:Dict[str, int], inp_column_name_type_dict: Dict[str, Any]):
    feature_map = {}
    for key, value in row.items():
      try:
        element_type = inp_column_name_type_dict[key]
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
        feature_map[key] = f
      except Exception as ex:
        logging.error(
          f'ERROR: row={row},\nkey-{key}, value={value}, type of value={type(value)}\n'
          f' element_type={element_type}')
        raise ex
    return tf.train.Example(features=tf.train.Features(feature=feature_map))
  
  def _left_outer_join_predictions_and_pivot(self):
    """
    left outer join of movie_id, predictions from self.output_movie_mm_preds_uri
    with the pivot file of movie_id, "1, "2", ... from self.output_uri0
    where join is on movie_id
    
    inputs:
    outputs: output_pivot_uri
    """
    if not self.rewrite_all:
      pivot_exists = os.path.exists(self.output_pivot_uri) and bool(
        os.listdir(self.output_pivot_uri))
      print(f'pivot with preds files exist: {pivot_exists}')
      if pivot_exists:
        return None
      
    try:
      shutil.rmtree(self.output_pivot_uri)
    except Exception as ex:
      pass
    os.makedirs(self.output_uri0, exist_ok=True)
    
    pipeline = beam.Pipeline(options=self.pipeline_options)
    
    # NOTE: beam.CoGroupByKey() needs the keys and values in the groups to be serializable, so
    # in _ParseSerializedExamples, the tensors are extracted to scalars after parsing to dictionaries
    
    feature_spec_preds = {
      'movie_id': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
      'prediction_mm': tf.io.FixedLenFeature(shape=[], dtype=tf.float32, default_value=None),
      }
    
    pc_preds = (pipeline | f"read_movie_preds_{random.randint(0, 10000000000)}"
      >> beam.io.ReadFromTFRecord(file_pattern=f'{self.output_movie_mm_preds_uri}/tfrecords*',
      compression_type=beam.io.filesystem.CompressionTypes.GZIP)
      | f"parse_movie_preds_{random.randint(0, 10000000000)}"
      >>  beam.ParDo(_ParseSerializedExamples(serialize_to_string(feature_spec_preds)))
      )
    
    #pc_preds | "print_pc_preds" >> beam.Map(lambda x: print(f'pc_preds={x}'))
    
    feature_spec_pivot = {
      'movie_id': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
      '1': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
      '2': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
      '3': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
      '4': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
      '5': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
      }
    
    pc_pivot = (pipeline | f"read_pivot_{random.randint(0, 10000000000)}"
      >> beam.io.ReadFromTFRecord(file_pattern=f'{self.output_uri0}/tfrecords*',
      compression_type=beam.io.filesystem.CompressionTypes.GZIP)
      | f"parse_pivot_{random.randint(0, 10000000000)}"
      >>  beam.ParDo(_ParseSerializedExamples(serialize_to_string(feature_spec_pivot)))
      )
    
    #pc_pivot | "print_pc_pivot" >> beam.Map(lambda x: print(f'pc_pivot={x}'))

    l_keyed = pc_preds | f'kv_l_{random.randint(0,1000000000)}' \
      >> beam.Map(lambda x: (x['movie_id'], x))
    
    #l_keyed | "print_l_keyed" >> beam.Map(lambda x: print(f'l_keyed={x}'))

    r_keyed = pc_pivot | f'kv_r_{random.randint(0,1000000000)}' \
      >> beam.Map(lambda x: (x['movie_id'], x))
    
    #r_keyed | "print_r_keyed" >> beam.Map(lambda x: print(f'r_keyed={x}'))

    grouped_data = ({'left': l_keyed, 'right': r_keyed} \
      | f'group_by_key_preds_pivot_{random.randint(0,1000000000)}' \
      >> beam.CoGroupByKey())
    
    #grouped_data | "print_grouped_data" >> beam.Map(lambda x: print(f'grouped={x}'))
    
    try:
      joined_pivot_data = grouped_data \
        | f'left_outer_join_preds_pivot_{random.randint(0,1000000000)}' \
        >> beam.ParDo(_LeftOuterJoinAndSumFn())
    except Exception as ex:
      logging.error(f"ERROR {ex}")
      raise ex
    
    #joined_pivot_data | "print_joined_pivot" >> beam.Map(lambda x: print(f'joined_pivot={x}'))
    
    column_name_type_dict = {'movie_id': int, 'prediction_mm': float, '1': int, '2': int,
      '3': int, '4': int, '5': int}
    
    (joined_pivot_data | f'joined_pivot_ToTFExample_{random.randint(0, 1000000000000)}'
      >> beam.Map(WriteRetrievalInputTFRecords.create_pivot_example,column_name_type_dict)
      | f"Serialize_joined_pivot_{random.randint(0, 1000000000000)}"
      >> beam.Map(lambda x: x.SerializeToString())
      | f"write_joined_pivot_to_tfrecord_{random.randint(0, 1000000000000)}"
      >> beam.io.tfrecordio.WriteToTFRecord(
      file_path_prefix=f'{self.output_pivot_uri}/tfrecords', file_name_suffix='.gz'))
    
    result = pipeline.run()
    
    return result

class _CalcMetadataModelPredictions(beam.DoFn):
  def __init__(self, saved_model_path:str, schema_path:str):
    self.saved_model_path = saved_model_path
    self.schema_path = schema_path
    self.metadata_model = None #pickling error, so delay until setup
    self.feature_spec = None
    self.INPUT_KEY = None
  
  def setup(self):
    """
    Called once per DoFn instance (per worker process) after being unpickled.
    This is where you load heavy, non-serializable resources.
    """
    loaded_user_movie_model = tf.saved_model.load(self.saved_model_path)
    self.metadata_model = loaded_user_movie_model.signatures["serving_default"]
    raw_schema = tfx.utils.parse_pbtxt_file(self.schema_path, schema_pb2.Schema())
    self.feature_spec = schema_utils.schema_as_feature_spec(raw_schema).feature_spec
    self.INPUT_KEY = list(self.metadata_model.structured_input_signature[1].keys())[0]

  def process(self, example_ser):
    parsed_features = tf.io.parse_single_example(example_ser, self.feature_spec)
    #wrap example_ser in a list because it expects a batch of inputs
    pred = self.metadata_model(**{self.INPUT_KEY: [example_ser]})['outputs']
    #parsed_features['movie_id'] is a tensor like: < tf.Tensor: shape=(1,), dtype = int64, numpy = array([7])
    #emb is tensor like: tf.Tensor: shape = (1, 32), dtype = float32, numpy = array([[ 0.08...
    m_id = int(parsed_features['movie_id'].numpy()[0])
    pred = float(pred.numpy()[0])
    yield (m_id, pred)


class _LeftOuterJoinAndSumFn(beam.DoFn):
  def setup(self):
    self.empty_right = {'1': 0, '2': 0, '3': 0,'4': 0,  '5': 0}
    
  def process(self, kv):
    """
    perform left outter join and sum over each of the right entries
    """
    movie_id, grouped_elements = kv
    
    #there should only be 1 left
    left = grouped_elements['left']
    if len(left) != 1:
      raise ValueError(f"left should contain 1 dictionary.  left={left}")
    left = left[0]
    
    right = grouped_elements['right']
    row = left.copy()
    row.update(self.empty_right)
    
    if len(right) == 0:
      yield row
    else:
      for right_row in right:
         for key, value in right_row.items():
           row[key] += value
    yield row

class _ParseSerializedExamples(beam.DoFn):
  """
  converts string serialized examples into dictionaries of tensors, then converts
  the dictionary to dictionary of scalars.
  """
  def __init__(self, feature_spec_ser: str):
    self.feature_spec_ser = feature_spec_ser
    self.feature_spec = None
  
  def setup(self):
    self.feature_spec = deserialize(self.feature_spec_ser)
  
  def process(self, example_ser):
    tensor_dict = tf.io.parse_single_example(example_ser, self.feature_spec)
    # beam.CoGroupByKey cannot handle tensors (unless there is some way with a type hint)
    # so extract scalars from tensors
    for key in tensor_dict:
      val = tensor_dict[key].numpy()
      if isinstance(val, np.floating):
        tensor_dict[key] = float(val)
      else:
        tensor_dict[key] = int(val)
    yield tensor_dict
    
class _PivotCombineFn(beam.CombineFn):
  def create_accumulator(self):
    return (0, 0, 0, 0, 0)
  
  def add_input(self, accumulator, input_rating_count_tuple):
    c1, c2, c3, c4, c5 = accumulator
    i1, i2, i3, i4, i5 = input_rating_count_tuple
    return (c1 + i1, c2 + i2, c3 + i3, c4 + i4, c5 + i5)
  
  def merge_accumulators(self, accumulators):
    c1_sum, c2_sum, c3_sum, c4_sum, c5_sum = 0, 0, 0, 0, 0
    for c1, c2, c3, c4, c5 in accumulators:
      c1_sum += c1
      c2_sum += c2
      c3_sum += c3
      c4_sum += c4
      c5_sum += c5
    return (c1_sum, c2_sum, c3_sum, c4_sum, c5_sum)
  
  def extract_output(self, accumulator):
    return accumulator

def create_example_with_fake_for_missing(row,
  inp_column_name_type_list: List[Tuple[str, Any]],
  outp_column_name_type_list: List[Tuple[str, Any]]):
  
  #row is like: ['25', 'Leaving Las Vegas (1995)', 'Drama|Romance']
 
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
      logging.error(
        f"ERROR: {ex}\nrow={row}, name={name}, element_type={element_type}"
        f"\ni={i}\ncolumn_name_type_list={inp_column_name_type_list}")
      raise ex
  # add fake entries to make consistent with the joined ratings file columns
  for out_name, out_type in outp_column_name_type_list:
    if out_name in feature_map:
      continue
    if out_type == float:
      f = tf.train.Feature(float_list=tf.train.FloatList(value=[0.0]))
    elif out_type == int or element_type == bool:
      if out_name == "timestamp":
        value = 956703932
      else:
        value = 0
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    elif out_type == str:
      if out_name == "genres":
        value = b"Drama"
      elif out_name == "gender":
        value = random.choice([b"M", b"F"])
      else:
        value = b""
      f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    else:
      raise ValueError(f"out_type={out_type}, but only float, int, and str classes are handled.")
    feature_map[out_name] = f
  return tf.train.Example(features=tf.train.Features(feature=feature_map))