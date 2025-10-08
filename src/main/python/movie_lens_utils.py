from typing import Any, Dict, List, Literal, Union, Tuple
from tensorflow_metadata.proto.v0 import schema_pb2

import bisect
import hashlib
import pickle
import base64

import tensorflow as tf

from tfx.proto import example_gen_pb2

def create_infile_dict(for_file: Literal['ratings', 'movies', 'users'], \
  uri: str, col_names: List[str], col_types : List[Any], \
  headers_present:bool, delim:str) -> Dict[str, Union[str, Dict]]:
  """
  create a dictionary for an input file.
  :param delim: string of delimiter between the columns of a row
  :param headers_present: True if first line is header line, else False
  :param for_file: string having value 'ratings', 'movies', or 'usres'
  :param uri: a string giving the uri
  :param col_names: a list of the column names
  :param col_types: a list of the column types.
     types can be int, float or str or bytes.
  :return:
    dictionary with key-value pairs:
      key=for_file value which is a string having value 'ratings', 'movies', or 'usres',
      value = dictionary having key-value pairs:
         key='uri', value = string of uri,
         key='cols', value = dictionary with key-value pairs:
           key=col_name, value=dictionary with key-value pairs:
             'index': int, 'type': Any
  """
  if for_file not in ['ratings', 'movies', 'users']:
    raise ValueError(f"for_file must be 'ratings', 'movies', or 'users'. "
      f"you entered {for_file}")

  if len(col_names) != len(col_types):
    raise ValueError(f"col_names length must be same as col_types length")

  for _type in col_types:
    if _type not in [int, float, str, bytes]:
      raise ValueError(
        f"col_types can hold int, float, str, or bytes.  unrecognized: {_type}")

  out = {for_file:{'cols':{}, 'uri':uri, 'headers_present':headers_present, \
    'delim':delim}}

  for i, name in enumerate(col_names):
    out[for_file]['cols'][name] = {'index':i, 'type': col_types[i]}

  return out

def create_infiles_dict(ratings_dict: Dict[str, Union[str, Dict]], \
  movies_dict: Dict[str, Union[str, Dict]], \
  users_dict: Dict[str, Union[str, Dict]], version:int=1) \
  -> Dict[str, Union[str, Dict]]:
  """
  merge the 3 dictionaries, each created by create_infile_dict, into single
  output dictionary
  :param ratings_dict:
  :param movies_dict:
  :param users_dict:
  :param version: version for output tf.train.Examples artifact
  :return: a merge of the 3 dictionaries
  """

  return {**ratings_dict, **movies_dict, **users_dict, 'version': version}

def _assert_dict_1(ml_dict: Dict) -> Union[str, None]:
  """test contents of dict[key]"""
  if 'cols' not in ml_dict:
    return f"dictionary is missing ['cols']"

  if 'uri' not in ml_dict:
    return "dictionary is missing key ['uri']"

  if ml_dict['uri'] is None:
    return "dictionary is missing value for ['uri']"

  if 'headers_present' not in ml_dict:
    return "dictionary is missing key ['headers_present']"

  if ml_dict['headers_present'] is None:
    return "dictionary is missing value for ['headers_present']"

  if 'delim' not in ml_dict:
    return "dictionary is missing key ['delim']"

  if ml_dict['delim'] is None:
    return "dictionary is missing value for ['delim']"

  for name in ml_dict['cols']:
    if 'index' not in ml_dict['cols'][name]:
      return f"missing ml_dict[key]['cols'][{name}]['index']"
    if 'index' not in ml_dict['cols'][name]:
      return f"missing ml_dict[key]['cols'][{name}]['type']"

  return None

def create_example(row, column_name_type_list: List[Tuple[str, Any]]):
  """Creates a tf.train.Example from given feature values.
  row were created from beam.io.ReadFromText so are all strings.
  """
  feature_map = {}
  for i, value in enumerate(row):
    element_type = column_name_type_list[i][1]
    name = column_name_type_list[i][0]
    if element_type == float:
      f = tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))
    elif element_type == int or element_type == bool:
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))
    elif element_type == str:
      f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))
    else:
      raise ValueError(f"element_type={element_type}, but only float, int, and str classes are handled.")
    feature_map[name] = f
  return tf.train.Example(features=tf.train.Features(feature=feature_map))

def create_namedtuple_schemas(infiles_dict: Dict[str, Union[str, Dict]]) -> Dict[str, List[Tuple]]:
  """
  from a dictionary created with create_infiles_dict, create a dictionary of
  of lists of tuples of column names and types that can be used as a
  schema for a PCollection for each top level key
  :param infiles_dict:
  :return: a dictionary of list of tuples of column name and types,
  useable with coders.registry.register_coder
  """
  out = {}
  for key in infiles_dict:
    if key == "version":
      continue
    s = []
    for col_name in infiles_dict[key]['cols']:
      idx = infiles_dict[key]['cols'][col_name]['index']
      t = infiles_dict[key]['cols'][col_name]['type']
      s.append((col_name, idx, t))
    s.sort(key=lambda x: x[1])
    s2 = []
    for c, i, t in s:
      s2.append((c, t))
    out[key] = s2
  return out

def infiles_dict_formedness_error(ml_dict: Dict[str, Union[str, Dict]]) -> Union[str, None]:
  """
  verify expected structure of ml_dict, and return None if correct else
  error message if not
  :param ml_dict: dictionary of information for the ratings, movies, and
    users files.
  :return: None if ml_dict structure is as expected, else return error
    message
  """
  for key in ml_dict:
    if key not in ['ratings', 'movies', 'users', 'version']:
      err = (f"key expected to be one of 'ratings', 'movies', 'users', but is {key}"
             f" ml_dict={ml_dict}")
      return err
    if key != "version":
      r = _assert_dict_1(ml_dict[key])
      if r:
        return r

  return None

def create_schema_pb2(column_name_type_list: List[Tuple[str, Any]]) -> schema_pb2.Schema:
  """
  create a schema_pb2.Schema for an input data file from a list of
  tuples of column_name as string and column data type.
  :param column_name_type_list:
     a list of tuples of column_name as string and column data type.
  :return: a schema_pb2.Schema usable by Executors when creating
  tf.train.Examples from PCollection.
  """
  schema = schema_pb2.Schema()

  #schema_pb2.FeatureType:
  #  TYPE_UNKNOWN = 0;
  #  BYTES = 1;
  #  INT = 2;
  #  FLOAT = 3;
  #  STRUCT = 4;
  for _name, _type in column_name_type_list:
    feature = schema.feature.add()
    feature.name = _name
    if isinstance(_type, float):
      feature.type = schema_pb2.FeatureType.FLOAT
    elif isinstance(_type, int):
      feature.type = schema_pb2.FeatureType.INT
    else:
      #str or bytes
      feature.type = schema_pb2.FeatureType.BYTES

    # You can then save this schema to a file
    # with open('my_custom_schema.pbtxt', 'w') as f:
    #     f.write(str(schema))
    """
    NOTE: if a feature has a valid domain range or vocabulary, it can
    be set by adding a domain. 
    for example:
      domain = schema.string_domain.add()
      domain.name = 'my_domain'
      domain.value.extend(['value1', 'value2', 'value3'])
    then in the feature defining stage above, add the line:
      feature.domain.name = 'my_domain'
      
    might want to further define. see:
      https://www.tensorflow.org/tfx/tf_metadata/api_docs/python/tfmd/proto/schema_pb2/Schema
    """
  return schema


# ========== partition functions ============

#from ht_generate_partition_keytps://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/components/example_gen/base_example_gen_executor.py
def _generate_partition_key(record: Union[tf.train.Example,\
  tf.train.SequenceExample, bytes, Dict[str, Any]], \
  split_config: example_gen_pb2.SplitConfig) -> bytes:
  """Generates key for partition."""

  if not split_config.HasField('partition_feature_name'):
    if isinstance(record, bytes):
      return record
    if isinstance(record, dict) or isinstance(record, list):
      return pickle.dumps(record)
    return record.SerializeToString(deterministic=True)

  if isinstance(record, tf.train.Example):
    features = record.features.feature  # pytype: disable=attribute-error
  elif isinstance(record, tf.train.SequenceExample):
    features = record.context.feature  # pytype: disable=attribute-error
  else:
    raise RuntimeError(
      'Split by `partition_feature_name` is only supported '
      'for FORMAT_TF_EXAMPLE and FORMAT_TF_SEQUENCE_EXAMPLE '
      'payload format.')

  # Use a feature for partitioning the examples.
  feature_name = split_config.partition_feature_name
  if feature_name not in features:
    raise RuntimeError(
      'Feature name `{}` does not exist.'.format(feature_name))
  feature = features[feature_name]
  if not feature.HasField('kind'):
    raise RuntimeError('Partition feature does not contain any value.')
  if (not feature.HasField('bytes_list') and
    not feature.HasField('int64_list')):
    raise RuntimeError(
      'Only `bytes_list` and `int64_list` features are '
      'supported for partition.')
  return feature.SerializeToString(deterministic=True)

#from https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/components/example_gen/base_example_gen_executor.py#L72
def partition_fn(\
    record: Union[tf.train.Example, tf.train.SequenceExample, bytes, Dict[str,Any]], \
    num_partitions: int, \
    cumulative_buckets: List[int], \
    split_config: example_gen_pb2.SplitConfig,\
) -> int:
  """Partition function for the ExampleGen's output splits."""
  assert num_partitions == len(cumulative_buckets), 'Partitions do not match bucket number.'
  partition_str = _generate_partition_key(record, split_config)
  bucket = int(hashlib.sha256(partition_str).hexdigest(), 16) % cumulative_buckets[-1]
  # For example, if buckets is [10,50,80], there will be 3 splits:
  #   bucket >=0 && < 10, returns 0
  #   bucket >=10 && < 50, returns 1
  #   bucket >=50 && < 80, returns 2
  return bisect.bisect(cumulative_buckets, bucket)

def serialize_proto_to_string(output_config : example_gen_pb2.Output) -> str:
  #return base64.b64encode(output_config.SerializeToString()).decode('utf-8')
  return base64.b64encode(output_config.SerializeToString())

def deserialize_to_proto(output_config_ser: str) -> example_gen_pb2.Output:
  new_output_config = example_gen_pb2.Input()
  #new_output_config.ParseFromString(base64.b64decode(output_config_ser.encode('utf-8')))
  new_output_config.ParseFromString(base64.b64decode(output_config_ser))
  return new_output_config

def serialize_to_string(x : Any) -> str:
  return (base64.b64encode(pickle.dumps(x))).decode('utf-8')

def deserialize(x_ser: str) -> Any:
  return pickle.loads(base64.b64decode(x_ser.encode('utf-8')))


