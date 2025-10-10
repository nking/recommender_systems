import unittest
from typing import  Dict, Union
#from ... main.python.infile_dict_util import *
#from ... main.python.infile_dict_util import _assert_dict_1
from movie_lens_utils import *
from movie_lens_utils import _assert_dict_1
import random
from tfx.proto import example_gen_pb2

import absl
from absl import logging
absl.logging.set_verbosity(absl.logging.DEBUG)

class TestInfileDictUtils(unittest.TestCase):

  def _assert_dict_content(self, ml_dict: Dict[str, Union[str, Dict]]) -> None:
    key = None
    for k in ml_dict:
      if k in ['ratings', 'movies', 'users']:
        key = k
        break
    self.assertIsNotNone(key, "dictionary does not contain one of expected keys: 'ratings', 'movies', 'users'")
    r = _assert_dict_1(ml_dict[key])
    self.assertIsNone(r, r)

  def _assert_merged_dict_content(self, merged_dict: Dict[str, Union[str, Dict]]) -> None:
    r = infiles_dict_formedness_error(merged_dict)
    self.assertIsNone(r, r)

  def make_infiles(self):
    kaggle = True
    if kaggle:
      prefix = '/kaggle/working/ml-1m/'
    else:
      prefix = "../resources/ml-1m/"
    ratings_uri = f"{prefix}ratings.dat"
    movies_uri = f"{prefix}movies.dat"
    users_uri = f"{prefix}users.dat"

    ratings_col_names = ["user_id", "movie_id", "rating", "timestamp"]
    ratings_col_types = [int, int, int, int] #for some files, ratings are floats
    movies_col_names = ["movie_id", "title", "genres"]
    movies_col_types = [int, str, str]
    users_col_names = ["user_id", "gender", "age",  "occupation", "zipcode"]
    users_col_types = [int, str, int, int, str]

    ratings_dict = create_infile_dict(for_file='ratings',\
      uri=ratings_uri, col_names=ratings_col_names, \
      col_types=ratings_col_types, headers_present=False, delim="::")

    movies_dict = create_infile_dict(for_file='movies', \
      uri=movies_uri, col_names=movies_col_names, \
      col_types=movies_col_types, headers_present=False, delim="::")

    users_dict = create_infile_dict(for_file='users', \
      uri=users_uri,  col_names=users_col_names, \
      col_types=users_col_types, headers_present=False, delim="::")

    infiles_dict = create_infiles_dict(ratings_dict=ratings_dict, \
                              movies_dict=movies_dict, \
                              users_dict=users_dict, version=1)

    return ratings_dict, movies_dict, users_dict, infiles_dict

  def test_make_file_dict(self):
    ratings_dict, movies_dict, users_dict, infiles_dict = \
      self.make_infiles()

    self._assert_dict_content(ratings_dict)

    self._assert_dict_content(movies_dict)

    self._assert_dict_content(users_dict)

    self.infiles_dict = infiles_dict

    self._assert_merged_dict_content(self.infiles_dict)

    schema_dict = create_namedtuple_schemas(self.infiles_dict)
    for k in schema_dict:
      self.assertTrue(k in ['ratings', 'movies', 'users'], f"key {k} not recognized")
      file_cols = schema_dict[k]
      # should be a list of tuples of (column_name, column_type)
      for _name, _type in file_cols:
        self.assertTrue(isinstance(_name, str))
        self.assertIsNotNone(_type)

  def test_prot_ser_deser(self):
    buckets = [80, 10, 10]
    bucket_names = ['train', 'eval', 'test']
    output_config = example_gen_pb2.Output(
      split_config=example_gen_pb2.SplitConfig(
        splits=[
          example_gen_pb2.SplitConfig.Split(name=n, hash_buckets=b) \
          for n, b in zip(bucket_names, buckets)]
      )
    )
    output_config_ser = serialize_proto_to_string(output_config)
    deser = deserialize_to_proto(output_config_ser)
    self.assertIsNotNone(deser)
    self.assertNotEqual(deser, "")
    #logging.debug(f"\noutput_config={output_config}\noutput_config_ser={output_config_ser}\ndeser={deser}")
    #protocol buffers deserialization is not deterministic
    #self.assertEqual(output_config, deser, \
    #  f"output_config and deser should be same"
    #  f"\noutput_config={output_config}\noutput_config_ser={output_config_ser}\ndeser={deser}")

  def test_ser_deser(self):
    ratings_dict, movies_dict, users_dict, infiles_dict = \
      self.make_infiles()
    infiles_dict_ser = serialize_to_string(infiles_dict)
    deser = deserialize(infiles_dict_ser)
    self.assertEqual(infiles_dict, deser, \
      "infiles_dict and deser should be same")
 
if __name__ == '__main__':
    unittest.main()
