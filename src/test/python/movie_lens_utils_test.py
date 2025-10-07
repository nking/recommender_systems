import unittest
from typing import  Dict, Union
#from ... main.python.infile_dict_util import *
#from ... main.python.infile_dict_util import _assert_dict_1
from movie_lens_utils import *
from movie_lens_utils import _assert_dict_1
import random

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

  def test_make_file_dict(self):
    kaggle = True
    if kaggle:
      prefix = '/kaggle/working/ml-1m/'
    else:
      prefix = "../resources/ml-1m/"
    ratings_uri = f"{prefix}ratings.dat"
    movies_uri = f"{prefix}movies.dat"
    users_uri = f"{prefix}users.dat"

    ratings_col_names = ["user_id", "movie_id", "rating"]
    ratings_col_types = [int, int, int] #for some files, ratings are floats
    movies_col_names = ["movie_id", "title", "genres"]
    movies_col_types = [int, str, str]
    users_col_names = ["user_id", "gender", "age",  "occupation", "zipcode"]
    users_col_types = [int, str, int, int, str]

    ratings_dict = create_infile_dict(for_file='ratings',\
      uri=ratings_uri, col_names=ratings_col_names, \
      col_types=ratings_col_types, headers_present=False, delim="::")

    self._assert_dict_content(ratings_dict)

    movies_dict = create_infile_dict(for_file='movies', \
      uri=movies_uri, col_names=movies_col_names, \
      col_types=movies_col_types, headers_present=False, delim="::")

    self._assert_dict_content(movies_dict)

    users_dict = create_infile_dict(for_file='users', \
      uri=users_uri,  col_names=users_col_names, \
      col_types=users_col_types, headers_present=False, delim="::")

    self._assert_dict_content(users_dict)

    self.infiles_dict = create_infiles_dict(ratings_dict=ratings_dict, \
                              movies_dict=movies_dict, \
                              users_dict=users_dict, version=1)

    self._assert_merged_dict_content(self.infiles_dict)

    schema_dict = create_namedtuple_schemas(self.infiles_dict)
    for k in schema_dict:
      self.assertTrue(k in ['ratings', 'movies', 'users'], f"key {k} not recognized")
      file_cols = schema_dict[k]
      # should be a list of tuples of (column_name, column_type)
      for _name, _type in file_cols:
        self.assertTrue(isinstance(_name, str))
        self.assertIsNotNone(_type)

if __name__ == '__main__':
    unittest.main()
