import unittest
from typing import  Dict, Union
#from ... main.python.infile_dict_util import *
from infile_dict_util import *

class TestInfileDictUtils(unittest.TestCase):

  def _assert_dict_1(self, ml_dict: Dict) -> None:
    """test contents of dict[key]"""
    self.assertTrue('cols' in ml_dict, f"dictionary is missing ['cols']")

    self.assertTrue('uri' in ml_dict, "dictionary is missing key ['uri']")
    self.assertIsNotNone(ml_dict['uri'], "dictionary is missing value for ['uri']")

    for name in ml_dict['cols']:
      self.assertTrue('index' in ml_dict['cols'][name], \
        f"missing ml_dict[key]['cols'][{name}]['index']")
      self.assertTrue('index' in ml_dict['cols'][name], \
        f"missing ml_dict[key]['cols'][{name}]['type']")

  def _assert_dict_content(self, ml_dict: Dict[str, Union[str, Dict]]) -> None:
    key = None
    for k in ml_dict:
      if k in ['ratings', 'movies', 'users']:
        key = k
        break
    self.assertIsNotNone(key, f"dictionary does not contain one of expected keys: {'ratings', 'movies', 'users'}")
    self._assert_dict_1(ml_dict)

  def _assert_merged_dict_content(self,
    merged_dict: Dict[str, Union[str, Dict]]) -> None:
    for key in merged_dict:
      self.assertTrue(key in ['ratings', 'movies', 'users'], \
        f"key expected to be one of 'ratings', 'movies', 'users', but is {key}")
      self._assert_dict_1(merged_dict[key])

  def make_file_dict_test(self):
    ratings_uri = "../resources/ml-1m/ratings.dat"
    movies_uri = "../resources/ml-1m/movies.dat"
    users_uri = "../resources/ml-1m/users.dat"
    ratings_col_names = ["user_id", "movie_id", "rating"]
    ratings_col_types = [int, int, int] #for some files, ratings are floats
    movies_col_names = ["movie_id", "title", "genres"]
    movies_col_types = [int, str, str]
    users_col_names = ["user_id", "gender", "age",  "occupation", "zipcode"]
    users_col_types = [int, str, int, int, str]

    ratings_dict = make_file_dict(for_file='ratings',\
      uri=ratings_uri, col_names=ratings_col_names, \
      col_types=ratings_col_types)

    self._assert_dict_content(ratings_dict)

    movies_dict = make_file_dict(for_file='movies', \
      uri=movies_uri, col_names=movies_col_names, \
      col_types=movies_col_types)

    self._assert_dict_content(movies_dict)

    users_dict = make_file_dict(for_file='users', \
      uri=users_uri,  col_names=users_col_names, \
      col_types=users_col_types)

    self._assert_dict_content(users_dict)

    merged_dict = merge_dicts(ratings_dict=ratings_dict, \
                              movies_dict=movies_dict, \
                              users_dict=users_dict)

    self._assert_merged_dict_content(merged_dict)

if __name__ == '__main__':
    unittest.main()
