from tfx.proto import example_gen_pb2
from typing import Tuple

from movie_lens_utils import *

def get_test_data(use_small=True, kaggle=True) -> Tuple[str, str, list[str]]:
  """
  :param use_small:
  :param kaggle:
  :return: Tuple of infiles_dict serialized to string,
     output_config serialized to string, and list of split names
  """

  if kaggle:
    prefix = '/kaggle/working/ml-1m/'
  else:
    prefix = "../resources/ml-1m/"
  if use_small:
    ratings_uri = f"{prefix}ratings_1000.dat"
    users_uri = f"{prefix}users_100.dat"
  else:
    ratings_uri = f"{prefix}ratings.dat"
    users_uri = f"{prefix}users.dat"
  movies_uri = f"{prefix}movies.dat"

  ratings_col_names = ["user_id", "movie_id", "rating", "timestamp"]
  ratings_col_types = [int, int, int, int]  # for some files, ratings are floats
  movies_col_names = ["movie_id", "title", "genres"]
  movies_col_types = [int, str, str]
  users_col_names = ["user_id", "gender", "age", "occupation",
                     "zipcode"]
  users_col_types = [int, str, int, int, str]

  ratings_dict = create_infile_dict(for_file='ratings',
                                    uri=ratings_uri,
                                    col_names=ratings_col_names,
                                    col_types=ratings_col_types,
                                    headers_present=False, delim="::")

  movies_dict = create_infile_dict(for_file='movies',
                                   uri=movies_uri,
                                   col_names=movies_col_names,
                                   col_types=movies_col_types,
                                   headers_present=False, delim="::")

  users_dict = create_infile_dict(for_file='users',
                                  uri=users_uri,
                                  col_names=users_col_names,
                                  col_types=users_col_types,
                                  headers_present=False, delim="::")

  infiles_dict = create_infiles_dict(ratings_dict=ratings_dict,
                                          movies_dict=movies_dict,
                                          users_dict=users_dict,
                                          version=1)
  infiles_dict_ser = serialize_to_string(infiles_dict)

  buckets = [80, 10, 10]
  split_names = ['train', 'eval', 'test']
  output_config = example_gen_pb2.Output(
    split_config=example_gen_pb2.SplitConfig(
      splits=[example_gen_pb2.SplitConfig.Split(name=n, hash_buckets=b) \
        for n, b in zip(split_names, buckets)]
    )
  )
  output_config_ser = serialize_proto_to_string(output_config)
  return infiles_dict_ser, output_config_ser, split_names