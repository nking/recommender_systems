#contains tf import:

from movie_lens_tfx.utils.movie_lens_utils import *

def get_kaggle() -> bool:
  cwd = os.getcwd()
  if "recommender_systems" in cwd:
    kaggle = False
  else:
    kaggle = True
  return kaggle

def get_project_dir() -> str:
  if get_kaggle():
    return "/kaggle/working"
  cwd = os.getcwd()
  head = cwd
  proj_dir = ""
  while head and head != os.sep:
    head, tail = os.path.split(head)
    if tail:  # Add only if not an empty string (e.g., from root or multiple separators)
      if tail == "recommender_systems":
        proj_dir = os.path.join(head, tail)
        break
  return proj_dir

def get_bin_dir() -> str:
  return os.path.join(get_project_dir(), "bin")


def add_to_sys(proj_dir):
  src_module_dir = os.path.join(proj_dir, "src/main/python")
  #sys.path.insert(0, self.module_dir)


def get_test_data(use_small=True) -> Tuple[str, str, list[str]]:
  """
  :param use_small:
  :param kaggle:
  :return: Tuple of infiles_dict serialized to string,
     output_config serialized to string, and list of split names
  """

  kaggle = get_kaggle()
  print(f"CWD={os.getcwd()}, kaggle={kaggle}")

  if kaggle:
    prefix = '/kaggle/working/ml-1m/'
    if use_small:
      ratings_uri = os.path.join(prefix, "ratings_1000.dat")
      users_uri = os.path.join(prefix, "users_100.dat")
    else:
      ratings_uri = os.path.join(prefix,"ratings.dat")
      users_uri = os.path.join(prefix, "users.dat")
    movies_uri = os.path.join(prefix, "movies.dat")
  else:
    proj_dir = get_project_dir()
    prefix_main = os.path.join(proj_dir, "src/main/resources/ml-1m/")
    prefix = os.path.join(proj_dir, "src/test/resources/ml-1m/")
    if use_small:
      ratings_uri = os.path.join(prefix, "ratings_1000.dat")
      users_uri = os.path.join(prefix, "users_100.dat")
    else:
      ratings_uri = os.path.join(prefix_main,"ratings.dat")
      users_uri = os.path.join(prefix_main, "users.dat")
    movies_uri = os.path.join(prefix_main, "movies.dat")
    add_to_sys(proj_dir)

  ratings_col_names = ["user_id", "movie_id", "rating", "timestamp"]
  ratings_col_types = [int, int, int, int]  # for some files, ratings are floats
  movies_col_names = ["movie_id", "title", "genres"]
  movies_col_types = [int, str, str]
  users_col_names = ["user_id", "gender", "age", "occupation", "zipcode"]
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

def get_expected_col_name_feature_types() -> Dict[str, tf.train.Feature]:
  return {"user_id": tf.train.Int64List, "movie_id":tf.train.Int64List,
    "rating" : tf.train.Int64List, "timestamp" : tf.train.Int64List,
    "gender" : tf.train.BytesList,
    "age" : tf.train.Int64List, "occupation" : tf.train.Int64List,
    "genres" : tf.train.BytesList}

def get_expected_col_name_feature_types2():
  return {"user_id": tf.io.FixedLenFeature([], tf.int64),
    "movie_id":tf.io.FixedLenFeature([], tf.int64),
    "rating" : tf.io.FixedLenFeature([], tf.int64),
    "timestamp": tf.io.FixedLenFeature([], tf.int64),
    "gender" : tf.io.FixedLenFeature([], tf.string),
    "age" : tf.io.FixedLenFeature([], tf.int64),
    "occupation" : tf.io.FixedLenFeature([], tf.int64),
    "genres" : tf.io.FixedLenFeature([], tf.string)}

#user_id, movie_id, rating, timestamp, gender, age, occupation, zipcode, genres
 # int,    int      int     str    int  int       str     str
