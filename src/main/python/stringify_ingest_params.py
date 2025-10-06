import pickle
import base64

def stringify_ingest_params(ratings_uri : str, movies_uri : str, users_uri : str, \
  ratings_key_col_dict : dict[str, int], \
  movies_key_col_dict : dict[str, int], \
  users_key_col_dict : dict[str, int], \
  partitions : list[int]) -> str:
  """
  serialize content into a string for input into ReadMergeAndSplit component.

  :param ratings_uri:
  :param movies_uri:
  :param users_uri:
  :param ratings_key_col_dict:
  :param movies_key_col_dict:
  :param users_key_col_dict:
  :param partitions:
  :return: pickle and base64 serialized string of a dictionary, which after
    deserialization will have keys:
    "ratings_uri", "ratings_uri", "ratings_uri",
    "ratings_key_dict", "movies_key_dict", "users_key_dict",
    "partitions"
  """
  params = {
    "ratings_uri" : ratings_uri, \
    "movies_uri" : movies_uri, \
    "users_uri" : users_uri, \
    "ratings_key_dict" : ratings_key_col_dict, \
    "movies_key_dict" : movies_key_col_dict, \
    "users_key_dict" : users_key_col_dict, \
    "partitions" : partitions}

  serialized = base64.b64encode(pickle.dumps(params)).decode('utf-8')

  return serialized

if __name__ == "__main__":
  # TODO: move this out of source code and into test code

  _ratings_uri = "../resources/ml-1m/ratings.dat"
  _movies_uri = "../resources/ml-1m/movies.dat"
  _users_uri = "../resources/ml-1m/users.dat"
  _ratings_key_col_dict = {"user_id":0, "movie_id":1,"rating":2}
  _movies_key_col_dict = {"movie_id":0,"genres":1}
  _users_key_col_dict = {"user_id":0,"gender":1,"age":2,\
    "occupation":3,"zipcode":4}
  _partitions=[80, 10, 10]

  input_dict_ser = stringify_ingest_params(_ratings_uri, _movies_uri, _users_uri, \
    _ratings_key_col_dict, \
    _movies_key_col_dict, \
    _users_key_col_dict, \
    _partitions)

  input_dict = pickle.loads(base64.b64decode(input_dict_ser.encode('utf-8')))

  ratings_uri = input_dict['ratings_uri']
  movies_uri = input_dict['ratings_uri']
  users_uri = input_dict['ratings_uri']

  ratings_key_dict = input_dict['ratings_key_dict']
  movies_key_dict = input_dict['movies_key_dict']
  users_key_dict = input_dict['users_key_dict']

  partitions = input_dict['partitions']

  assert(_ratings_uri == ratings_uri)
  assert (ratings_key_dict == _ratings_key_col_dict)
  assert (partitions == _partitions)