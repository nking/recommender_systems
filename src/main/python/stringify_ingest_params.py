import json

def stringify_ingest_params(ratings_uri : str, movies_uri : str, users_uri : str, \
  ratings_key_col_dict : dict[str, int], \
  movies_key_col_dict : dict[str, int], \
  users_key_col_dict : dict[str, int], \
  partitions : list[int]) -> str:
  '''
  serialize content into a string for input into ReadMergeAndSplit component.

  :param ratings_uri:
  :param movies_uri:
  :param users_uri:
  :param ratings_key_col_dict:
  :param movies_key_col_dict:
  :param users_key_col_dict:
  :param partitions:
  :return: json serialized string of a dictionary, which after
    json.loads will have keys:
    "ratings.dat", "movies.dat", "users.dat",
    "ratings_key_dict", "movies_key_dict", "users_key_dict",
    "partitions"
  '''
  params = {"ratings.dat" : ratings_uri, \
    "movies.dat" : movies_uri, \
    "users.dat" : users_uri, \
    "ratings_key_dict" : ratings_key_col_dict, \
    "movies_key_dict" : movies_key_col_dict, \
    "users_key_dict" : users_key_col_dict, \
    "partitions" : partitions}

  serialized = json.dumps(params)

  return serialized

if __name__ == "__main__":
  _ratings_uri = "ratings.dat"
  _movies_uri = "movies.dat"
  _users_uri = "users.dat"
  _ratings_key_col_dict = {"user_id":0, "movie_id":1,"rating":2}
  _movies_key_col_dict = {"movie_id":0,"genres":1}
  _users_key_col_dict = {"user_id":0,"gender":1,"age":2,\
    "occupation":3,"zipcode":4}
  _partitions=[80, 10, 10]

  input_dict_ser = stringify(_ratings_uri, _movies_uri, _users_uri, \
    _ratings_key_col_dict, \
    _movies_key_col_dict, \
    _users_key_col_dict, \
    _partitions)

  input_dict = json.loads(input_dict_ser)

  ratings_uri = input_dict['ratings.dat']
  movies_uri = input_dict['movies.dat']
  users_uri = input_dict['users.dat']

  ratings_key_dict = input_dict['ratings_key_dict']
  movies_key_dict = input_dict['movies_key_dict']
  users_key_dict = input_dict['users_key_dict']

  partitions = input_dict['partitions']

  assert(_ratings_uri == ratings_uri)
  assert (ratings_key_dict == _ratings_key_col_dict)
  assert (partitions == _partitions)