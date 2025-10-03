from typing import Any, Dict, List, Literal, Union

def make_file_dict(for_file: Literal['ratings', 'movies', 'users'], \
  uri: str, col_names: List[str], col_types : List[Any]) \
  -> Dict[str, Union[str, Dict]]:
  """
  create a dictionary for an input file.
  :param for_file: string having value 'ratings', 'movies', or 'usres'
  :param uri: a string giving the uri
  :param col_names: a list of the column names
  :param col_types: a list of the column types
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

  out = {for_file:{'cols':{}, 'uri':uri}}

  for i, name in enumerate(col_names):
    out[for_file]['cols'][name] = {'index':i, 'type': col_types[i]}

  return out

def merge_dicts(ratings_dict: Dict[str, Union[str, Dict]], \
  movies_dict: Dict[str, Union[str, Dict]], \
  users_dict: Dict[str, Union[str, Dict]]) -> Dict[str, Union[str, Dict]]:

  return {**ratings_dict, **movies_dict, **users_dict}
