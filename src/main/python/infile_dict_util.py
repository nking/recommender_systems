from typing import Any, Dict, List, Literal, Union, Tuple

def make_file_dict(for_file: Literal['ratings', 'movies', 'users'], \
  uri: str, col_names: List[str], col_types : List[Any], \
  headers_present:bool, delim:str) -> Dict[str, Union[str, Dict]]:
  """
  create a dictionary for an input file.
  :param delim: string of delimiter between the columns of a row
  :param headers_present: True if first line is header line, else False
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

  out = {for_file:{'cols':{}, 'uri':uri, 'headers_present':headers_present, \
    'delim':delim}}

  for i, name in enumerate(col_names):
    out[for_file]['cols'][name] = {'index':i, 'type': col_types[i]}

  return out

def merge_dicts(ratings_dict: Dict[str, Union[str, Dict]], \
  movies_dict: Dict[str, Union[str, Dict]], \
  users_dict: Dict[str, Union[str, Dict]]) -> Dict[str, Union[str, Dict]]:
  """
  merge the 3 dictionaries, each created by make_file_dict, into single
  output dictionary
  :param ratings_dict:
  :param movies_dict:
  :param users_dict:
  :return: a merge of the 3 dictionaries
  """

  return {**ratings_dict, **movies_dict, **users_dict}

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

def create_namedtuple_schemas(infiles_dict: Dict[str, Union[str, Dict]]) -> Dict[str, List[Tuple]]:
  """
  from a dictionary created with merge_dicts, create a dictionary of
  of lists of tuples of column names and types that can be used as a
  schema for a PCollection for each top level key
  :param infiles_dict:
  :return: a dictionary of list of tuples of column name and types,
  useable with coders.registry.register_coder
  """
  out = {}
  for key in infiles_dict:
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

def dict_formedness_error(ml_dict: Dict[str, Union[str, Dict]]) -> Union[str, None]:
  """
  verify expected structure of ml_dict, and return None if correct else
  error message if not
  :param ml_dict: dictionary of information for the ratings, movies, and
    users files.
  :return: None if ml_dict structure is as expected, else return error
    message
  """
  for key in ml_dict:
    if key not in ['ratings', 'movies', 'users']:
      return f"key expected to be one of 'ratings', 'movies', 'users', but is {key}"
    r = _assert_dict_1(ml_dict[key])
    if r:
      return r

  return None

