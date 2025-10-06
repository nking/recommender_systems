import apache_beam as beam
#from apache_beam.coders import coders

import time

from typing import Any, Dict, List, Union, Tuple
from movie_lens_utils import *
#from apache_beam.pvalue import TaggedOutput

from CustomUTF8Coder import CustomUTF8Coder

@beam.typehints.with_input_types(beam.pvalue.PCollection)
@beam.typehints.with_output_types(beam.pvalue.PCollection)
class LeftJoinFn(beam.DoFn):
  """
  left join of left PCollection rows with right PCollection row.

  if there is  more than one row in the right, a ValueError is thrown.

  :return returns merged rows
  """
  def __init__(self, right_filter_cols):
    super().__init__()
    self.right_filter_cols = right_filter_cols

  def process(self, kv):
    key, grouped_elements = kv
    # grouped_elements is a dictionary with keys 'left' and 'right'
    # both are lists of lists.
    if len(grouped_elements['right']) != 1:
      raise ValueError(f"in join, right list length != 1: key={key}, grouped_elements={grouped_elements}")

    right_row = grouped_elements['right'][0]

    #does nothing if this is a right join with no left element
    for left in grouped_elements['left']:
      # join grouped_elements['left'] and grouped_elements['right'][0]
      # merge, reorder etc. as wanted
      row = left.copy()
      for i, right in enumerate(right_row):
        if i not in self.right_filter_cols:
          row.append(right)
      yield row

#@beam.ptransform_fn
#@beam.typehints.with_input_types(beam.pvalue.PCollection,\
#  beam.pvalue.PCollection, Dict[str, int], Dict[str, int], List[int])
#@beam.typehints.with_output_types(Dict[str, beam.pvalue.PCollection])
def merge_by_key(l_pc : beam.pvalue.PCollection, r_pc : beam.pvalue.PCollection, \
  l_key_col : int, r_key_col : int, \
  filter_cols : List[int], debug_tag : str = "") -> beam.pvalue.PCollection:
  """
  merges PCollection l_pc with PCollection r_pc on the columns given by
  l_key_col and r_key_col.  While merging, it excludes any columns
  in the right dataset given by filter_cols.

  if there is an error in columns or more than one row in r_pc, a
  ValueError is thrown.

  :param l_pc:
  :param r_pc:
  :param l_key_col:
  :param r_key_col:
  :param filter_cols:
  :param debug_tag:
  :return: a merged PCollection
  """
  # need unique names for each beam process, so adding a timestamp
  ts = time.time_ns()
  l_keyed = l_pc | f'kv_l_{ts}' >> beam.Map(lambda x: (x[l_key_col], x))
  r_keyed = r_pc | f'kv_r_{ts}' >> beam.Map(lambda x: (x[r_key_col], x))

  #l_keyed | f'Left keyed: {time.time_ns()}' >> \
  #  beam.Map(lambda x: print(f'{debug_tag} l_key_col={l_key_col}, row={x}'))
  # r_keyed | f'Right keyed: {time.time_ns()}' >> 'beam.Map(print)

  # multiple lefts on one line, and one in right's list:
  grouped_data = ({'left': l_keyed, 'right': r_keyed} \
                  | f'group_by_key_{ts}' >> beam.CoGroupByKey())

  #grouped_data | f'{time.time_ns()}' >> beam.Map(lambda x: print(f'{debug_tag}:::{x}'))

  joined_data = grouped_data \
      | f'left_join_values_{ts}' >> beam.ParDo(LeftJoinFn(filter_cols))

  return joined_data

@beam.ptransform_fn
@beam.typehints.with_input_types(beam.pvalue.Pipeline, Dict[str, Union[str, Dict]])
@beam.typehints.with_output_types(Dict[str, beam.pvalue.PCollection])
def _read_files(pipeline : beam.pvalue.Pipeline, \
  infiles_dict: Dict[str, Union[str, Dict]]) -> \
  Dict[str, beam.pvalue.PCollection]:
  pc = {}

  for key in ['ratings', 'movies', 'users']:
    if infiles_dict[key]['headers_present']:
      skip = 1
    else:
      skip = 0
    pc[key] = pipeline | f"read_{key}_{time.time_ns()}" >> beam.io.ReadFromText(\
      infiles_dict[key]['uri'], skip_header_lines=skip, coder=CustomUTF8Coder()) \
      | f'parse_{key}_{time.time_ns()}' >> beam.Map(lambda line: line.split(infiles_dict[key]['delim']))
  return pc

@beam.ptransform_fn
@beam.typehints.with_input_types(beam.pvalue.PCollection, List[Tuple[str, Any]])
@beam.typehints.with_output_types(tf.train.Example)
def convert_to_tf_example(pcollection: beam.PCollection, column_name_type_list) -> beam.PCollection:
    return pcollection | f'ToTFExample {time.time_ns()}' >> beam.Map(create_example, column_name_type_list)

@beam.ptransform_fn
@beam.typehints.with_input_types(beam.pvalue.PCollection, str, str, str)
@beam.typehints.with_output_types(None)
def write_to_csv(pcollection : beam.pvalue.PCollection, \
  column_names : str, prefix_path:str, delim:str='_') -> None:
  # format the lines into a delimiter separated string then write to
  # file
  pcollection | f"format_for_writing {time.time_ns()}" >> beam.Map( \
    lambda x: delim.join(x)) \
    | f"write to text {time.time_ns()}"  >> beam.io.WriteToText( \
    file_path_prefix=prefix_path, file_name_suffix='.csv', \
    header=column_names)

@beam.ptransform_fn
@beam.typehints.with_input_types(beam.pvalue.Pipeline, Dict[str, Union[str, Dict]])
@beam.typehints.with_output_types(Tuple[beam.pvalue.PCollection, List[Tuple[str, Any]]])
def ingest_and_join( \
  pipeline : beam.pvalue.Pipeline, \
  infiles_dict: Dict[str, Union[str, Dict]]) -> \
  Tuple[beam.pvalue.PCollection, List[Tuple[str, Any]]]:
  """
  reads in the 3 expected files from the uris given in infiles_dict, and then uses
  left joins of ratings with user information and movie genres to
  make a PCollection.  The PCollection has an associated schema in it.

  :param pipeline:
  :param infiles_dict
    a dictionary of file information for each of the 3 files, that is
    the ratings file, movies file, and users file.
    the dictionary is made by using movie_lens_utils.create_infile_dict
    for each file, then movie_lens_utils.create_infiles_dict.
  :return: a tuple of PCollection of ratings with joined information from users and movies where each tuple is for a
     partition specified in partition list.  The PCollection has an associated schema
  """

  err = infiles_dict_formedness_error(infiles_dict)
  if err:
    raise ValueError(err)

  pc = _read_files(pipeline, infiles_dict)

  # ratings: user_id,movie_id,rating
  # movie_id,title,genre
  # user_id::gender::age::occupation::zipcode

  # user_id,movie_id,rating,timestamp,geder,age,occupation,zipcode
  ratings_1 = merge_by_key( pc['ratings'], pc['users'], \
    infiles_dict['ratings']['cols']['user_id']['index'], \
    infiles_dict['users']['cols']['user_id']['index'], \
    filter_cols=[infiles_dict['users']['cols']['zipcode']['index'],\
    infiles_dict['users']['cols']['user_id']['index']], debug_tag="R-U")

  # user_id,movie_id,rating,gender,age,occupation,zipcode,genres
  ratings = merge_by_key(ratings_1, pc['movies'], \
    infiles_dict['ratings']['cols']['movie_id']['index'], \
    infiles_dict['movies']['cols']['movie_id']['index'], \
    filter_cols=[infiles_dict['movies']['cols']['title']['index'], \
    infiles_dict['movies']['cols']['movie_id']['index']], debug_tag="R-M")

  schemas = create_namedtuple_schemas(infiles_dict)
  #format, compatible with apache_beam.coders.registry.
  # list of tuples of column_name, column_type
  columns = schemas['ratings'].copy()
  #append users, skipping infiles_dict['users']['cols']['zipcode']
  for _name, _type in schemas['users']:
    if _name != 'zipcode' and _name != 'user_id':
      columns.append((_name, _type))
  # append movies, skipping title
  for _name, _type in schemas['movies']:
    if _name != 'title' and _name != 'movie_id':
      columns.append((_name, _type))

  # for i, part in enumerate(ratings_parts):
  #  part | f'PARTITIONED_{i}_{time.time_ns()}' >> beam.io.WriteToText(\
  #    file_path_prefix=f'a_{i}_', file_name_suffix='.txt')

  return ratings, columns
