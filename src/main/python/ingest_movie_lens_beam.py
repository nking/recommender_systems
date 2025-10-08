import apache_beam as beam
#from apache_beam.coders import coders

import time
import random

from typing import Any, Dict, List, Union, Tuple
from movie_lens_utils import *
#from apache_beam.pvalue import TaggedOutput

from CustomUTF8Coder import CustomUTF8Coder

import absl
from absl import logging
absl.logging.set_verbosity(absl.logging.DEBUG)

class LeftJoinFn(beam.DoFn):
  """
  left join of left PCollection rows with right PCollection row.

  if there is  more than one row in the right, a ValueError is thrown.

  :return returns merged rows
  """
  def __init__(self, right_filter_cols):
    super().__init__()
    self.right_filter_cols = right_filter_cols

  #TODO: consider also implementing def process_batch
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

class ReadFiles(beam.PTransform):
  """
  read ratings, movies, and users independently
  """
  def __init__(self, infiles_dict):
    super().__init__()
    self.infiles_dict = infiles_dict

  def expand(self, pcoll=None):
    pc = {}
    for key in ['ratings', 'movies', 'users']:
      if self.infiles_dict[key]['headers_present']:
        skip = 1
      else:
        skip = 0
      # class 'apache_beam.transforms.ptransform._ChainedPTransform
      pc[key] = pcoll | f"r{random.randint(0,10000000000)}" >> \
        beam.io.ReadFromText(\
        self.infiles_dict[key]['uri'], skip_header_lines=skip,
        coder=CustomUTF8Coder()) \
        | f'parse_{key}_{random.randint(0, 1000000000000)}' >> \
        beam.Map(lambda line: line.split(self.infiles_dict[key]['delim']))
    return pc

@beam.ptransform_fn
@beam.typehints.with_input_types(beam.PCollection, List[Tuple[str, Any]])
@beam.typehints.with_output_types(tf.train.Example)
def convert_to_tf_example(pcollection: beam.PCollection, column_name_type_list) -> beam.PCollection:
    return pcollection | f'ToTFExample_{random.randint(0, 1000000000000)}' >> beam.Map(create_example, column_name_type_list)

@beam.ptransform_fn
@beam.typehints.with_input_types(beam.PCollection, str, str, str)
@beam.typehints.with_output_types(None)
def write_to_csv(pcollection : beam.PCollection, \
  column_names : str, prefix_path:str, delim:str='_') -> None:
  # format the lines into a delimiter separated string then write to
  # file
  pcollection | f"format_for_writing_{random.randint(0, 1000000000000)}" >> beam.Map( \
    lambda x: delim.join(x)) \
    | f"write_to_text_{random.randint(0, 1000000000000)}"  >> beam.io.WriteToText( \
    file_path_prefix=prefix_path, file_name_suffix='.csv', \
    header=column_names)

class IngestAndJoin(beam.PTransform):
  """
  reads in the 3 expected files from the uris given in infiles_dict, and then
  left joins on ratings with user information and movie genres to
  make a PCollection.  The PCollection has an associated schema in it.

  :param infiles_dict
    a dictionary of file information for each of the 3 files, that is
    the ratings file, movies file, and users file.
    the dictionary is made by using movie_lens_utils.create_infile_dict
    for each file, then movie_lens_utils.create_infiles_dict.
  :return: a tuple of PCollection of ratings with joined information from users and movies where each tuple is for a
     partition specified in partition list.  The PCollection has an associated schema
  """
  def __init__(self, infiles_dict):
    super().__init__()
    self.infiles_dict = infiles_dict

  def expand(self, pcoll=None):
    err = infiles_dict_formedness_error(self.infiles_dict)
    if err:
      raise ValueError(err)

    pc = pcoll | f"read_{random.randint(0, 1000000000000)}" >> ReadFiles(self.infiles_dict)

    # ratings: user_id,movie_id,rating
    # movie_id,title,genre
    # user_id::gender::age::occupation::zipcode
    l_keyed_1 = pc['ratings'] | f'kv_l_RU_{random.randint(0, 1000000000)}' \
      >> beam.Map(lambda x: (x[self.infiles_dict['ratings']['cols']['user_id']['index']], x))
    r_keyed_1 = pc['users'] | f'kv_r_RU_{random.randint(0, 1000000000)}' \
      >> beam.Map(lambda x: (x[self.infiles_dict['users']['cols']['user_id']['index']], x))

    # multiple lefts on one line, and one in right's list:
    grouped_data_1 = ({'left': l_keyed_1, 'right': r_keyed_1} \
      | f'group_by_key_1_{random.randint(0, 1000000000)}' \
      >> beam.CoGroupByKey())

    try:
      ratings_1 = grouped_data_1 \
        | f'left_join_values_1_{random.randint(0, 1000000000)}' \
        >> beam.ParDo(LeftJoinFn(\
        [self.infiles_dict['users']['cols']['zipcode']['index'],\
        self.infiles_dict['users']['cols']['user_id']['index']]))
    except Exception as ex:
      logging.error("ERROR for R-U")
      raise ex

    # user_id,movie_id,rating,gender,age,occupation,zipcode,genres
    l_keyed_2 = ratings_1 | f'kv_l_RM_{random.randint(0, 1000000000)}' \
      >> beam.Map(lambda x: (x[self.infiles_dict['ratings']['cols']['movie_id']['index']], x))
    r_keyed_2 = pc['movies'] | f'kv_r_RM_{random.randint(0, 1000000000)}' \
      >> beam.Map(lambda x: (x[self.infiles_dict['users']['cols']['movie_id']['index']], x))

    # multiple lefts on one line, and one in right's list:
    grouped_data_2 = ({'left': l_keyed_2, 'right': r_keyed_2} \
      | f'group_by_key_2_{random.randint(0, 1000000000)}' \
      >> beam.CoGroupByKey())

    try:
      ratings = grouped_data_2 \
        | f'left_join_values_2_{random.randint(0, 1000000000)}' \
        >> beam.ParDo(LeftJoinFn(\
        [self.infiles_dict['movies']['cols']['title']['index'],\
        self.infiles_dict['movies']['cols']['movie_id']['index']]))
    except Exception as ex:
      logging.error("ERROR for R-M")
      raise ex

    schemas = create_namedtuple_schemas(self.infiles_dict)
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

    logging.debug(f"columns={columns}")

    return ratings, columns
