import apache_beam as beam
import pyarrow as pa
from apache_beam.io import parquetio

"""
offers an IngestAndJoin which reads Parquet files into pyarrow tables,
performs a left inner join on ratings with users and movies.
"""

import os
#from apache_beam.coders import coders

#TODO: replace use of random in labels for PTransform with unique
#  sequential numbers

import random
import time

from typing import Any, List, Tuple
from movie_lens_utils import *
#from apache_beam.pvalue import TaggedOutput

from CustomUTF8Coder import CustomUTF8Coder

import absl
from absl import logging
logging.set_verbosity(logging.WARNING)
logging.set_stderrthreshold(logging.WARNING)

class ReadCSVToRecords(beam.PTransform):
  def __init__(self, infiles_dict):
    super().__init__()
    self.infiles_dict = infiles_dict

  def line_to_record(self, line, infile_dict, pa_schema):
    items = line.split(infile_dict['delim'])
    out = {}
    for i in range(len(items)):
      if pa_schema[i][1] == pa.int64():
        out[pa_schema[i][0]] = int(items[i])
      elif pa_schema[i][1] == pa.float64():
        out[pa_schema[i][0]] = float(items[i])
      else:
        out[pa_schema[i][0]] = items[i]
    return out

class WriteParquet(beam.PTransform):
  def __init__(self, infile_dict, file_path_prefix):
    """
     write parquet records to the given file_path_prefix.
     The infile_dict is used to create the pyarrow schema.
    :param infile_dict: the dictionary created
      from movie_lens_utils.create_infile_dict
    :param file_path_prefix: absolute path to file prefix, that is an
       absolute pathe to a directory joined to the file name prefix
    """
    super().__init__()
    self.infile_dict = infile_dict
    self.file_path_prefix = file_path_prefix

  def expand(self, pcoll=None):
    pa_schema_list = create_pa_schema(self.infile_dict)
    logging.debug(f'file_path_prefix={self.file_path_prefix}')
    pa_schema = pa.schema(pa_schema_list)
    logging.debug(f'pa_schema={pa_schema}')
    pcoll \
      | f'write_to_parquet_{time.time_ns()}' \
      >> parquetio.WriteToParquet(\
      file_path_prefix=self.file_path_prefix,\
      schema=pa_schema, file_name_suffix='.parquet')

class WriteJoinedRatingsParquet(beam.PTransform):
  def __init__(self, file_path_prefix, column_name_type_list):
    """
    write the joined ratings file to a parquet file given the
    file_path_prefix and the column_name_type_list returned by
    ingest_movie_lens_beam_pa.IngestAndJoin.
    column_name_type_list is used for creating a pyarrow schema
    for parquetio.WriteToParquetBatched.
    The joined ratings is a PCollection of pyarraow table and so has a schema
    for the first, etc. but parquetio.WriteToParquetBatched input schema
    must be deterministic, and cannot be derived from PCollection at runtime.
    So this method is here to write the pa.schema from the list of tuples
    of column names and types that was created after the ratings left
    inner joins... details so I don't forget and try to use an extracted
    schema from the ratings PCollection again...

    :param file_path_prefix:
    :param column_name_type_list:
    """
    super().__init__()
    self.file_path_prefix = file_path_prefix
    self.column_name_type_list = column_name_type_list

  def expand(self, pcoll=None):
    pa_schema = create_pa_schema_from_list(self.column_name_type_list)
    pcoll \
      | f'write_to_parquet_{time.time_ns()}' \
      >> parquetio.WriteToParquet(\
      file_path_prefix=self.file_path_prefix,\
      schema=pa_schema, file_name_suffix='.parquet')

class LeftJoinFn(beam.DoFn):
  """
  left join of left PCollection rows with right PCollection row.

  if there is  more than one row in the right, a ValueError is thrown.

  :return returns merged rows
  """
  def __init__(self, right_filter_cols,\
    l_key_col : int, r_key_col : int, debug_tag:str=""):
    super().__init__()
    self.right_filter_cols = right_filter_cols
    self.debug_tag = debug_tag
    self.l_key_col = l_key_col
    self.r_key_col = r_key_col

  #TODO: consider also implementing def process_batch
  def process(self, kv):
    key, grouped_elements = kv
    # grouped_elements is a dictionary with keys 'left' and 'right'
    # both are lists of lists.

    if len(grouped_elements['right']) != 1:
      raise ValueError(f"{self.debug_tag}: in join, right list length != 1: key={key}, grouped_elements={grouped_elements}")

    right_table = grouped_elements['right'][0]

    #does nothing if this is a right join with no left element
    output_rows = []
    for left_table in grouped_elements['left']:
      # join grouped_elements['left'] and grouped_elements['right'][0]
      for left_row in left_table.to_pylist():
        for right_row in right_table.to_pylist():
          if left_row[self.l_key_col] == right_row[self.r_key_col]:
            for d in self.right_filter_cols:
              del right_row[d]
            output_rows.append({**left_row, **right_row})

    if output_rows:
      yield pa.Table.from_pylist(output_rows)

class MergeByKey(beam.PTransform):
  """
  the PTransform operates on 1 PCollection, but we can add the other
  as a side input by giving it to the constructor.

  Merge the left and right PCollections on the given key columns,
  and exclude filter_cols of the right PCollection from entering
  the output left joined rows.

  The left PCollection is the main PCollection piped to the PTransform
  while the right PCollection is given as side input to the
  constructor, along with the other parameters"""
  def __init__(self, r_pc : beam.PCollection, \
    l_key_col : str, r_key_col : str, \
    filter_cols : List[str], debug_tag : str = ""):
    super().__init__()
    self.r_pc = r_pc
    self.l_key_col = l_key_col
    self.r_key_col = r_key_col
    self.filter_cols = filter_cols
    self.debug_tag = debug_tag

  def expand(self, l_pc):
    l_keyed = (l_pc | beam.FlatMap(
        lambda table: [(row[self.l_key_col], table)
          for row in table.to_pylist()])
        )

    r_keyed = (self.r_pc | beam.FlatMap(
        lambda table: [(row[self.r_key_col], table)
          for row in table.to_pylist()])
        )

    # multiple lefts on one line, and one in right's list:
    grouped_data = ({'left': l_keyed, 'right': r_keyed} \
      | f'group_by_key_{random.randint(0,1000000000)}' \
      >> beam.CoGroupByKey())

    try:
      joined_data = grouped_data \
        | f'left_join_values_{random.randint(0,1000000000)}' \
        >> beam.ParDo(LeftJoinFn(self.filter_cols, \
           self.l_key_col, self.r_key_col, debug_tag=self.debug_tag))
    except Exception as ex:
      logging.error(f"ERROR for {self.debug_tag}: l_key_col={self.l_key_col}, r_key_col={self.r_key_col}")
      raise ex

    return joined_data

#TODO: change the use of uris here and in the parquet writes
class ReadFiles(beam.PTransform):
  """
  read ratings, movies, and users independently from parquet files
  into pytables
  """
  def __init__(self, infiles_dict):
    super().__init__()
    self.infiles_dict = infiles_dict

  def expand(self, pcoll=None):
    pc = {}
    for key in ['ratings', 'movies', 'users']:
      infile_dict = self.infiles_dict[key]
      dir_path = os.path.dirname(os.path.abspath(infile_dict['uri']))
      file_path_prefix = f'{dir_path}/{key}'

      file_path_pattern = f"{file_path_prefix}*.parquet"

      pc[key] = pcoll | f"read_parquet_{random.randint(0,10000000000)}" >> \
        parquetio.ReadFromParquetBatched(file_path_pattern)

    return pc

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

    tables_dict = pcoll | f"read_{random.randint(0, 1000000000000)}" \
      >> ReadFiles(self.infiles_dict)

    # ratings: user_id,movie_id,rating,timestamp
    # movie_id,title,genre
    # user_id::gender::age::occupation::zipcode
       # user_id,movie_id,rating,timestamp,gender,age,occupation,zipcode
    ratings_1 = tables_dict['ratings'] | \
      f"left_join_ratings_users_{random.randint(0,1000000000)}" \
      >> MergeByKey(tables_dict['users'], \
      'user_id', 'user_id', \
      filter_cols=['zipcode', 'user_id'], debug_tag="R-U")

    # user_id,movie_id,rating,timestamp,gender,age,occupation,zipcode,genres
    ratings = ratings_1 | \
      f"left_join_ratings_users_movies_{random.randint(0,1000000000)}" \
      >> MergeByKey(tables_dict['movies'], \
      'movie_id', 'movie_id', \
      filter_cols=['title', 'movie_id'], \
      debug_tag="R-M")

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
