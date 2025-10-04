import apache_beam as beam
#from apache_beam.coders import coders

import time
import random

from typing import Any, Dict, List, Union, Tuple
from infile_dict_util import *
from apache_beam.pvalue import TaggedOutput

from CustomUTF8Coder import CustomUTF8Coder

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

def _read_files(infiles_dict: Dict[str, Union[str, Dict]]) -> \
  Dict[str, beam.pvalue.PCollection]:
  pc = {}
  for key in ['ratings', 'movies', 'users']:
    if infiles_dict[key]['headers_present']:
      skip = 1
    else:
      skip = 0
    pc[key] = pipeline | f'read_{key}_{time.time_ns()}' >> beam.io.ReadFromText(\
      infiles_dict[key]['uri'], skip_header_lines=skip, coder=CustomUTF8Coder()) \
      | f'parse_{key}_{time.time_ns()}' >> beam.Map(lambda line: line.split(infiles_dict[key]['delim']))
  return pc

#@beam.ptransform_fn
#@beam.typehints.with_input_types(beam.Pipeline, str, str, str,\
#  Dict[str, int], Dict[str, int], Dict[str, int], \
#  List[int])
#@beam.typehints.with_output_types(Dict[str, beam.pvalue.PCollection])
def ingest_and_join(\
  #pipeline: BeamComponentParameter[beam.Pipeline],
  pipeline : beam.pipeline.Pipeline, \
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
    the dictionary is made by using infile_dict_util.make_file_dict
    for each file, then infile_dict_util.merge_dicts.
  :return: a tuple of PCollection of ratings with joined information from users and movies where each tuple is for a
     partition specified in partition list.  The PCollection has an associated schema
  """

  err = dict_formedness_error(infiles_dict)
  if err:
    raise ValueError(err)

  pc = _read_files(infiles_dict)

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

  return ratings, columns

#@beam.ptransform_fn
#@beam.typehints.with_input_types(beam.Pipeline, str, str, str,\
#  Dict[str, int], Dict[str, int], Dict[str, int], \
#  List[int])
#@beam.typehints.with_output_types(Dict[str, beam.pvalue.PCollection])
def ingest_join_and_split(\
  #pipeline: BeamComponentParameter[beam.Pipeline],
  pipeline : beam.pipeline.Pipeline, \
  infiles_dict: Dict[str, Union[str, Dict]], \
  buckets : List[int], bucket_names : List[str]) -> \
  Tuple[Dict[str, beam.pvalue.PCollection], List[Tuple[str, Any]]]:
  """
  reads in the 3 expected files from the uris given, and then uses
  left joins of ratings with user information and movie genres to
  make a PCollection, and then splits the PCollection into the
  given buckets, randomly, returning a dictionary of the partitioned
  PCollections.

  :param pipeline:

  :param infiles_dict
    a dictionary of file information for each of the 3 files, that is
    the ratings file, movies file, and users file.
    the dictionary is made by using infile_dict_util.make_file_dict
    for each file, then infile_dict_util.merge_dicts.

  :param buckets: list of partitions in percent
  :param bucket_names: list of partition bucket names
  :return: a tuple of PCollection of ratings with joined information from users and movies where each tuple is for a
     partition specified in partition list
  """

  # user_id,movie_id,rating,gender,age,occupation,zipcode,genres
  ratings, schema_list = ingest_and_join(pipeline=pipeline, \
    infiles_dict=infiles_dict)

  if buckets is None or len(buckets)==0:
    buckets = [100]
    if bucket_names is None or len(bucket_names) == 0:
      bucket_names = ['train']

  #print(f'RATINGS type{type(ratings)}')
  #ratings | f'ratings_{time.time_ns()}' >> beam.Map(print)
  #['user_id', 'movie_id', 'rating', 'gender', 'age', 'occupation', 'genres']

  #consider ways to handle this scatter gather more efficiently
  def split_fn(row, ppartitions):
    # Using a deterministic hash function ensures the splits are consistent
    total = sum(ppartitions)
    b = [p / total for p in ppartitions]
    rand_num = random.random()
    s = 0
    for i, p in enumerate(b):
      s += p
      if rand_num < s:
        return i
    return len(ppartitions) - 1

  #type: apache_beam.pvalue.DoOutputsTuple
  ratings_parts = ratings \
    | f'split_{time.time_ns()}' >> beam.Partition(split_fn, buckets)

  #dictionary of PCollections:
  output_dict = {f'{bucket_names[i]}]' : pc for i, pc in enumerate(ratings_parts)}

  #for i, part in enumerate(ratings_parts):
  #  part | f'PARTITIONED_{i}_{time.time_ns()}' >> beam.io.WriteToText(\
  #    file_path_prefix=f'a_{i}_', file_name_suffix='.txt')

  return output_dict, schema_list

if __name__ == "__main__":

  #TODO: move this out of source code and into test code

  import apache_beam as beam
  from apache_beam.testing.util import assert_that, is_not_empty, \
    equal_to

  kaggle = False
  if kaggle:
    prefix = '/kaggle/working/ml-1m/'
  else:
    prefix = "../resources/ml-1m/"
  ratings_uri = f"{prefix}ratings.dat"
  movies_uri = f"{prefix}movies.dat"
  users_uri = f"{prefix}users.dat"

  ratings_col_names = ["user_id", "movie_id", "rating"]
  ratings_col_types = [int, int, int]  # for some files, ratings are floats
  movies_col_names = ["movie_id", "title", "genres"]
  movies_col_types = [int, str, str]
  users_col_names = ["user_id", "gender", "age", "occupation","zipcode"]
  users_col_types = [int, str, int, int, str]

  expected_schema_cols = [ \
    ("user_id", int), ("movie_id", int), ("rating", int), \
    ("gender", str), ("age", int), ("occupation", int), \
    ("genres", str)]

  ratings_dict = make_file_dict(for_file='ratings', \
                                uri=ratings_uri,
                                col_names=ratings_col_names, \
                                col_types=ratings_col_types,
                                headers_present=False, delim="::")

  movies_dict = make_file_dict(for_file='movies', \
                               uri=movies_uri,
                               col_names=movies_col_names, \
                               col_types=movies_col_types,
                               headers_present=False, delim="::")

  users_dict = make_file_dict(for_file='users', \
                              uri=users_uri, col_names=users_col_names, \
                              col_types=users_col_types,
                              headers_present=False, delim="::")

  infiles_dict = merge_dicts(ratings_dict=ratings_dict, \
                            movies_dict=movies_dict, \
                            users_dict=users_dict)

  buckets = [80, 10, 10]
  bucket_names = ['train', 'eval', 'test']

  #DirectRunner is default pipeline if options is not specified
  from apache_beam.options.pipeline_options import PipelineOptions

  #apache-beam 2.59.0 - 2.68.0 with SparkRunner supports pyspark 3.2.x
  #but not 4.0.0
  #pyspark 3.2.4 is compatible with java >= 8 and <= 11 and python >= 3.6 and <= 3.9
  # start Docker, then use portable SparkRunner
  # https://beam.apache.org/documentation/runners/spark/
  #from pyspark import SparkConf
  options = PipelineOptions(\
    #runner='SparkRunner',\
    #runner='PortableRunner',\
    runner='DirectRunner',\
    #spark_conf=spark_conf_list,\
  )

  with beam.Pipeline(options=options) as pipeline:

    #test read files
    pc = _read_files(infiles_dict)
    #pc['ratings'] | f'ratings: {time.time_ns()}' >> \
    #  beam.Map(lambda x: print(f'ratings={x}'))
    r_count = pc['ratings']  | 'count' >> beam.combiners.Count.Globally()
    #r_count | 'count ratings' >> beam.Map(lambda x: print(f'len={x}'))
    assert_that(r_count, equal_to([1000209]))

    assert_that(pc['movies']  | f'count {time.time_ns()}' >> beam.combiners.Count.Globally(), \
      equal_to([3883]))
    assert_that(pc['users'] | f'count {time.time_ns()}' >> beam.combiners.Count.Globally(), \
      equal_to([6040]))

    ratings, schema_list = ingest_and_join(pipeline=pipeline, \
      infiles_dict=infiles_dict)

    assert expected_schema_cols == schema_list

    #ratings_tuple, schema_list = ingest_join_and_split(
    #  pipeline=pipeline, infiles_dict=infiles_dict, buckets=buckets,
    #  bucket_names=bucket_names)

    assert_that(ratings, is_not_empty(), label=f'Assert Non-Empty ratings PCollection')

    if False:
      pre = f'../../../bin/ml1m_ingest_'
      import os
      import glob
      for file_path in glob.glob(os.path.join("../../../bin", "ml1m_ingest_*")):
        try:
          os.remove(file_path)
          print(f"Removed: {file_path}")
        except OSError as e:
          print(f"Error removing {file_path}: {e}")
      ratings | f'write_{time.time_ns()}' >> beam.io.WriteToText(\
        file_path_prefix=f'{pre}_ratings', file_name_suffix='.txt')

    if False:
      ratings_tuple, schema_list = ingest_join_and_split(
        pipeline=pipeline, infiles_dict=infiles_dict, buckets=buckets,
        bucket_names=bucket_names)

      for k, v in ratings_tuple.items():
        assert_that(v, is_not_empty(), label=f'Assert Non-Empty {k} PCollection')

      if False:
        pre = f'../../../bin/ml1m_ingest_'
        import os
        import glob
        for file_path in glob.glob(os.path.join("../../../bin", "ml1m_ingest_*")):
          try:
            os.remove(file_path)
            print(f"Removed: {file_path}")
          except OSError as e:
            print(f"Error removing {file_path}: {e}")
        for k, v in ratings_tuple.items():
          v | k >> beam.io.WriteToText(\
          file_path_prefix=f'{pre}_{k}', file_name_suffix='.txt')

    print(f'tests done')