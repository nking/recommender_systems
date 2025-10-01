import apache_beam as beam
import time
import random

from typing import Any, Dict, List, Text

from apache_beam.pvalue import TaggedOutput
from CustomUTF8Coder import CustomUTF8Coder

class LeftJoinFn(beam.DoFn):
  """
  left join of left PCollection rows with right PCollection row.

  if there is  more than one row in the right, a ValueError is thrown.

  :return returns merged rows
  """
  def __init__(self, right_filter_cols):
    self.right_filter_cols = right_filter_cols

  def process(self, kv):
    key, grouped_elements = kv
    # grouped_elements is a dictionary with keys 'left' and 'right'
    # both are lists of lists.
    if len(grouped_elements['right']) != 1:
      raise ValueError(f"in join, right list length != 1: key={key}, grouped_elements={grouped_elements}")

    for left in grouped_elements['left']:
      # join grouped_elements['left'] and grouped_elements['right'][0]
      # merge, reorder etc. as wanted
      row = left.copy()
      for i, right in enumerate(grouped_elements['right'][0]):
        if i not in self.right_filter_cols:
          row.append(right)
      yield row

#@beam.ptransform_fn
#@beam.typehints.with_input_types(beam.pvalue.PCollection,\
#  beam.pvalue.PCollection, Dict[str, int], Dict[str, int], List[int])
#@beam.typehints.with_output_types(Dict[str, beam.pvalue.PCollection])
def merge_by_key(l_pc : beam.pvalue.PCollection, r_pc : beam.pvalue.PCollection, \
  l_key_col : Dict[str, int], r_key_col : Dict[str, int], \
  filter_cols : List[int]) -> beam.pvalue.PCollection:
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
  :return: a merged PCollection
  """
  # need unique names for each beam process, so adding a timestamp
  ts = time.time_ns()
  l_keyed = l_pc | f'kv_l_{ts}' >> beam.Map(lambda x: (x[l_key_col], x))
  r_keyed = r_pc | f'kv_r_{ts}' >> beam.Map(lambda x: (x[r_key_col], x))

  # l_keyed | beam.Map(print)
  # r_keyed | 'beam.Map(print)

  # multiple lefts on one line, and one in right's list:
  grouped_data = ({'left': l_keyed, 'right': r_keyed} \
                  | f'group_by_key_{ts}' >> beam.CoGroupByKey())

  joined_data = grouped_data \
      | f'left_join_values_{ts}' >> beam.ParDo(LeftJoinFn(filter_cols))

  return joined_data

#@beam.ptransform_fn
#@beam.typehints.with_input_types(beam.Pipeline, str, str, str,\
#  Dict[str, int], Dict[str, int], Dict[str, int], \
#  List[int])
#@beam.typehints.with_output_types(Dict[str, beam.pvalue.PCollection])
def ingest_and_join(\
  #pipeline: BeamComponentParameter[beam.Pipeline],
  pipeline : beam.pipeline.Pipeline, \
  ratings_uri : str, movies_uri : str, users_uri : str, \
  headers_present: bool, delim: str, \
  ratings_key_dict : Dict[str, int], movies_key_dict : Dict[str, int], \
  users_key_dict : Dict[str, int]) -> Dict[str, beam.pvalue.PCollection]:
  """
  reads in the 3 expected files from the uris given, and then uses
  left joins of ratings with user information and movie genres to
  make a PCollection, and then splits the PCollection into the
  given buckets, randomly, returning a dictionary of the partitioned
  PCollections.

  :param pipeline:
  :param ratings_uri:
  :param movies_uri:
  :param users_uri:
  :param headers_present:
  :param ratings_key_dict: for ratings file, a dictionary with key:values
    being header_column_name:column number
  :param movies_key_dict: for movies file, a dictionary with key:values being header_column_name:column number
  :param users_key_dict: for users file, a dictionary with key:values being header_column_name:column number
  :param buckets: list of partitions in percent
  :return: a tuple of PCollection of ratings with joined information from users and movies where each tuple is for a
     partition specified in partition list
  """

  if headers_present:
    skip = 1
  else:
    skip = 0

  # user_id,movie_id,rating
  ratings_pc = pipeline \
    | 'ReadRatings' >> beam.io.ReadFromText(ratings_uri, \
    skip_header_lines=skip, coder=CustomUTF8Coder()) \
    | 'ParseRatings' >> beam.Map(lambda line: line.split(delim))

  # movie_id,title,genre
  movies_pc = pipeline \
    | 'ReadMovies' >> beam.io.ReadFromText(movies_uri, \
    skip_header_lines=skip, coder=CustomUTF8Coder()) \
    | 'ParseMovies' >> beam.Map(lambda line: line.split(delim))

  # user_id::gender::age::occupation::zipcode
  users_pc = pipeline \
    | 'ReadUsers' >> beam.io.ReadFromText(users_uri, \
    skip_header_lines=skip, coder=CustomUTF8Coder()) \
    | 'ParseUsers' >> beam.Map(lambda line: line.split(delim))

  # user_id,movie_id,rating,timestamp,geder,age,occupation,zipcode
  ratings_1 = merge_by_key(ratings_pc, users_pc, \
    ratings_key_dict['user_id'], users_key_dict['user_id'], \
    filter_cols=[users_key_dict['zipcode']])

  # user_id,movie_id,rating,gender,age,occupation,zipcode,genres
  ratings = merge_by_key(ratings_1, movies_pc, \
    ratings_key_dict['movie_id'], movies_key_dict['movie_id'],\
    filter_cols=[movies_key_dict['title']])

  return ratings

#@beam.ptransform_fn
#@beam.typehints.with_input_types(beam.Pipeline, str, str, str,\
#  Dict[str, int], Dict[str, int], Dict[str, int], \
#  List[int])
#@beam.typehints.with_output_types(Dict[str, beam.pvalue.PCollection])
def ingest_join_and_split(\
  #pipeline: BeamComponentParameter[beam.Pipeline],
  pipeline : beam.pipeline.Pipeline, \
  ratings_uri : str, movies_uri : str, users_uri : str, \
  headers_present: bool, delim: str, \
  ratings_key_dict : Dict[str, int], movies_key_dict : Dict[str, int], \
  users_key_dict : Dict[str, int], \
  buckets : List[int]) -> Dict[str, beam.pvalue.PCollection]:
  """
  reads in the 3 expected files from the uris given, and then uses
  left joins of ratings with user information and movie genres to
  make a PCollection, and then splits the PCollection into the
  given buckets, randomly, returning a dictionary of the partitioned
  PCollections.

  :param pipeline:
  :param ratings_uri:
  :param movies_uri:
  :param users_uri:
  :param headers_present:
  :param ratings_key_dict: for ratings file, a dictionary with key:values
    being header_column_name:column number
  :param movies_key_dict: for movies file, a dictionary with key:values being header_column_name:column number
  :param users_key_dict: for users file, a dictionary with key:values being header_column_name:column number
  :param buckets: list of partitions in percent
  :return: a tuple of PCollection of ratings with joined information from users and movies where each tuple is for a
     partition specified in partition list
  """

  # user_id,movie_id,rating,gender,age,occupation,zipcode,genres
  ratings = ingest_and_join(pipeline=pipeline, \
    ratings_uri=ratings_uri, movies_uri=movies_uri, \
    users_uri=users_uri, headers_present=headers_present, delim=delim,\
    ratings_key_dict=ratings_key_col_dict, \
    users_key_dict=users_key_col_dict, \
    movies_key_dict=movies_key_col_dict)

  if buckets is None or len(buckets)==0:
    buckets = [100]

  #print(f'RATINGS type{type(ratings)}')
  #ratings | f'ratings_{time.time_ns()}' >> beam.Map(print)
  #['user_id', 'movie_id', 'rating', 'gender', 'age', 'occupation', 'genres']

  #TODO: look into extending BaseExampleGenExecutor for its split by buckets.
  #it might handle the scatter-gather more efficiently
  def split_fn(row, num_partitions, ppartitions):
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
    | f'split_{time.time_ns()}' >> beam.Partition(\
    split_fn, len(buckets), buckets)

  #dictionary of PCollections:
  output_dict = {f'part_{i}' : pc for i, pc in enumerate(ratings_parts)}

  #for i, part in enumerate(ratings_parts):
  #  part | f'PARTITIONED_{i}_{time.time_ns()}' >> beam.io.WriteToText(\
  #    file_path_prefix=f'a_{i}_', file_name_suffix='.txt')

  return output_dict

if __name__ == "__main__":

  #TODO: move this out of source code and into test code

  import apache_beam as beam

  kaggle = True
  if kaggle:
    prefix = '/kaggle/working/ml-1m/'
  else:
    prefix = "../resources/ml-1m/"
  ratings_uri = f"{prefix}ratings.dat"
  movies_uri = f"{prefix}movies.dat"
  users_uri = f"{prefix}users.dat"

  ratings_key_col_dict = {"user_id": 0, "movie_id": 1, "rating": 2, "timestamp": 3}
  movies_key_col_dict = {"movie_id": 0, "title": 1, "genres": 2}
  users_key_col_dict = {"user_id": 0, "gender": 1, "age": 2, \
                         "occupation": 3, "zipcode": 4}
  headers_present = False
  delim = "::"
  partitions = [80, 10, 10]

  #DirectRunner is default pipeline if options is not specified
  from apache_beam.options.pipeline_options import PipelineOptions, \
    StandardOptions

  import argparse

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
    ratings = ingest_join_and_split(pipeline=pipeline, \
                                    ratings_uri=ratings_uri, movies_uri=movies_uri, \
                                    users_uri=users_uri, headers_present=headers_present, delim=delim, \
                                    ratings_key_dict=ratings_key_col_dict, \
                                    users_key_dict=users_key_col_dict, \
                                    movies_key_dict=movies_key_col_dict, \
                                    buckets=partitions)

    from apache_beam.testing.util import assert_that, is_not_empty

    for k, v in ratings.items():
      assert_that(v, is_not_empty(), label=f'Assert Non-Empty {k} PCollection')
    print(f'tests done')

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
      for k, v in ratings.items():
        v | k >> beam.io.WriteToText(\
        file_path_prefix=f'{pre}_{k}', file_name_suffix='.txt')
