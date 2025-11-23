import apache_beam as beam
#from apache_beam.coders import coders

#TODO: replace use of random in labels for PTransform with unique
#  sequential numbers

import random

from apache_beam.io.filesystems import FileSystems

from helper import get_bin_dir
from movie_lens_tfx.utils.movie_lens_utils import *
#from apache_beam.pvalue import TaggedOutput

from movie_lens_tfx.utils.CustomUTF8Coder import CustomUTF8Coder

from absl import logging
logging.set_verbosity(logging.WARNING)
logging.set_stderrthreshold(logging.WARNING)

"""
This is a condensed version of ingest_movie_lens_beam.py that performs the join by using global RAM
tables for users and movies.

It should not be used on large input user and movie, but is instructive to see.
"""

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
  
class LeftJoinFn(beam.DoFn):
  def process(self, left, right_lookup, l_key_col, r_key_col, right_filter_cols, debug_tag):
    """
    given global side-input right_lookup, perform a left inner join on left using the given key columns
    and excluding the right_filter_cols
    """
    
    # user_lookup is now a standard Python dict containing
    # EVERY user from ALL partitions of the file.
    
    ## DEBUG assert full side-input is seen for each worker
    """
    import json
    debug_path = os.path.join(get_bin_dir(), "DEBUG")
    tmp = os.path.join(debug_path, f"TMP_users_{random.randint(0, 10000000)}.dat")
    writer = FileSystems.create(tmp)
    data_str = json.dumps(right_lookup, default=str)
    writer.write(data_str.encode('utf-8'))
    writer.close()
    """
    
    #left is a single row
    id = left[l_key_col]
    right_row = right_lookup.get(id)
    if right_row is None:
      raise ValueError(f"ERROR {debug_tag}: no matching right found for left key={id}")
    row = left.copy()
    for i, right in enumerate(right_row):
      if i not in right_filter_cols:
        row.append(right)
    yield row

class IngestAndJoin(beam.PTransform):
  """
  reads in the 3 expected files from the uris given in infiles_dict, and then
  left joins on ratings with user information and movie genres to
  make a PCollection.  The PCollection has an associated schema in it.
  
  Note that the Join requires the Users file to be able to fit in RAM, and then
  the Movies file to be able to fit into RAM.

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
    
    debug_tag = "R-U"
    l_key_col = self.infiles_dict['ratings']['cols']['user_id']['index']
    r_key_col = self.infiles_dict['users']['cols']['user_id']['index'],
    filter_cols = [
      self.infiles_dict['users']['cols']['zipcode']['index'],
      self.infiles_dict['users']['cols']['user_id']['index']]
   
    # Prepare the Users (The Side Input)
    # CRITICAL STEP: Apply GlobalWindows to ensures ALL users are seen
    # regardless of when the rating timestamp is.
    users_kv = (
      pc["users"]
      | f"UserKV_{random.randint(0, 1000000000000)}" >> beam.Map(lambda x: (x[r_key_col], x))
      | f"GlobalUsers_{random.randint(0, 1000000000000)}" >> beam.WindowInto(beam.window.GlobalWindows())
    )
    
    # Create the View
    # This is the command that forces "All Partitions into Single RAM"
    user_view = beam.pvalue.AsDict(users_kv)
    
    import os
    import shutil
    debug_path = os.path.join(get_bin_dir(),"DEBUG")
    try:
      shutil.rmtree(debug_path)
    except OSError as e:
      pass
    os.makedirs(debug_path, exist_ok=True)

    ratings_1 = (
      pc["ratings"]
      | f"join_{debug_tag}_{random.randint(0, 1000000000000)}" >> beam.ParDo(
          LeftJoinFn(), right_lookup=user_view, l_key_col=l_key_col, r_key_col=r_key_col,
          right_filter_cols=filter_cols, debug_tag=debug_tag
      )
    )
    
    #TODO: ow to delete user_view and users_kv when no longer in use in graph?
    debug_tag = "R-M"
    l_key_col = self.infiles_dict['ratings']['cols']['movie_id']['index']
    r_key_col = self.infiles_dict['movies']['cols']['movie_id']['index']
    filter_cols = [
      self.infiles_dict['movies']['cols']['title']['index'],
      self.infiles_dict['movies']['cols']['movie_id']['index']]
    movies_kv = (
      pc["movies"]
      | f"MovieKV_{random.randint(0, 1000000000000)}" >> beam.Map(
      lambda x: (x[r_key_col], x))
      | f"GlobalUsers_{random.randint(0, 1000000000000)}" >> beam.WindowInto(
      beam.window.GlobalWindows())
    )
    movie_view = beam.pvalue.AsDict(movies_kv)
    
    ratings = (
      ratings_1
      | f"join_{debug_tag}_{random.randint(0, 1000000000000)}" >> beam.ParDo(
          LeftJoinFn(), right_lookup=movie_view, l_key_col=l_key_col, r_key_col=r_key_col,
          right_filter_cols=filter_cols, debug_tag=debug_tag
      )
    )
    
    # ratings: user_id,movie_id,rating,timestamp
    # movie_id,title,genre
    # user_id::gender::age::occupation::zipcode
       # user_id,movie_id,rating,timestamp,gender,age,occupation,zipcode

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

    logging.debug(f"output ingest columns={columns}")
    
    return ratings, columns
