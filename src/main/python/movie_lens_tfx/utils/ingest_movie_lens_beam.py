import apache_beam as beam
#from apache_beam.coders import coders

#TODO: replace use of random in labels for PTransform with unique
#  sequential numbers

import random

from movie_lens_tfx.utils.movie_lens_utils import *
#from apache_beam.pvalue import TaggedOutput

from movie_lens_tfx.utils.CustomUTF8Coder import CustomUTF8Coder

from absl import logging
logging.set_verbosity(logging.INFO)
logging.set_stderrthreshold(logging.INFO)

#@beam.typehints.with_output_types(beam.PCollection)
class LeftJoinFn(beam.DoFn):
  """
  left join of left PCollection rows with right PCollection row.

  if there is  more than one row in the right, a ValueError is thrown.

  :return returns merged rows
  """
  def __init__(self, right_filter_cols, debug_tag:str=""):
    super().__init__()
    self.right_filter_cols = right_filter_cols
    self.debug_tag = debug_tag

  #TODO: consider also implementing def process_batch
  def process(self, kv):
    key, grouped_elements = kv
    # grouped_elements is a dictionary with keys 'left' and 'right'
    # both are lists of lists.
    if len(grouped_elements['right']) != 1:
      raise ValueError(f"{self.debug_tag}: in join, right list length != 1: key={key}, grouped_elements={grouped_elements}")

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
    l_key_col : int, r_key_col : int, \
    filter_cols : List[int], debug_tag : str = ""):
    super().__init__()
    self.r_pc = r_pc
    self.l_key_col = l_key_col
    self.r_key_col = r_key_col
    self.filter_cols = filter_cols
    self.debug_tag = debug_tag

  def expand(self, l_pc):
    #TODO: follow up on improvements like using .with_resource_hints
    l_keyed = l_pc | f'kv_l_{random.randint(0,1000000000)}' \
      >> beam.Map(lambda x: (x[self.l_key_col], x))
    r_keyed = self.r_pc | f'kv_r_{random.randint(0,1000000000)}' \
      >> beam.Map(lambda x: (x[self.r_key_col], x))

    # multiple lefts on one line, and one in right's list:
    grouped_data = ({'left': l_keyed, 'right': r_keyed} \
      | f'group_by_key_{random.randint(0,1000000000)}' \
      >> beam.CoGroupByKey())

    try:
      joined_data = grouped_data \
        | f'left_join_values_{random.randint(0,1000000000)}' \
        >> beam.ParDo(LeftJoinFn(self.filter_cols, debug_tag=self.debug_tag))
    except Exception as ex:
      logging.error(f"ERROR for {self.debug_tag}: l_key_col={self.l_key_col}, r_key_col={self.r_key_col}")
      raise ex

    return joined_data

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

    # ratings: user_id,movie_id,rating,timestamp
    # movie_id,title,genre
    # user_id::gender::age::occupation::zipcode
       # user_id,movie_id,rating,timestamp,gender,age,occupation,zipcode
    ratings_1 = pc['ratings'] | \
      f"left_join_ratings_users_{random.randint(0,1000000000)}" \
      >> MergeByKey(pc['users'], \
      self.infiles_dict['ratings']['cols']['user_id']['index'], \
      self.infiles_dict['users']['cols']['user_id']['index'], \
      filter_cols=[self.infiles_dict['users']['cols']['zipcode']['index'],\
      self.infiles_dict['users']['cols']['user_id']['index']], debug_tag="R-U")

    # user_id,movie_id,rating,timestamp,gender,age,occupation,zipcode,genres
    ratings = ratings_1 | \
      f"left_join_ratings_users_movies_{random.randint(0,1000000000)}" \
      >> MergeByKey(pc['movies'], \
      self.infiles_dict['ratings']['cols']['movie_id']['index'], \
      self.infiles_dict['movies']['cols']['movie_id']['index'], \
      filter_cols=[self.infiles_dict['movies']['cols']['title']['index'], \
      self.infiles_dict['movies']['cols']['movie_id']['index']], \
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

    logging.debug(f"output ingest columns={columns}")
    
    return ratings, columns
