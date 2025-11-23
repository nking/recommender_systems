import apache_beam as beam
from apache_beam.io import parquetio
from apache_beam.io.filesystems import FileSystems

"""
offers an IngestAndJoin which reads Parquet files into pyarrow tables,
performs a left inner join on ratings with users and movies.
"""

#from apache_beam.coders import coders

#TODO: replace use of random in labels for PTransform with unique
#  sequential numbers

import random
import time

from movie_lens_tfx.utils.movie_lens_utils import *
#from apache_beam.pvalue import TaggedOutput

from movie_lens_tfx.utils.CustomUTF8Coder import CustomUTF8Coder

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

  def expand(self, pcoll=None):
    pc = {}
    for key in ['ratings', 'movies', 'users']:
      if self.infiles_dict[key]['headers_present']:
        skip = 1
      else:
        skip = 0
      pa_schema_list = create_pa_schema(self.infiles_dict[key])
      pc[key] = (pcoll | f"r{random.randint(0,10000000000)}"
        >> beam.io.ReadFromText(self.infiles_dict[key]['uri'], skip_header_lines=skip,
        coder=CustomUTF8Coder())
        | f'parse_{key}_{time.time_ns()}'
        >> beam.Map(self.line_to_record, self.infiles_dict[key],
        pa_schema_list))
        #| f'write_to_parquet_{key}_{time.time_ns()}' \
        #>> parquetio.WriteToParquet(\
        #file_path_prefix=file_path_prefix,\
        #schema=pa_schema, file_name_suffix='.parquet')
      #print(f'TYPE {key}={type(pc[key])}')
      #class 'apache_beam.transforms.ptransform._ChainedPTransform'
    return pc

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
    logging.debug(f'file_path_prefix={self.file_path_prefix}')
    pa_schema_list = create_pa_schema(self.infile_dict)
    pa_schema = pa.schema(pa_schema_list)
    logging.debug(f'pa_schema=\n{pa_schema}')
    print(f'pa_schema={pa_schema}')
    
    """
    sampled_data = pcoll | f'Sample Elements_{random.randint(0, 1000000000)}' >> beam.combiners.Sample.FixedSizeGlobally(5)
    (
      sampled_data
      | f'Flatten List_{random.randint(0, 1000000000)}' >> beam.FlatMap(lambda x: x)
      | f'Print Rows_{random.randint(0, 1000000000)}' >> beam.Map(print)
    )
    """
    
    pcoll | (f'write_to_parquet_{time.time_ns()}'
      >> parquetio.WriteToParquet(
      file_path_prefix=self.file_path_prefix,
      schema=pa_schema, file_name_suffix='.parquet'))

class WriteJoinedRatingsParquet(beam.PTransform):
  def __init__(self, file_path_prefix, column_name_type_list):
    """
    write the joined ratings file to a parquet file given the
    file_path_prefix and the column_name_type_list returned by
    ingest_movie_lens_beam_pa.IngestAndJoin.
    column_name_type_list is used for creating a pyarrow schema
    for parquetio.WriteToParquet.
    The joined ratings is a PCollection of pyarraow table and so has a schema
    for the first, etc. but parquetio.WriteToParquet input schema
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
    pa_schema_list = create_pa_schema_from_list(self.column_name_type_list)
    self.pa_schema = pa.schema(pa_schema_list)
  
  def line_to_record(self, line):
    if len(self.column_name_type_list) != len(line):
      raise ValueError(f"line length = {len(line)} and column_name_type_list length = {len(self.column_name_type_list)}")
    out = {}
    for i, item in enumerate(line):
      key = self.column_name_type_list[i][0]
      type = self.column_name_type_list[i][1]
      if type == int:
        out[key] = int(item)
      elif type == float:
        out[key] = float(item)
      else:
        out[key] = item
    return out
  
  def expand(self, pcoll=None):
    
    pcollrecord = pcoll | (f'format_to_records_{random.randint(0, 1000000000)}'
      >> beam.Map(self.line_to_record))
    
    pcollrecord | (f'write_to_parquet_{time.time_ns()}'
      >> parquetio.WriteToParquet(file_path_prefix=self.file_path_prefix,
      schema=self.pa_schema, file_name_suffix='.parquet'))

#TODO: change the use of uris here and in the parquet writes
class ReadFiles(beam.PTransform):
  """
  read ratings, movies, and users independently from parquet files
  into pytables
  """
  def __init__(self, infiles_dict):
    super().__init__()
    self.infiles_dict = infiles_dict
    logging.debug(f"ReadFiles")

  def expand(self, pcoll=None):
    pc = {}
    for key in ['ratings', 'movies', 'users']:
      infile_dict = self.infiles_dict[key]
      dir_path = infile_dict['uri']
      if not dir_path.startswith('/'):
        dir_path = os.path.abspath(infile_dict['uri'])
      if os.path.isfile(dir_path):
        dir_path = os.path.dirname(dir_path)
      if dir_path.endswith("/"):
        dir_path = dir_path[:-1]
      file_path_prefix = f'{dir_path}/{key}'
      file_path_pattern = f"{file_path_prefix}*.parquet"
      logging.debug(f"ReadFromParquet {key}: file_path_pattern={file_path_pattern}")

      pc[key] = pcoll | f"read_parquet_{random.randint(0,10000000000)}" >> \
        parquetio.ReadFromParquet(file_path_pattern)

      logging.debug(f"read {key} parquet file")
    return pc