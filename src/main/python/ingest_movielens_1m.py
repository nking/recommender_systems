

from typing import Any, Dict, List, Text

import numpy as np
import tensorflow as tf
import apache_beam as beam

'''
if not os.path.exists(proj_dir):
    from urllib.request import urlretrieve
    from zipfile import ZipFile
    urlretrieve("http://files.grouplens.org/datasets/movielens/ml-1m.zip", \
                filename=working_dir +"movielens.zip")
    os.mkdir(proj_dir)
    ZipFile(file=working_dir + "movielens.zip", mode="r").extractall(path=working_dir)
    # tensorflow delimeter is single character, so change delimeter :: to \t
    for file in ['movies.dat', 'users.dat', 'ratings.dat']:
'''

# following custom funcction component via TFX
# because it will handle the parititioning automatically.
# for the split by time, the count of all records happens first
# and capture of vocabularies,
# then the split then the processing which will be partitioned.
#
# https://www.tensorflow.org/tfx/guide/custom_function_component

import apache_beam as beam
from datetime import datetime

#ratings.dat
#- UserIDs range between 1 and 6040
#- MovieIDs range between 1 and 3952
#- Ratings are made on a 5-star scale (whole-star ratings only)
#- Timestamp is represented in seconds since the epoch as returned by time(2)
#- Each user has at least 20 ratings
#1::1193::5::978300760
def parse_date(element):
    # Assuming the date is in the first column of a CSV
    parts = element.split('::')
    date_str = parts[3]
    date_obj = datetime.fromtimestamp(date_str)
    return (date_obj, element)  # Return date object and original element


parsed_data = lines | 'ParseDate' >> beam.Map(parse_date)

output = proto.Output(
             split_config=example_gen_pb2.SplitConfig(splits=[
                 proto.SplitConfig.Split(name='train', hash_buckets=3),
                 proto.SplitConfig.Split(name='eval', hash_buckets=1)
             ]))
example_gen = CsvExampleGen(input_base=input_dir, output_config=output)
