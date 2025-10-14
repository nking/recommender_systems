import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils
#from tfx.components.trainer import fn_args_utils
#from tfx_bsl.tfxio import dataset_options
from datetime import datetime
import pytz

import absl
from absl import logging
logging.set_verbosity(absl.logging.DEBUG)

## fixed vocabularies, known ahead of time
genres = ["Action", "Adventure", "Animation", "Children", "Comedy",
          "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
          "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
          "Thriller", "War", "Western"]
genders = ['F', 'M']

## 0-17, 18-24, 25-34, 35-44, 45-49, 50-55, >= 56
age_groups = [1, 18, 25, 35, 45, 50, 56]

num_occupations = 21

#for now, fudging timezones:
CTZ = pytz.timezone("America/Chicago")

# Tf.Transform considers these features as "raw"
def _get_raw_feature_spec(schema):
  return schema_utils.schema_as_feature_spec(schema).feature_spec

def create_static_table(var_list, key_dtype, value_dtype):
  return tf.lookup.StaticHashTable( \
    tf.lookup.KeyValueTensorInitializer(\
      keys=tf.constant(var_list), \
      values=tf.range(len(var_list), dtype=tf.int64),\
      key_dtype=key_dtype, value_dtype=value_dtype), \
      default_value=-1, dtype=tf.int64)

def preprocessing_fn(inputs):
  """
  :param inputs: map from feature keys to raw not-yet-transformed features.
     features have the following column names and types:
     column_names = user_id,movie_id,rating,gender,age,occupation,zipcode,genres
     column_types = int,    int ,     int ,  str, int, int,       str,     str
  :return: tuple of (processed features without label, label)
  """

  outputs = {'user_id': inputs['user_id'], 'movie_id': inputs['movie_id']}

  #outputs['rating'] = inputs['rating']/5.0
  labels = inputs['rating']/5.0

  gender_table = create_static_table(genders, key_dtype=tf.string, value_dtype=tf.int64)
  outputs['gender'] = gender_table.lookup(inputs['gender'])
  outputs['gender'] = tf.one_hot( outputs['gender'], depth=len(genders), dtype=tf.int64)

  age_groups_table = create_static_table(age_groups, key_dtype=tf.int64, value_dtype=tf.int64)
  outputs['age'] = age_groups_table.lookup(inputs['age'])
  outputs['age'] = tf.one_hot( outputs['age'], depth=len(age_groups), dtype=tf.int64)

  outputs['occupation'] = inputs['occupation'].copy()
  outputs['occupation'] = tf.one_hot( outputs['occupation'], depth=num_occupations, dtype=tf.int64)

  #omitting zipcode for now, but considering ZCTAs for future

  outputs['genres'] = tf.strings.regex_replace(
      input = inputs['genres'], pattern="Children's", rewrite="Children")
  #creates a RaggedTensor of strings:
  outputs['genres'] = tf.strings.split(outputs['genres'], "|")
  genres_table = create_static_table(genres, key_dtype=tf.string, value_dtype=tf.int64)
  # creates a RaggedTensor of tf.int64:
  outputs['genres'] = genres_table.lookup(outputs['genres'])
  #the model needs tensors to be same size, so make it dense multithot
  outputs['genres'] = tf.reduce_sum(tf.one_hot(\
    indices=outputs['genres'], depth=len(genres), dtype=tf.int64), axis=-2)
  #the sum is across columns for each example.
  #assuming batch_size is the first dimension, then cols and rows follow
  # so axis=-2 should sum along columns per example whether batched or not

  local_time = datetime.fromtimestamp(inputs["timestamp"], tz=CTZ)
  outputs["hr"] = int(round(local_time.hour + (local_time.minute / 60.)))
  outputs["weekday"] = local_time.weekday()
  outputs["hr_wk"] = outputs["hr"] * 7 + outputs["weekday"]

  return outputs, labels

#def stats_options_updater_fn(stats_type, stats_options):
#  "define custom constraints on the pre-transform or post-transform statistics"
#  if stats_type == stats_options_util.StatsType.PRE_TRANSFORM:
    # Update stats_options to modify pre-transform statistics computation.
    # Most constraints are specified in the schema which can be accessed
    # via stats_options.schema.
#  if stats_type == stats_options_util.StatsType.POST_TRANSFORM
    # Update stats_options to modify post-transform statistics computation.
    # Most constraints are specified in the schema which can be accessed
    # via stats_options.schema.
#  return stats_options
