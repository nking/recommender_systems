import tensorflow as tf
import tensorflow_transform as tft
from absl import logging
from tensorflow_transform.tf_metadata import schema_utils

# from tfx.components.trainer import fn_args_utils
# from tfx_bsl.tfxio import dataset_options
logging.set_verbosity(logging.DEBUG)
logging.set_stderrthreshold(logging.DEBUG)

## fixed vocabularies, known ahead of time
#genres = ["Action", "Adventure", "Animation", "Children", "Comedy",
#          "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
#          "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
#          "Thriller", "War", "Western"]
genres = [b'Action', b'Adventure', b'Animation', b'Children', b'Comedy',
          b'Crime', b'Documentary', b'Drama', b'Fantasy', b'Film-Noir',
          b'Horror', b'Musical', b'Mystery', b'Romance', b'Sci-Fi',
          b'Thriller', b'War', b'Western']

genders = ['F', 'M']

## 0-17, 18-24, 25-34, 35-44, 45-49, 50-55, >= 56
age_groups = [1, 18, 25, 35, 45, 50, 56]

num_occupations = 21

#for now, fudging timezones:

# Tf.Transform considers these features as "raw"
def _get_raw_feature_spec(schema):
  return schema_utils.schema_as_feature_spec(schema).feature_spec

def create_static_table(var_list, var_dtype):
  init = tf.lookup.KeyValueTensorInitializer(\
      keys=tf.constant(var_list, dtype=var_dtype), \
      values=tf.range(len(var_list), dtype=tf.int64),\
      key_dtype=var_dtype, value_dtype=tf.int64)
  return tf.lookup.StaticHashTable(init, default_value=-1)

def preprocessing_fn(inputs):
  """
  :param inputs: map of feature keys with values = tensors of
     raw not-yet-transformed features.
     features have the following keys (column names) and types:
     column_names = user_id,movie_id,rating,timestamp,gender,age,occupation,genres
     column_types = int,    int ,     int ,  int,       str, int, int,          str
  :return: dictionary preprocessed features
  
  ========================================
  inputs={
  'genres': <tf.Tensor 'inputs_copy:0' shape=(None, 1) dtype=string>,
  'age': <tf.Tensor 'inputs_1_copy:0' shape=(None, 1) dtype=int64>,
  'gender': <tf.Tensor 'inputs_2_copy:0' shape=(None, 1) dtype=string>,
  'movie_id': <tf.Tensor 'inputs_3_copy:0' shape=(None, 1) dtype=int64>,
  'occupation': <tf.Tensor 'inputs_4_copy:0' shape=(None, 1) dtype=int64>,
  'rating': <tf.Tensor 'inputs_5_copy:0' shape=(None, 1) dtype=int64>,
  'timestamp': <tf.Tensor 'inputs_6_copy:0' shape=(None, 1) dtype=int64>,
  'user_id': <tf.Tensor 'inputs_7_copy:0' shape=(None, 1) dtype=int64>}

  outputs={
  'user_id': <tf.Tensor 'inputs_7_copy:0' shape=(None, 1) dtype=int64>,
  'movie_id': <tf.Tensor 'inputs_3_copy:0' shape=(None, 1) dtype=int64>,
  'rating': <tf.Tensor 'truediv:0' shape=(None, 1) dtype=float32>,
  'gender': <tf.Tensor 'None_Lookup/LookupTableFindV2:0' shape=(None, 1) dtype=int64>,
  'age': <tf.Tensor 'None_Lookup_1/LookupTableFindV2:0' shape=(None, 1) dtype=int64>,
  'occupation': <tf.Tensor 'inputs_4_copy:0' shape=(None, 1) dtype=int64>,
  'genres': <tf.Tensor 'RaggedToTensor_1/RaggedTensorToTensor:0' shape=(None, 1, 18) dtype=float32>,
  'hr': <tf.Tensor 'FloorMod:0' shape=(None, 1) dtype=int64>,
  'weekday': <tf.Tensor 'Add:0' shape=(None, 1) dtype=int64>,
  'hr_wk': <tf.Tensor 'Add_1:0' shape=(None, 1) dtype=int64>,
  'month': <tf.Tensor 'Cast_8:0' shape=(None, 1) dtype=int64>}
  """
  logging.debug(f"inputs={inputs}")

  outputs = {'user_id': inputs['user_id'],
             'movie_id': inputs['movie_id']}

  outputs['rating'] = tf.divide(tf.cast(inputs['rating'], tf.float32), \
    tf.constant(5.0, dtype=tf.float32))

  gender_table = create_static_table(genders, var_dtype=tf.string)
  outputs['gender'] = gender_table.lookup(inputs['gender'])
  #outputs['gender'] = tf.one_hot( outputs['gender'], depth=len(genders), dtype=tf.int64)

  age_groups_table = create_static_table(age_groups, var_dtype=tf.int64)
  outputs['age'] = age_groups_table.lookup(inputs['age'])
  #outputs['age'] = tf.one_hot( outputs['age'], depth=len(age_groups), dtype=tf.int64)
  
  outputs['occupation'] = inputs['occupation']
  #outputs['occupation'] = tf.one_hot(outputs['occupation'], depth=num_occupations, dtype=tf.int64)

  def transform_genres(input_genres):
    genres_table = create_static_table(genres, var_dtype=tf.string)
    out = tf.strings.regex_replace(
      input=input_genres, pattern="Children's", rewrite="Children")
    out = tf.strings.split(out, "|")
    # need fixed length tensors for tf.lookup.StaticHashTable
    pad_shape = [i for i in out.shape]
    pad_shape[-1] = len(genres) #or [-2]?
    fixed = out.to_tensor(default_value="<PAD>", shape=pad_shape)
    idx = genres_table.lookup(fixed)
    filtered = tf.ragged.boolean_mask(idx, tf.not_equal(idx, -1))
    oh = tf.one_hot(indices=filtered, depth=len(genres))
    m_genres = tf.reduce_sum(oh, axis=-2)
    norm = tf.reduce_sum(m_genres, axis=-1)
    norm = tf.expand_dims(norm, axis=-1)
    res = tf.divide(m_genres, norm)
    #change from RaggedTensor to Tensor:
    res = res.to_tensor(shape=[None, 1, res.shape[-1]])
    return res

  #omitting zipcode for now, but considering ZCTAs for future

  logging.debug(f"inputs['genres']={inputs['genres']}")
  outputs['genres'] = transform_genres(inputs['genres'])

  #diff due to leap sec is < 1 minute total since 1972

  #chicago is -5 hours from UTC = 60sec*60min*5hr = 18000
  chicago_tz_offset = tf.constant(18000, dtype=tf.int64)
  #ts is w.r.t. 1970 01 01, Thursday, 0 hr
  ts = tf.subtract(inputs["timestamp"], chicago_tz_offset)

  outputs["hr"] = tf.cast(tf.round(tf.divide(\
    tf.cast(ts, dtype=tf.float64), tf.constant(3600., dtype=tf.float64))),\
    dtype=tf.int64)
  outputs["hr"] = tf.math.mod(outputs['hr'], tf.constant(24, dtype=tf.int64))

  days_since_1970 = tf.cast(tf.round(tf.divide(\
    tf.cast(ts, dtype=tf.float64), tf.constant(86400., dtype=tf.float64))),\
    dtype=tf.int64)

  outputs["weekday"] = tf.math.mod(days_since_1970, tf.constant(7, dtype=tf.int64))
  #week starting on Monday
  outputs["weekday"] = tf.add(outputs["weekday"], tf.constant(4, dtype=tf.int64))
  #a cross of hour and weekday: hr * 7 + weekday
  outputs["hr_wk"] = tf.add(tf.multiply(outputs["hr"], tf.constant(7, dtype=tf.int64)),\
      outputs["weekday"])

  #there is probably a relationship between genres and month, so calc month too.
  outputs["month"] = tf.cast(tf.round(\
    tf.divide(tf.cast(days_since_1970, tf.float64), tf.constant(30, dtype=tf.float64))), dtype=tf.int64)

  ## year can be useful for timeseries analysis.
  ## there is a leap year every 4 years, starting at 1972.
  ## month calc using 30 days for every month, roughly.
  ## 365 days for non-leap years, 366 for leap years
  ## number of Leap years = (dY // 4)
  ## (dY - (dY//4)) * 365 + (dY//4) * 366 = days_since_1970
  ## dY = days_since_1970 / (365 - 365/4 + 366/4)
  #dy = tf.cast(tf.round(\
  #  tf.divide(\
  #    tf.cast(days_since_1970, dtype=tf.float32),\
  #    tf.constant(365 - (365./4) + (366./4), dtype=tf.float32))), dtype=tf.int64)
  #outputs["year"] = tf.add(tf.constant(1970, dtype=tf.int64), dy)

  logging.debug(f"outputs={outputs}")

  return outputs

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
