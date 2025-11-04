import tensorflow as tf
import tensorflow_transform as tft
from absl import logging
from tensorflow_transform.tf_metadata import schema_utils

# from tfx.components.trainer import fn_args_utils
# from tfx_bsl.tfxio import dataset_options
logging.set_verbosity(logging.INFO)
logging.set_stderrthreshold(logging.INFO)

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
    'user_id': <tf.Tensor 'Cast:0' shape=(None, 1) dtype=float32>,
    'movie_id': <tf.Tensor 'Cast_1:0' shape=(None, 1) dtype=float32>,
    'rating': <tf.Tensor 'truediv:0' shape=(None, 1) dtype=float32>,
    'gender': <tf.Tensor 'Cast_3:0' shape=(None, 1) dtype=float32>,
    'age': <tf.Tensor 'Cast_4:0' shape=(None, 1) dtype=float32>,
    'occupation': <tf.Tensor 'Cast_5:0' shape=(None, 1) dtype=float32>,
    'genres': <tf.Tensor 'RaggedToTensor_1/RaggedTensorToTensor:0' shape=(None, 1, 18) dtype=float32>,
    'hr': <tf.Tensor 'Cast_8:0' shape=(None, 1) dtype=float32>,
    'weekday': <tf.Tensor 'Cast_9:0' shape=(None, 1) dtype=float32>,
    'hr_wk': <tf.Tensor 'Cast_10:0' shape=(None, 1) dtype=float32>,
    'month': <tf.Tensor 'Cast_11:0' shape=(None, 1) dtype=float32>,
    'yr': <tf.Tensor 'Cast_12:0' shape=(None, 1) dtype=float32>,
    'sec_into_yr': <tf.Tensor 'Cast_13:0' shape=(None, 1) dtype=float32>
      }
  """
  #tf.print(f"inputs={inputs}")

  outputs = {'user_id': tf.cast(inputs['user_id'], dtype=tf.float32),
             'movie_id': tf.cast(inputs['movie_id'], dtype=tf.float32)}

  outputs['rating'] = tf.divide(tf.cast(inputs['rating'], tf.float32), \
    tf.constant(5.0, dtype=tf.float32))

  gender_table = create_static_table(genders, var_dtype=tf.string)
  outputs['gender'] = tf.cast(gender_table.lookup(inputs['gender']), dtype=tf.float32)
  #outputs['gender'] = tf.one_hot( outputs['gender'], depth=len(genders), dtype=tf.int64)

  age_groups_table = create_static_table(age_groups, var_dtype=tf.int64)
  outputs['age'] = tf.cast(age_groups_table.lookup(inputs['age']), dtype=tf.float32)
  #outputs['age'] = tf.one_hot( outputs['age'], depth=len(age_groups), dtype=tf.int64)
  
  outputs['occupation'] = tf.cast(inputs['occupation'], dtype=tf.float32)
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
  #tf.print("ts=", ts)
  
  outputs["hr"] = tf.math.floordiv(ts,  tf.constant(3600, dtype=tf.int64))
  outputs["hr"] = tf.math.mod(outputs['hr'], tf.constant(24, dtype=tf.int64))

  days_since_1970 = tf.math.floordiv(ts, tf.constant(86400, dtype=tf.int64))

  outputs["weekday"] = tf.math.mod(days_since_1970, tf.constant(7, dtype=tf.int64))
  #week starting on Monday
  outputs["weekday"] = tf.add(outputs["weekday"], tf.constant(4, dtype=tf.int64))
  #a cross of hour and weekday: hr * 7 + weekday.  range is [0,168]. in UsrModel, tf.keras.layers.Embedding further modtransforms
  outputs["hr_wk"] = tf.add(tf.multiply(outputs["hr"], tf.constant(7, dtype=tf.int64)),
    outputs["weekday"])
  
  #there is probably a relationship between genres and month, so calc month too.
  outputs["month"] = tf.math.floordiv(days_since_1970, tf.constant(30, dtype=tf.int64))
  
  for key in ["hr", "weekday", "hr_wk", "month"]:
    outputs[key] = tf.cast(outputs[key], dtype=tf.float32)
  
  ## adding year and sec_into_yr to later reconstruct timestamp for prefixspan, time series, etc
  _365 = tf.constant(365, dtype=tf.int64)
  _366 = tf.constant(366, dtype=tf.int64)
  _1 = tf.constant(1, dtype=tf.int64)
  _2 = tf.constant(2, dtype=tf.int64)
  _366_plus_3_times_365 = tf.constant(1461, dtype=tf.int64)
  _2_times_365 = tf.constant(730, dtype=tf.int64)
  days_since_1972 = tf.subtract(days_since_1970, _2_times_365)
  ## n_ly = 1 + (days_since_1972//(366 + 365*3))
  ## n_non_ly = (days_since_1972 - n_ly*366)//365
  ## yr = 1972 + n_ly + n_non_ly
  ## sec_into_yr = ts - (((366 * n_ly) + (365 * (n_non_ly + 2))) * 24 * 60 * 60)
  #    range is [0, 31536000 || 31622400]
  
  n_ly = tf.add(_1, tf.math.floordiv(days_since_1972, _366_plus_3_times_365))
  n_non_ly = tf.math.floordiv(tf.subtract(days_since_1972, tf.multiply(n_ly, _366)), _365)
  outputs["yr"] = tf.add(tf.add(tf.constant(1972, dtype=tf.int64), n_ly), n_non_ly)
  outputs["yr"] = tf.cast(outputs["yr"], tf.float32)
  ##sec_into_yr = ts - (((366 * n_ly) + (365 * (n_non_ly + 2))) * 24 * 60 * 60)
  #   where +2 is for 1970, 1971
  _t1 = tf.multiply(_366, n_ly)
  _t2 = tf.multiply(_365, tf.add(n_non_ly, _2))
  outputs["sec_into_yr"] = tf.subtract(ts,
      tf.multiply(tf.add(_t1, _t2), tf.constant(24 * 60 * 60, dtype=tf.int64)))
  outputs["sec_into_yr"] = tf.cast(outputs["sec_into_yr"], dtype=tf.float32)
  
  #tf.print("sec_into_yr=", outputs["sec_into_yr"])
  #tf.print(f"outputs={outputs}")

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
