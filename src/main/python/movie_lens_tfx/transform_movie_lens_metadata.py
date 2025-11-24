"""for the movie metadata model"""
import tensorflow as tf
import tensorflow_transform as tft
from absl import logging
from tensorflow_transform.tf_metadata import schema_utils

# from tfx.components.trainer import fn_args_utils
# from tfx_bsl.tfxio import dataset_options
logging.set_verbosity(logging.WARNING)
logging.set_stderrthreshold(logging.WARNING)

## fixed vocabularies, known ahead of time
#genres = ["Action", "Adventure", "Animation", "Children", "Comedy",
#          "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
#          "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
#          "Thriller", "War", "Western"]
genres = [b'Action', b'Adventure', b'Animation', b'Children', b'Comedy',
          b'Crime', b'Documentary', b'Drama', b'Fantasy', b'Film-Noir',
          b'Horror', b'Musical', b'Mystery', b'Romance', b'Sci-Fi',
          b'Thriller', b'War', b'Western']

# Tf.Transform considers these features as "raw"
def _get_raw_feature_spec(schema):
  return schema_utils.schema_as_feature_spec(schema).feature_spec

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
    'movie_id': <tf.Tensor 'Cast_1:0' shape=(None, 1) dtype=float32>,
    'rating': <tf.Tensor 'truediv:0' shape=(None, 1) dtype=float32>,
    'genres': <tf.Tensor 'RaggedToTensor_1/RaggedTensorToTensor:0' shape=(None, 1, 18) dtype=float32>,
      }
  """
  #tf.print(f"inputs={inputs}")
  
  # make efficient static graphs for the tables
  with tf.init_scope():
    def create_static_table(var_list, var_dtype):
        init = tf.lookup.KeyValueTensorInitializer(
          keys=tf.constant(var_list, dtype=var_dtype),
          values=tf.range(len(var_list), dtype=tf.int64),
          key_dtype=var_dtype, value_dtype=tf.int64)
        # The table is created and initialized *eagerly* and only once here.
        return tf.lookup.StaticHashTable(init, default_value=-1)
    genres_table = create_static_table(genres, var_dtype=tf.string)
    
  outputs = {'movie_id': tf.cast(inputs['movie_id'], dtype=tf.float32)}

  outputs['rating'] = tf.divide(tf.cast(inputs['rating'], tf.float32), \
    tf.constant(5.0, dtype=tf.float32))

  def transform_genres(input_genres):
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
