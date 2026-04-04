# from
import base64
import pickle
# some code is adapted from https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils_base.py
# and related files
# they have co Copyright 2020 Google LLC. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
from typing import List, Tuple, Dict, Text, Any
import tensorflow as tf
import tensorflow.keras as keras
#import tf_keras as keras ## this fails
import enum
import os
import math
import json
import keras_tuner
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils
from tfx.types.standard_artifacts import Model
from tensorflow_metadata.proto.v0 import statistics_pb2
#tuner needs this:
from tfx.components.trainer.fn_args_utils import FnArgs

from tensorboard.plugins.hparams.api import hparams
# from tensorflow.python.ops.gen_experimental_dataset_ops import save_dataset
from tfx import v1 as tfx

from tfx_bsl.public import tfxio

from absl import logging

logging.set_verbosity(logging.WARNING)
logging.set_stderrthreshold(logging.WARNING)

'''
builds pipelines for training a TwoTowerDNN model to train Query and Candidate
embedding models.  The training is optimized using Contrastive Learning for a
Listwise Discriminative Model.

The run_fn defines the model, compile, fit and signatures.
The tuner_fn specifies that the custom metric "val_hit_rate" should be used
to decide which model is best.
'''

DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 20
DEFAULT_NUM_EXAMPLES = 100000

MAX_TUNE_TRIALS_DEFAULT = 10
EXECUTIONS_PER_TRIAL_DEFAULT = 1

#NOTE: could be improved by writing the headers to a file in the Transform stage and reading them here:
LABEL_KEY = 'rating'
N_GENRES = 18
N_AGE_GROUPS = 7

class Device(enum.Enum):
  CPU = "CPU"
  GPU = "GPU"
  TPU = "TPU"

package = "ttdnn"

# https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/examples/penguin/penguin_utils_base.py#L98
def input_fn(file_pattern: List[str],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  return data_accessor.tf_dataset_factory(
    file_pattern,
    tfxio.TensorFlowDatasetOptions(batch_size=batch_size, label_key=LABEL_KEY),
    tf_transform_output.transformed_metadata.schema).repeat().prefetch(
    tf.data.AUTOTUNE)


def _make_2tower_keras_model(hp: keras_tuner.HyperParameters) -> tf.keras.Model:
  
  #input_dataset_element_spec_raw = hp.get("input_dataset_element_spec_raw_ser")
  #input_dataset_element_spec_raw = pickle.loads(base64.b64decode(input_dataset_element_spec_raw.encode('utf-8')))
  
  input_dataset_element_spec_trans = hp.get("input_dataset_element_spec_trans_ser")
  input_dataset_element_spec_trans = pickle.loads(base64.b64decode(input_dataset_element_spec_trans.encode('utf-8')))
  
  @keras.utils.register_keras_serializable(package=package)
  class CyclicalEncoding(keras.layers.Layer):
    def __init__(self, max_val, **kwargs):
      super().__init__(**kwargs)
      self.max_val = max_val
    def call(self, inputs):
      radians = 2 * math.pi * tf.cast(inputs, tf.float32) / self.max_val
      return tf.concat([tf.sin(radians), tf.cos(radians)], axis=-1)
    def get_config(self):
      config = super(CyclicalEncoding, self).get_config()
      config.update({"max_val": self.max_val})
  
  @keras.utils.register_keras_serializable(package=package)
  class UserModel(keras.Model):
    # for init from a load, arguments are present for the compositional instance members too
    def __init__(self, max_user_id: int,
                 embed_out_dim: int = 32,
                 feature_acronym: str = "",
                 **kwargs):
      """
      a user feature model to create an initial vector of features for the QueryModel.
      NOTE: the user_ids are expected to be already unique and represented by range [1, n_users] and dtype np.int32.
      No integerlookup to rewrite to smaller number of ids is used here because the ratings and user data
      are densely populated for user.

      Args:
          n_users: the total number of users

          feature_acronym: a string of alphabetized single letters for each of the following to be in the embedding:
              a for age
              h for hr_wk cross
              m for month
              o for occupation
              s for gender
      """
      super(UserModel, self).__init__(**kwargs)
      self.embed_out_dim = embed_out_dim
      self.max_user_id = max_user_id
      self.feature_acronym = feature_acronym
      
      #NOTE: it is up to the using component to filter for OOV values
      #      to avoid using this incorrectly
      self.user_embedding = keras.Sequential([
        keras.layers.Embedding(self.max_user_id + 1, embed_out_dim),
        keras.layers.Flatten(data_format='channels_last'),
      ], name="user_emb")
      
      # ordinal, dist between items matters
      self.age_embedding = None
      if self.feature_acronym.find("a") > -1:
        self.age_embedding = keras.Sequential([
          keras.layers.Dense(embed_out_dim, activation='swish',
              kernel_initializer='he_normal', bias_initializer='zeros'),
          keras.layers.Flatten(data_format='channels_last'),
        ], name="age_emb")
        
      # ordinal, dist between items matters
      self.yr_z_embedding = None
      if self.feature_acronym.find("y") > -1:
        self.yr_z_embedding = keras.Sequential([
          keras.layers.Dense(embed_out_dim, activation='swish',
                kernel_initializer='he_normal',
                bias_initializer='zeros'),
            keras.layers.Flatten(data_format='channels_last'),
        ], name="yr_z_emb")
      
      # numerical, cyclical
      self.hr_wk_embedding = None
      if self.feature_acronym.find("h") > -1:
        self.hr_wk_embedding = keras.Sequential([
          CyclicalEncoding(max_val=24*7),
          keras.layers.Flatten(data_format='channels_last'),
        ], name="hr_wk_emb")
      
      #numerical, cyclical
      self.month_embedding = None
      if self.feature_acronym.find("m") > -1:
        self.month_embedding = keras.Sequential([
            CyclicalEncoding(max_val=12),
            keras.layers.Flatten(data_format='channels_last'),
        ], name="month_emb")
      
      # categorical, nominal, order doesn't matter
      self.occupation_embedding = None
      if self.feature_acronym.find("o") > -1:
        self.occupation_embedding = keras.layers.CategoryEncoding(
          # num_tokens=21, \
          num_tokens=embed_out_dim,
          output_mode="one_hot", name="occupation_emb")
      
      # categorical
      self.gender_embedding = None
      if self.feature_acronym.find("s") > -1:
        self.gender_embedding = keras.layers.CategoryEncoding(
          # num_tokens=2,
          num_tokens=embed_out_dim,
          output_mode="one_hot", name="gender_emb")
    
    def build(self, input_shape):
      # print(f'build {self.name} input_shape={input_shape}\n')
      self.user_embedding.build(input_shape['user_id'])
      if self.age_embedding:
        self.age_embedding.build(input_shape['age'])
      if self.yr_z_embedding:
        self.yr_z_embedding.build(input_shape['yr_z'])
      if self.hr_wk_embedding:
        self.hr_wk_embedding.build(input_shape['hr_wk'])
      if self.month_embedding:
        self.month_embedding.build(input_shape['month'])
      if self.occupation_embedding:
        self.occupation_embedding.build(input_shape['occupation'])
      if self.gender_embedding:
        self.gender_embedding.build(input_shape['gender'])
      # print(f'build {self.name} User {self.embed_out_dim} =>{self.user_embedding.compute_output_shape(input_shape['user_id'])}\n')
      self.built = True
    
    def compute_output_shape(self, input_shape):
      # print(f'compute_output_shape {self.name} input_shape={input_shape}\n')
      # This is invoked after build by QueryModel.
      # return (None, self.embed_out_dim)
      _shape = self.user_embedding.compute_output_shape(input_shape['user_id'])
      total_length = _shape[-1]
      if self.age_embedding:
        _shape = self.age_embedding.compute_output_shape(input_shape['age'])
        total_length += _shape[-1]
      if self.yr_z_embedding:
          _shape = self.yr_z_embedding.compute_output_shape(input_shape['yr_z'])
          total_length += _shape[-1]
      if self.hr_wk_embedding:
        _shape = self.hr_wk_embedding.compute_output_shape(
          input_shape['hr_wk'])
        total_length += _shape[-1]
      if self.month_embedding:
        _shape = self.month_embedding.compute_output_shape(
          input_shape['month'])
        total_length += _shape[-1]
      if self.occupation_embedding:
        _shape = self.occupation_embedding.compute_output_shape(
          input_shape['occupation'])
        total_length += _shape[-1]
      if self.gender_embedding:
        _shape = self.gender_embedding.compute_output_shape(
          input_shape['gender'])
        total_length += _shape[-1]
      return None, total_length
      # return (input_shape['movie_id'][0], total_length)
      # return self.user_embedding.compute_output_shape(input_shape['movie_id'])
    
    def call(self, inputs, **kwargs):
      # Take the input dictionary, pass it through each input layer,
      # and concatenate the result.
      # arrays are: 'user_id',  'gender', 'age_group', 'occupation','movie_id', 'rating'
      # print(f'call {self.name} type={type(inputs)}\n')
      # tf.print(inputs)
      results = []
      results.append(self.user_embedding(inputs['user_id']))
      if self.age_embedding:
        results.append(self.age_embedding(inputs['age']))
      if self.yr_z_embedding:
        results.append(self.yr_z_embedding(inputs['yr_z']))
      if self.hr_wk_embedding:
        results.append(self.hr_wk_embedding(inputs['hr_wk']))
      if self.month_embedding:
        results.append(self.month_embedding(inputs['month']))
      if self.occupation_embedding:
        results.append(self.occupation_embedding(inputs['occupation']))
      if self.gender_embedding:
        results.append(self.gender_embedding(inputs['gender']))
      res = keras.layers.Concatenate()(results)
      #logging.debug(f'call {self.name} SHAPE ={res.shape}')
      #tf.print('CALL', self.name, ' shape=', res.shape)
      return res
    
    def get_config(self):
      config = super(UserModel, self).get_config()
      config.update({"max_user_id": self.max_user_id,
                     "embed_out_dim": self.embed_out_dim,
                     "feature_acronym": self.feature_acronym,
                     })
      return config
  
  @keras.utils.register_keras_serializable(package=package)
  class MovieModel(keras.Model):
    """
    NOTE: the movie_ids are expected to be already unique and represented by range [1, n_movies] and dtype np.int32.
      No integerlookup to rewrite to smaller number of ids is used here because ratings.dat uses 96% of the
      movies.dat ids.
    """
    
    # for init from a load, arguments are present for the compositional instance members too
    def __init__(self, n_movies: int, movies_offset:int, n_genres: int,
                 embed_out_dim: int = 32, incl_genres: bool = True,
                 **kwargs):
      super(MovieModel, self).__init__(**kwargs)
      
      self.embed_out_dim = embed_out_dim
      self.n_movies = n_movies
      self.movies_offset = movies_offset
      self.n_genres = n_genres
      self.incl_genres = incl_genres
      # out_dim = int(np.sqrt(in_dim)) ~ 64
      
      #NOTE: it is up to the using component to filter for OOV values
      #      to avoid using this incorrectly
      self.movie_embedding = keras.Sequential([
        keras.layers.IntegerLookup(
            vocabulary=[i for i in range(movies_offset+1, movies_offset + n_movies + 1)],
            output_mode="int"),
        keras.layers.Embedding(self.n_movies + 1, self.embed_out_dim),
        keras.layers.Flatten(data_format='channels_last'),
      ], name="movie_emb")
      
      if self.incl_genres:
        # expand to embed_out_dim for concatenation
        self.genres_embedding = keras.Sequential([
          keras.layers.Dense(self.embed_out_dim),
          keras.layers.Flatten(data_format='channels_last'),
        ], name="genres_emb")
    
    def build(self, input_shape):
      #tf.print("build", self.name, "input_shape=:", input_shape)
      #tf.print(f"OUTPUT shapes:", self.movie_embedding.compute_output_shape( input_shape['movie_id']))
      self.movie_embedding.build(input_shape['movie_id'])
      if self.incl_genres:
        self.genres_embedding.build(input_shape['genres'])
        #tf.print(self.genres_embedding.compute_output_shape(input_shape['genres']))
      self.built = True
    
    def compute_output_shape(self, input_shape):
      # print(f'compute_output_shape {self.name} input_shape={input_shape}\n')
      # This is invoked after build by CandidateModel
      _shape = self.movie_embedding.compute_output_shape(input_shape['movie_id'])
      total_length = _shape[-1]
      if self.incl_genres:
        _shape = self.genres_embedding.compute_output_shape(input_shape['genres'])
        total_length += _shape[-1]
      return None, total_length
    
    def call(self, inputs, **kwargs):
      # Take the input dictionary, pass it through each input layer,
      # and concatenate the result.
      # print(f'call {self.name} type={type(inputs)}, kwargs={kwargs}\n')
      # print(f'    spec={inputs.element_spec}\n')
      x = tf.cast(inputs['movie_id'], dtype=tf.int32)
      results = [self.movie_embedding(x)]
      # shape is (batch_size, x, out_dim)
      if self.incl_genres:
        results.append(self.genres_embedding(inputs['genres']))
      #tf.print('concatenate shapes:', [r.shape for r in results])
      res = keras.layers.Concatenate(axis=-1)(results)
      #tf.print('call result,shape=', res.shape)
      # logging.debug(f'call {self.name} SHAPE ={res.shape}')
      #tf.print('CALL', self.name, ' shape=', res.shape)
      return res
    
    def get_config(self):
      # updating super config stomps over existing key names, so if need separate values one would need
      # to use some form of package and class name in keys or uniquely name the keys to avoid collision
      config = super(MovieModel, self).get_config()
      config.update(
        {"n_movies": self.n_movies, "movies_offset": self.movies_offset, "n_genres": self.n_genres,
         "embed_out_dim": self.embed_out_dim,
         'incl_genres': self.incl_genres
         })
      return config
  
  @keras.utils.register_keras_serializable(package=package)
  class QueryModel(keras.Model):
    """Model for encoding user queries."""
    
    # for init from a load, arguments are present for the compositional instance members too
    def __init__(self, n_users: int,
                 layer_sizes: list,
                 embed_out_dim: int = 32,
                 regl2:float = 0.0,
                 drop_rate: float = 0., feature_acronym: str = "",
                 **kwargs):
      """Model for encoding user queries.

              Args:
        layer_sizes:
          A list of integers where the i-th entry represents the number of units
          the i-th layer contains.
      """
      super(QueryModel, self).__init__(**kwargs)
      
      self.regl2 = regl2
      
      self.embedding_model = UserModel(max_user_id=n_users,
                                       embed_out_dim=embed_out_dim,
                                       feature_acronym=feature_acronym)
      if isinstance(layer_sizes, str):
        layer_sizes = json.loads(layer_sizes)
      
      self.dense_layers = keras.Sequential(name="dense_query")
      reg = None
      # Use the ReLU activation for all but the last layer.
      for layer_size in layer_sizes[:-1]:
        if self.regl2 > 0.0:
            reg = keras.regularizers.l2(self.regl2)
        self.dense_layers.add(
          keras.layers.Dense(layer_size, activation="elu",
                             kernel_regularizer=reg,
                             kernel_initializer="glorot_normal"))
        # self.dense_layers.add(keras.layers.BatchNormalization())
        self.dense_layers.add(keras.layers.LayerNormalization())
        self.dense_layers.add(keras.layers.Dropout(drop_rate))
      
      for layer_size in layer_sizes[-1:]:
        self.dense_layers.add(keras.layers.Dense(layer_size,
                                                 kernel_initializer="glorot_normal"))
      
      self.dense_layers.add(keras.layers.UnitNormalization(axis=-1))
      
      self.n_users = n_users
      self.feature_acronym = feature_acronym
      self.embed_out_dim = embed_out_dim
      self.layer_sizes = layer_sizes
      self.drop_rate = drop_rate
    
    def build(self, input_shape):
      # print(f'build {self.name} input_shape={input_shape}\n')
      self.embedding_model.build(input_shape)
      input_shape_2 = self.embedding_model.compute_output_shape(input_shape)
      self.dense_layers.build(input_shape_2)
      self.built = True
    
    def compute_output_shape(self, input_shape):
      # print(f'compute_output_shape {self.name} input_shape={input_shape}, {input_shape['user_id'][0]}, {self.layer_sizes[-1:]}\n')
      # This is invoked after build by TwoTower
      # return self.output_shapes[0]
      input_shape_3 = self.dense_layers.compute_output_shape(
        self.embedding_model.compute_output_shape(input_shape))
      _shape_3 = [i for i in input_shape_3]
      _shape_3[0] = None
      return _shape_3
      # return None, self.layer_sizes[-1]
      # return (input_shape['user_id'][0], self.layer_sizes[-1])
    
    def call(self, inputs, **kwargs):
      # inputs should contain columns:
      #print(f'call {self.name} type={type(inputs)}, inputs={inputs}\n')
      feature_embedding = self.embedding_model(inputs, **kwargs)
      res = self.dense_layers(feature_embedding)
      #tf.print('CALL', self.name, ' shape=', res.shape)
      return res
    
    def get_config(self):
      config = super(QueryModel, self).get_config()
      config.update({"n_users": self.n_users,
                     "embed_out_dim": self.embed_out_dim,
                     "drop_rate": self.drop_rate,
                     "layer_sizes": self.layer_sizes,
                     "feature_acronym": self.feature_acronym,
                     "regl2": self.regl2,
                     })
      return config
    
    @classmethod
    def from_config(cls, config):
      return cls(**config)
  
  @keras.utils.register_keras_serializable(package=package)
  class CandidateModel(keras.Model):
    """Model for encoding candidate features."""
    
    # for init from a load, arguments are present for the compositional instance members too
    def __init__(self, n_movies: int, movies_offset:int, n_genres: int, layer_sizes,
                 embed_out_dim: int = 32,
                 regl2: float = 0.0,
                 drop_rate: float = 0., incl_genres: bool = True,
                 **kwargs):
      """Model for encoding candidate features.

      Args:
        layer_sizes:
          A list of integers where the i-th entry represents the number of units
          the i-th layer contains.
      """
      super(CandidateModel, self).__init__(**kwargs)
      
      self.regl2 = regl2
      
      self.embedding_model = MovieModel(n_movies=n_movies, movies_offset=movies_offset,
        n_genres=n_genres,
        embed_out_dim=embed_out_dim,
        incl_genres=incl_genres, name = "movie_emb")
      
      self.dense_layers = keras.Sequential(name="dense_candidate")
      if isinstance(layer_sizes, str):
        layer_sizes = json.loads(layer_sizes)
      reg = None
      # Use the ReLU activation for all but the last layer.
      for layer_size in layer_sizes[:-1]:
        if self.regl2 > 0.0:
          reg = keras.regularizers.l2(self.regl2)
        self.dense_layers.add(
          keras.layers.Dense(layer_size, activation="elu",
                             kernel_regularizer=reg,
                             kernel_initializer="glorot_normal"))
        # self.dense_layers.add(keras.layers.BatchNormalization())
        self.dense_layers.add(keras.layers.LayerNormalization())
        self.dense_layers.add(keras.layers.Dropout(drop_rate))
      
      for layer_size in layer_sizes[-1:]:
        self.dense_layers.add(keras.layers.Dense(layer_size,
          kernel_initializer="glorot_normal"))
      
      self.dense_layers.add(keras.layers.UnitNormalization(axis=-1))
      
      self.n_movies = n_movies
      self.movies_offset = movies_offset
      self.n_genres = n_genres
      self.incl_genres = incl_genres
      self.embed_out_dim = embed_out_dim
      self.drop_rate = drop_rate
      self.layer_sizes = layer_sizes
      
    def build(self, input_shape):
      # print(f'build {self.name} input_shape={input_shape}\n')
      self.embedding_model.build(input_shape)
      input_shape_2 = self.embedding_model.compute_output_shape(input_shape)
      self.dense_layers.build(input_shape_2)
      self.built = True
    
    def compute_output_shape(self, input_shape):
      # print(f'compute_output_shape {self.name} input_shape={input_shape}\n')
      # This is invoked after build by TwoTower
      input_shape_3 = self.dense_layers.compute_output_shape(
        self.embedding_model.compute_output_shape(input_shape))
      _shape_3 = [i for i in input_shape_3]
      _shape_3[0] = None
      return _shape_3
      # return None, self.layer_sizes[-1]
    
    def call(self, inputs, **kwargs):
      # inputs should contain columns "movie_id", "genres"
      # logging.debug(f'call {self.name} type ={type(inputs)}\ntype ={inputs}\n')
      feature_embedding = self.embedding_model(inputs, **kwargs)
      #tf.print('invoked movie_emb.  shape=', feature_embedding.shape)
      res = self.dense_layers(feature_embedding)
      # returns an np.ndarray wrapped in a tensor if inputs is tensor, else not wrapped
      # logging.debug(f'CALL {self.name} SHAPE ={res.shape}\n')
      #tf.print('CALL', self.name, ' shape=', res.shape)
      return res
    
    def get_config(self):
      config = super(CandidateModel, self).get_config()
      config.update(
        {"n_movies": self.n_movies, "movies_offset":self.movies_offset,
          "n_genres": self.n_genres,
         "embed_out_dim": self.embed_out_dim,
         "drop_rate": self.drop_rate,
         "layer_sizes": self.layer_sizes,
         "regl2": self.regl2,
         "incl_genres": self.incl_genres
         })
      return config
    
    @classmethod
    def from_config(cls, config):
      return cls(**config)
  
  @keras.utils.register_keras_serializable(package=package)
  class TwoTowerDNN(keras.Model):
    """
    a Two-Tower (bi-encoder) DNN model that accepts input containing: user, context, and item information along with
    a label for training.

    when use_bias_corr is true, the Yi et al. paper is followed to calculate the item sampling probability
    within a mini-batch which is then used to correct probabilities and the batch loss sum.

    the number of layers is controlled by a list of their sizes in layer_sizes.
    
    The model trains the Query and Candidate models that are downstream used as a Retrieval model.
    TwoTowerDNN is optimized using In-Batch Negative Contrastive Learning and is a Listwise Discriminative Model.
    """
    
    # for init from a load, arguments are present for the compositional instance members too
    def __init__(self, n_users: int, n_movies: int, movies_offset: int,
         n_genres: int,
         layer_sizes: list, embed_out_dim: int,
         regl2: float = 0.0,
         drop_rate: float = 0,
         feature_acronym: str = "",
         use_bias_corr: bool = False,
         bias_corr_alpha: float=0.1,
         incl_genres: bool = True,
         temperature:float=1.0, **kwargs):
      super(TwoTowerDNN, self).__init__(**kwargs)
      
      if isinstance(layer_sizes, str):
        layer_sizes = json.loads(layer_sizes)
      
      self.query_model = QueryModel(n_users=n_users,
                                    layer_sizes=layer_sizes,
                                    embed_out_dim=embed_out_dim,
                                    regl2=regl2,
                                    drop_rate=drop_rate,
                                    feature_acronym=feature_acronym,
                                    **kwargs)
      
      self.candidate_model = CandidateModel(n_movies=n_movies, movies_offset=movies_offset,
                                            n_genres=n_genres,
                                            layer_sizes=layer_sizes,
                                            embed_out_dim=embed_out_dim,
                                            regl2=regl2,
                                            drop_rate=drop_rate,
                                            incl_genres=incl_genres,
                                            **kwargs)
      
      self.dot_layer = keras.layers.Dot(axes=1)
      self.loss_tracker = tf.keras.metrics.Mean(name="loss")
      self.hit_rate_metric = InBatchHitRate(name="hit_rate")
      
      self.regl2 = regl2
      
      self.n_users = n_users
      self.n_movies = n_movies
      self.movies_offset = movies_offset
      self.n_genres = n_genres
      self.incl_genres = incl_genres
      self.layer_sizes = layer_sizes
      self.feature_acronym = feature_acronym
      self.embed_out_dim = embed_out_dim
      self.drop_rate = drop_rate
      
      self.use_bias_corr = use_bias_corr
      self.bias_corr_alpha = bias_corr_alpha #for batch_size>=512 alpha ~ 0.01 else 0.1
      self.temperature = temperature
      
      if self.use_bias_corr:
          # Persistent state for item frequency estimation
          # A stores the last 't' (global step) the movie was seen
          self.table_A = tf.lookup.experimental.MutableHashTable(
              key_dtype=tf.int32, value_dtype=tf.float32, default_value=0.)
          # B stores the estimated probability (p_i)
          self.table_B = tf.lookup.experimental.MutableHashTable(
              key_dtype=tf.int32, value_dtype=tf.float32, default_value=1e-6)
          self.global_step = tf.Variable(0., trainable=False,
              dtype=tf.float32)
    
    @property
    def metrics(self):
        # OVERRIDE to workaround tf.keras handling of validation metrics
        # It tells the model: "When you finish an epoch, pull results from these two."
        return [self.loss_tracker, self.hit_rate_metric]
    
    @tf.function(input_signature=[input_dataset_element_spec_trans])
    def call(self, inputs):
      """
      compute the cosine similarity score for the user data to movie data.
      Args:
         inputs: transformed features
      Returns:
          cosine similarity score for the user data to movie data
      """
      #logging.debug(f'call {self.name} inputs={inputs}\n')
      user_vector = self.query_model(inputs)
      movie_vector = self.candidate_model(inputs)
      #tf.print('U,V SHAPES: ', user_vector.shape, movie_vector.shape)
      s = self.dot_layer([user_vector, movie_vector])
      return s
      
    @tf.function(input_signature=[input_dataset_element_spec_trans])
    def serve_query_model(self, inputs):
      """A dedicated function to trace and serve the trained Query Model."""
      return self.query_model(inputs)  #
    
    @tf.function(input_signature=[input_dataset_element_spec_trans])
    def serve_candidate_model(self, inputs):
      """A dedicated function to trace and serve the trained Candidate Model."""
      return self.candidate_model(inputs)  #
    
    def build(self, input_shape):
      # print(f'build {self.name} input_shape={input_shape}\n')
      # logging.debug(f'build {self.name} input_shape={input_shape}\n')
      self.query_model.build(input_shape)
      self.candidate_model.build(input_shape)
      s0 = self.query_model.compute_output_shape(input_shape)
      s1 = self.candidate_model.compute_output_shape(input_shape)
      self.dot_layer.build([s0, s1])
      s2 = self.dot_layer.compute_output_shape([s0, s1])
      self.built = True
    
    def compute_output_shape(self, input_shape):
      # (batch_size,)  a scalar for each row in batch
      # return input_shape['user_id']
      s0 = self.query_model.compute_output_shape(input_shape)
      s1 = self.candidate_model.compute_output_shape(input_shape)
      s2 = self.dot_layer.compute_output_shape([s0, s1])
      _shape_3 = [i for i in s2]
      _shape_3[0] = None
      return _shape_3
      # return (None,)
    
    def _update_frequencies(self, movie_ids):
        """Streaming frequency estimation logic from Yi et al."""
        self.global_step.assign_add(1.0)
        t = self.global_step
        
        movie_ids_int = tf.cast(movie_ids, tf.int32)
        movie_ids_flat = tf.reshape(movie_ids_int, [-1])
        
        last_t = self.table_A.lookup(movie_ids_flat)
        p_old = self.table_B.lookup(movie_ids_flat)
        
        # B = (1 - alpha) * B + alpha * (t - last_t)
        p_new = (1.0 - self.bias_corr_alpha) * p_old + self.bias_corr_alpha * (t - last_t)
        
        self.table_A.insert(movie_ids_flat, tf.fill(tf.shape(movie_ids_flat), t))
        self.table_B.insert(movie_ids_flat, p_new)
        return p_new
    
    def train_step(self, batch):
        x, y = batch  # y is typically not used in pure In-Batch Softmax (identity matrix is the target)
        movie_ids = x['movie_id']
        
        with tf.GradientTape() as tape:
            user_embeddings = self.query_model(x)  # [Batch, Dim]
            movie_embeddings = self.candidate_model(x)  # [Batch, Dim]
            
            # Compute ALL-TO-ALL Similarity (In-Batch Softmax)
            # scores[i, j] is similarity between user i and movie j
            # this is [batch_size X batch_size] and the diagonal is the dot product
            logits = tf.matmul(user_embeddings, movie_embeddings, transpose_b=True)
            logits = logits/self.temperature
            
            if self.use_bias_corr:
                # Get frequency corrections
                p_i = self._update_frequencies(movie_ids)
                log_q = tf.math.log(p_i)
                # Apply Log-Q correction to columns (the candidate side)
                # Broad-casting log_q across the batch
                logits = logits - tf.expand_dims(log_q, axis=0)
            
            # Define Targets
            # In-Batch Softmax target is the diagonal (user i liked movie i)
            batch_size = tf.shape(logits)[0]
            labels = tf.range(batch_size)
            labels = tf.reshape(labels, [-1])  # Forces it to be a 1D vector
            per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
            # Multiply by your scaled ratings (y)
            # If y is 0.1, the gradient for that user is tiny.
            # If y is 1.0, the gradient is full strength.
            loss = tf.reduce_mean(per_example_loss * tf.cast(y, tf.float32))
        
        # Optimization...
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.loss_tracker.update_state(loss)
        self.hit_rate_metric.update_state(y_true=labels, y_pred=logits, sample_weight=y)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        x, y = data
        user_embeddings = self.query_model(x, training=False)
        movie_embeddings = self.candidate_model(x, training=False)
        logits = tf.matmul(user_embeddings, movie_embeddings, transpose_b=True)
        logits = logits / self.temperature
        
        if self.use_bias_corr:
            # We use the frequencies learned during training
            movie_ids_keys = tf.cast(tf.reshape(x['movie_id'], [-1]),
                tf.int32)
            p_i = self.table_B.lookup(movie_ids_keys)
            logits = logits - tf.expand_dims(tf.math.log(p_i), axis=0)
        
        # Define Ranking Labels
        batch_size = tf.shape(logits)[0]
        labels = tf.range(batch_size)
        labels = tf.reshape(labels, [-1])  # Forces it to be a 1D vector
        
        # Calculate Loss (Weighted by y if you used that in train_step)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        )
        self.loss_tracker.update_state(loss)
        self.hit_rate_metric.update_state(y_true=labels, y_pred=logits, sample_weight=y)
        return {m.name: m.result() for m in self.metrics}
    
    def get_config(self):
      config = super(TwoTowerDNN, self).get_config()
      config.update({"n_users": self.n_users, "n_movies": self.n_movies,
        "movies_offset" : self.movies_offset,
        "n_genres": self.n_genres,
        "embed_out_dim": self.embed_out_dim,
        "drop_rate": self.drop_rate,
        "layer_sizes": self.layer_sizes,
        "use_bias_corr": self.use_bias_corr,
        "feature_acronym": self.feature_acronym,
        "regl2": self.regl2,
        "incl_genres": self.incl_genres,
        "bias_corr_alpha": self.bias_corr_alpha,
        "temperature": self.temperature,
        })
      return config
    
    @classmethod
    def from_config(cls, config):
      return cls(**config)
  
  @keras.utils.register_keras_serializable(package=package)
  class InBatchHitRate(keras.metrics.Metric):
      def __init__(self, name="batch_hit_rate", **kwargs):
          super(InBatchHitRate, self).__init__(name=name, **kwargs)
          self.hits = self.add_weight(name="total_hits",
              initializer="zeros")
          self.count = self.add_weight(name="total_count",
              initializer="zeros")
      
      def update_state(self, y_true, y_pred, sample_weight=None):
          """
          y_true: Ignored here (internally generated as tf.range), or used for weights
          y_pred: The [Batch, Batch] logits matrix
          sample_weight: Your ratings (y) from the dataset
          """
          # 1. Get the current batch size dynamically
          batch_size = tf.shape(y_pred)[0]
          
          # 2. Find the predicted index (the movie with the highest similarity)
          # y_pred shape: [Batch, Batch] -> preds shape: [Batch]
          preds = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
          
          # 3. Create the ground truth (the diagonal indices)
          targets = tf.range(batch_size, dtype=tf.int32)
          
          # 4. Compare: [Batch] boolean vector
          is_correct = tf.equal(preds, targets)
          is_correct = tf.cast(is_correct, tf.float32)
          
          # 5. Apply your ratings as weights if provided
          if sample_weight is not None:
              sample_weight = tf.cast(sample_weight, tf.float32)
              # Ensure sample_weight is 1D to match is_correct
              sample_weight = tf.reshape(sample_weight, [-1])
              is_correct = tf.multiply(is_correct, sample_weight)
              self.count.assign_add(tf.reduce_sum(sample_weight))
          else:
              self.count.assign_add(tf.cast(batch_size, tf.float32))
          
          self.hits.assign_add(tf.reduce_sum(is_correct))
      
      def result(self):
          return tf.math.divide_no_nan(self.hits, self.count)
      
      def reset_state(self):
          self.hits.assign(0.0)
          self.count.assign(0.0)
  
  @keras.utils.register_keras_serializable(package=package)
  class HeuristicLambdaLoss(keras.losses.Loss):
      """
      a loss for use in systems that do not have a re-ranker
      """
      def __init__(self, name="heuristic_lambda", temperature:float=0.07, **kwargs):
          super(HeuristicLambdaLoss, self).__init__(name=name, **kwargs)
          self.temperature = temperature
      
      def call(self, y_true, y_pred):
          """
          y_true: [batch_size, batch_size], is not used.  is assumed to be identity matric
          y_pred:  [batch_size, batch_size].  the result of matmul Q_embedd, C_embed^T
          """
          logits = y_pred/self.temperature
          pos_indices = tf.range(tf.shape(logits)[0])
          pos_scores = tf.linalg.diag_part(logits)
          
          #current rank of pos item in its row.
          # use a differentiable approx or stop gradient for the rank
          ranks = tf.reduce_sum(tf.cast(logits > pos_scores[:, tf.newaxis]), axis=1)
          
          #heuristic rank: 1/(log2(1 + rank)
          # compute stop gradient to avoid bacpro thru rank logic
          weights = 1.0/tf.math.log1p(tf.stop_gradient(ranks) + 1.0)
          
          #cross entropy scaled by weights
          loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pos_indices, logits=logits)
          
          return tf.reduce_mean(loss * weights)
      
      def get_config(self):
          config = super(HeuristicLambdaLoss, self).get_config()
          config.update({
              "temperature": self.temperature,
          })
          return config
      
  @keras.utils.register_keras_serializable(package=package)
  class MeanReciprocalRankAtK(keras.metrics.Metric):
      """
      """
      def __init__(self, name="mrr", k:int=10,**kwargs):
          name = f"{name}_k{k}"
          super(MeanReciprocalRankAtK, self).__init__(name=name, **kwargs)
          self.k = tf.cast(k, tf.float32)
          self.mrr_sum = self.add_weight(name="mrr_sum", initializer="zeros")
          self.count = self.add_weight(name="count", initializer="zeros")

      def update_state(self, y_true, y_pred, sample_weight=None):
          """
          y_true: Ignored here (internally generated as tf.range), or used for weights.  use an identity
                  matrix with same shape as y_pred
          y_pred: The [Batch, Batch] logits matrix result of matmul Q_embedd * C_embed^T
          """
          pos_scores = tf.linalg.diag_part(y_pred)[:, tf.newaxis]
          ranks = tf.reduce_sum(tf.cast(y_pred >= pos_scores, tf.float32), axis=1) + 1.0
          
          reciprocal_rank = tf.where(
              ranks <= self.k, 1.0/ranks, 0.0
          )
          self.mrr_sum.assign_add(tf.reduce_sum(reciprocal_rank))
          self.count.assign_add(tf.cast(tf.shape(y_pred)[0], tf.float32))
         
      def result(self):
          return self.mrr_sum / self.count
      
      def reset_state(self):
          self.mrr_sum.assign(0.0)
          self.count.assign(0.0)
          
      def get_config(self):
          config = super(MeanReciprocalRankAtK, self).get_config()
          config.update({
              "k": float(self.k.numpy())
          })
          return config
      
  @keras.utils.register_keras_serializable(package=package)
  class NDCGAtKForInBatchNegatives(keras.metrics.Metric):
      """
      """
      def __init__(self, name="ndcg", k: int = 20, **kwargs):
          name = f"{name}_k{k}"
          super(NDCGAtKForInBatchNegatives, self).__init__(name=name, **kwargs)
          self.k = tf.cast(k, tf.float32)
          self.ndcg_sum = self.add_weight(name="ndcg_sum",
              initializer="zeros")
          self.count = self.add_weight(name="count", initializer="zeros")
      
      def update_state(self, y_true, y_pred, sample_weight=None):
          """
          y_true: Ignored here (internally generated as tf.range), or used for weights.  use an identity
                  matrix with same shape as y_pred
          y_pred: The [Batch, Batch] logits matrix result of matmul Q_embedd * C_embed^T
          """
          pos_scores = tf.linalg.diag_part(y_pred)[:, tf.newaxis]
          ranks = tf.reduce_sum(tf.cast(y_pred >= pos_scores, tf.float32),
              axis=1) + 1.0
          log2_rank = tf.math.log(ranks + 1.0) / tf.math.log(2.0)
          dcg = 1./ log2_rank
          
          #IDCG is 1.0 because best possible rank is 1. for in batch negatives
          ndcg_at_k = tf.where(
              ranks <= self.k, dcg, 0.0
          )
          self.ndcg_sum.assign_add(tf.reduce_sum(ndcg_at_k))
          self.count.assign_add(tf.cast(tf.shape(y_pred)[0], tf.float32))
      
      def result(self):
          return self.ndcg_sum / self.count
      
      def reset_state(self):
          self.ndcg_sum.assign(0.0)
          self.count.assign(0.0)
      
      def get_config(self):
          config = super(NDCGAtKForInBatchNegatives, self).get_config()
          config.update({
              "k": float(self.k.numpy())
          })
          return config
      
  @keras.utils.register_keras_serializable(package=package)
  class RecallAtKForInBatchNegatives(keras.metrics.Metric):
      """
      same as hit_rate_at_k for in batch negatives
      """
      def __init__(self, name="ndcg", k: int = 100, **kwargs):
          name = f"{name}_k{k}"
          super(RecallAtKForInBatchNegatives, self).__init__(name=name, **kwargs)
          self.k = tf.cast(k, tf.float32)
          self.hits = self.add_weight(name="hits", initializer="zeros")
          self.count = self.add_weight(name="count", initializer="zeros")
      
      def update_state(self, y_true, y_pred, sample_weight=None):
          """
          y_true: Ignored here (internally generated as tf.range), or used for weights.  use an identity
                  matrix with same shape as y_pred
          y_pred: The [Batch, Batch] logits matrix result of matmul Q_embedd * C_embed^T
          """
          pos_scores = tf.linalg.diag_part(y_pred)[:, tf.newaxis]
          ranks = tf.reduce_sum(tf.cast(y_pred >= pos_scores, tf.float32),
              axis=1)
          
          is_hit = tf.cast(ranks <= self.k, tf.float32)
          
          self.hits.assign_add(tf.reduce_sum(is_hit))
          self.count.assign_add(tf.cast(tf.shape(y_pred)[0], tf.float32))
      
      def result(self):
          return self.hits / self.count
      
      def reset_state(self):
          self.hits.assign(0.0)
          self.count.assign(0.0)
      
      def get_config(self):
          config = super(RecallAtKForInBatchNegatives, self).get_config()
          config.update({
              "k": float(self.k.numpy())
          })
          return config
  
  # use strategy
  d = hp.get("device")
  if d == "GPU":
    device = Device.GPU
  elif d == "TPU":
    device = Device.TPU
  else:
    device = Device.CPU
  strategy, device = _get_strategy(device)
  
  #METRICS_FN_LIST = [InBatchHitRate(name="hit_rate")]
  #tf.keras.metrics.SparseCategoricalAccuracy(name="acc")
  
  with strategy.scope():
    model = TwoTowerDNN(
      n_users=hp.get("n_users") + 1,
      n_movies=hp.get("n_movies") + 1,
      movies_offset = hp.get("n_users") + 1,
      n_genres=hp.get("n_genres"),
      layer_sizes=hp.get('layer_sizes'),
      embed_out_dim=hp.get('embed_out_dim'),
      regl2=hp.get('regl2'),
      drop_rate=hp.get('drop_rate'),
      feature_acronym=hp.get("feature_acronym"),
      use_bias_corr=hp.get('use_bias_corr'),
      bias_corr_alpha = hp.get('bias_corr_alpha'),
      incl_genres=hp.get('incl_genres'),
      temperature=hp.get('temperature'),
    )
    
    # call once to trace methods.
    fake_trans_ds = create_fake_transformed_batch(input_dataset_element_spec_trans)
    model(fake_trans_ds, training=False)
    
    BATCH_SIZE_PER_REPLICA = hp.get("BATCH_SIZE")
    NUM_EPOCHS = hp.get("NUM_EPOCHS")
    n_replicas = strategy.num_replicas_in_sync
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * n_replicas
    # virtual epochs:
    TRAIN_STEPS_PER_EPOCH = math.ceil(hp.get("num_train") / GLOBAL_BATCH_SIZE)
    TOTAL_STEPS = TRAIN_STEPS_PER_EPOCH * NUM_EPOCHS
    #NOTE: warmup should be 10-20% of total steps
    WARMUP_STEPS = TOTAL_STEPS//10
    
    lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=TOTAL_STEPS,
        warmup_steps=WARMUP_STEPS,
        warmup_target=hp.get('learning_rate')
    )
    
    optimizer = keras.optimizers.AdamW(learning_rate=lr_scheduler,
        weight_decay=hp.get("weight_decay"))

    #NOTE: do not set metrics here as they are hard-coded in model
    model.compile(
        loss=None, # internally fixed to sparse softmax cross entropy for logits
        optimizer=optimizer,
        run_eagerly=hp.get("run_eagerly")
    )
  
  model.summary(print_fn=logging.info)
  
  is_tf_dot_keras_model = isinstance(model, tf.keras.Model)
  is_keras_model = isinstance(model, keras.Model)
  is_keras_models_Model = isinstance(model, keras.models.Model)
  logging.debug(f"is_tf_dot_keras_model: {is_tf_dot_keras_model}, "
    f"is_keras_model: {is_keras_model}, is_keras_models_Model={is_keras_models_Model}")
  if not isinstance(model, keras.models.Model):
    logging.debug(f'this is the fail at tuner.py line 167')
  # TO debug, user run_eagerly=False
  return model


def get_default_hyperparameters(custom_config) -> keras_tuner.HyperParameters:
  """Returns hyperparameters for building Keras model."""
  #print(f'get_default_hyperparameters: custom_config={custom_config}')
  hp = keras_tuner.HyperParameters()
  # Defines search space.
  hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
  hp.Float('weight_decay', 1e-6, 1e-2, sampling='log')
  hp.Float('regl2', 1e-6, 1e-4, sampling="log")
  hp.Float('drop_rate', min_value=0.1, max_value=0.5, default=0.5)
  hp.Choice("embed_out_dim", values=[32], default=32)
  #layers_sizes is a list of ints, so encode each list as a string, chices can only be int,float,bool,str
  hp.Choice("layer_sizes", values=[json.dumps([32])], default=json.dumps([32]))
  # ahmos for "age", "hr_wk", "month", "occupation", "gender"
  hp.Fixed("feature_acronym", custom_config.get("feature_acronym", "h"))
  hp.Fixed("incl_genres", custom_config["incl_genres"])
  hp.Fixed('BATCH_SIZE', custom_config.get("BATCH_SIZE", DEFAULT_BATCH_SIZE))
  hp.Fixed('NUM_EPOCHS', custom_config.get("NUM_EPOCHS", DEFAULT_NUM_EPOCHS))
  #use_bias_corr = hp.Choice("use_bias_corr", values=[True, False], default=True)
  use_bias_corr = hp.Fixed("use_bias_corr", value=True)
  if use_bias_corr:
      hp.Choice("bias_corr_alpha", values=[0.01, 0.05, 0.1], default=0.1) #0.01, 0.05, 0.1
      hp.Float('temperature', 0.05, 0.2, step=0.05)
  else:
      hp.Choice("bias_corr_alpha", values=[0.1], default=0.1)  # 0.01, 0.05, 0.1
      hp.Choice("temperature", values=[1.0], default=1.0)
  hp.Fixed('n_users', value=custom_config["n_users"])
  hp.Fixed('n_movies', custom_config["n_movies"])
  hp.Fixed('n_genres', custom_config["n_genres"])
  hp.Fixed('run_eagerly', custom_config["run_eagerly"])
  hp.Fixed('device', custom_config.get("device", 'CPU'))
  hp.Fixed('MAX_TUNE_TRIALS', custom_config.get("MAX_TUNE_TRIALS", MAX_TUNE_TRIALS_DEFAULT))
  hp.Fixed('EXECUTIONS_PER_TRIAL', custom_config.get("EXECUTIONS_PER_TRIAN", EXECUTIONS_PER_TRIAL_DEFAULT))
  num_examples = custom_config.get("num_examples", DEFAULT_NUM_EXAMPLES)
  num_train = int(num_examples * 0.8)
  num_eval = int(num_examples * 0.1)
  hp.Fixed("num_train", num_train)
  hp.Fixed("num_eval", num_eval)
  hp.Fixed('version', custom_config.get("version", "1.0.0"))
  if "model_name" in custom_config:
    hp.Fixed('model_name', custom_config["model_name"])
  if "team_lead" in custom_config:
    hp.Fixed('team_lead', custom_config["team_lead"])
  if "git_hash" in custom_config:
    hp.Fixed('git_hash', custom_config["git_hash"])
  
  return hp

# TFX Trainer will call this function.
def _get_strategy(device: Device) -> Tuple[tf.distribute.Strategy, Device]:
  strategy = None
  if device == Device.GPU:
    try:
      device_physical = tf.config.list_physical_devices('GPU')[0]
      #batch_size is the global batch size
      strategy = tf.distribute.MirroredStrategy(
        devices=[device_physical])  # or [device_physical.name]
      tf.config.optimizer.set_jit(False)
      #or choose MultiWorkerMirroredStrategy
    except Exception as ex:
      logging.error(ex)
      strategy = None
  # or: # tf.distribute.OneDeviceStrategy(device=device_physical)
  elif device == Device.TPU:
    if tf.config.list_physical_devices('TPU'):
      try:
        device_name = os.environ.get('TPU_NAME')
        logging.info(f"os.environ TPU_NAME={device_name}")
        # device_physical = tf.config.list_physical_devices('TPU')[0]
        # tf.config.set_visible_devices(device_physical, 'TPU')
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
          tpu='local')
        # instantiate a distribution strategy
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        # https://www.kaggle.com/docs/tpu
        # TPU v3-8 on Kaggle has 8 cores.  increase batch_size MXU is not near 100% in TPU monitor
        # e.g. batch_size = 16 * strategy.num_replicas_in_sync
        tf.config.optimizer.set_jit(False)
        logging.info(
          f"TPU is available and set as default device:{tpu.master()}")
      except Exception as ex2:
        logging.error(ex2)
        strategy = None
    else:
      logging.debug("TPU is not available")
      strategy = None
  
  if not strategy:
    strategy = tf.distribute.get_strategy()
    #or use MirroredStrategy
    device = Device.CPU
  return strategy, device

def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example."""

  # the layer is added as an attribute to the model in order to make sure that
  # the model assets are handled correctly when exporting.
  model.tft_layer = tf_transform_output.transform_features_layer()

def create_fake_transformed_batch(input_signature):
    dummy_batch = {}
    for feat, config in input_signature.items():
        raw_shape = config.shape.as_list() if hasattr(config.shape, 'as_list') else []
        shape = [1] + [dim if dim is not None else 1 for dim in raw_shape[1:]]
        #shape = [_ if _ is not None else 1 for _ in config.shape]
        dtype = config.dtype
        if dtype == tf.string:
            dummy_batch[feat] = tf.constant([[""]], dtype=tf.string)
        else:
            dummy_batch[feat] = tf.zeros(shape, dtype=dtype)
    return dummy_batch

def convert_feature_spec_to_tensor_spec(raw_feature_spec: Dict) -> Dict[
  str, tf.TensorSpec]:
  """
  Converts a raw_feature_spec() dictionary (containing Feature objects)
  to a dictionary of tf.TensorSpec objects.
    # Example Usage:
  # raw_spec = tf_transform_output.raw_feature_spec()
  # raw_tensor_spec = convert_feature_spec_to_tensor_spec(raw_spec)

  """
  tensor_spec = {}
  
  for name, feature in raw_feature_spec.items():
    # Handle FixedLenFeature (most common)
    if isinstance(feature, tf.io.FixedLenFeature):
      if name == "genres":
        if len(feature.shape) == 1:
          _shape = (None, feature.shape[0])
        elif len(feature.shape) == 2:
          _shape = (None, feature.shape[0], feature.shape[1])
        else:
          raise ValueError(f"Feature shape {feature.shape} is not supported.")
        tensor_spec[name] = tf.TensorSpec(
          shape=_shape,
          dtype=feature.dtype,
          name=name
        )
      else:
        tensor_spec[name] = tf.TensorSpec(
          shape=(None, feature.shape[0]),
          dtype=feature.dtype,
          name=name
        )
    
    # Handle VarLenFeature (uncommon for raw data, but needed if present)
    elif isinstance(feature, tf.io.VarLenFeature):
      # VarLen features are typically represented by a RaggedTensor
      # or a sparse tensor after parsing. When requesting a TensorSpec
      # for input, we usually define the shape as partially dynamic.
      # However, for simple use cases, the shape of the resulting
      # dense tensor is [None] or [None, ...].
      tensor_spec[name] = tf.TensorSpec(
        shape=[None],  # Unknown length
        dtype=feature.dtype,
        name=name
      )
    
    # Handle SparseFeature (more complex and requires multiple TensorSpecs: indices, values, dense_shape)
    # For simplicity, we can skip or raise an error for complex types here,
    # but a full utility would handle them explicitly.
    elif isinstance(feature, tf.io.SparseFeature):
      # Sparse features are usually handled by the parsing function itself,
      # which returns a SparseTensor (or RaggedTensor), not a single dense TensorSpec.
      raise NotImplementedError(
        f"Conversion for SparseFeature '{name}' is complex and not included.")
    
    else:
      raise TypeError(
        f"Unsupported feature type for '{name}': {type(feature)}")
  
  return tensor_spec


# tfx.components.FnArgs
def run_fn(fn_args):
  """Train the model based on given args.
  
  expects hyperparameters or cutom_config, but not both
  
  fn_args = fn_args_utils.get_common_fn_args(input_dict, exec_properties,
    working_dir)
    where exec_properties are the PARAMETERS from the Tuner Spec
    and working_dir is from the Executor's get_tmp_dir()
https://github.com/tensorflow/tfx/blob/master/tfx/types/standard_component_specs.py
  fn_args: Holds args as name/value pairs.
      - working_dir (supplied by software): working dir for tuning.
      - train_files (supplied by software, obtained from examples):
          List of file paths containing training tf.Example data.
      - eval_files (supplied by software, obtained from examples):
          List of file paths containing eval tf.Example data.
      - train_steps (from train_args):
          number of train steps.
      - eval_steps (from eval_args:
          number of eval steps.
      - schema_path (optional, supplied already by graph):
         schema of the input data.
      - transform_graph_path (required):
         transform graph produced by TFT.
      - model_path
      - custom_config (required):
          'n_users'
          'n_movies'
          'n_genres'
          'run_eagerly'
          'device'

    fn_args.hyperparameters (required) : keras_tuner.HyperParameters with keys
      'lr'
      "regl2"
      "drop_rate"
      "embed_out_dim"
      "layer_sizes"
      "feature_acronym"
      "incl_genres"
      'num_epochs'
      'batch_size'
      "use_bias_corr"
      'n_users'
      'n_movies'
      'n_genres'
      'run_eagerly'
      'device'

    other Example:
      module_file=os.path.abspath(_trainer_module_file),
      examples=ratings_transform.outputs['transformed_examples'],
      transform_graph=ratings_transform.outputs['transform_graph'],
      schema=ratings_transform.outputs['post_transform_schema'],
      train_args=tfx.proto.TrainArgs(num_steps=500),
      eval_args=tfx.proto.EvalArgs(num_steps=10),
      custom_config={
          'epochs':5,
          'movies':movies_transform.outputs['transformed_examples'],
          'movie_schema':movies_transform.outputs['post_transform_schema'],
          'ratings':ratings_transform.outputs['transformed_examples'],
          'ratings_schema':ratings_transform.outputs['post_transform_schema']
          'device' : 'TPU' or 'GPU' of 'CPU', if none, CPU will be used
          })
  """
  #print(f"RUN_FN fn_args={fn_args}")
  for attr_name in dir(fn_args):
    # Filter out built-in methods and private attributes
    if not attr_name.startswith('__') and not callable(
      getattr(fn_args, attr_name)):
      attr_value = getattr(fn_args, attr_name)
      logging.debug(f"{attr_name}: {attr_value}")
  
  if not fn_args.hyperparameters:
    raise ValueError('hyperparameters must be provided')
  
  hp = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
  
  print('HyperParameters for training: %s' % hp.get_config())
  
  d = hp.get("device")
  if d == "GPU":
    device = Device.GPU
  elif d == "TPU":
    device = Device.TPU
  else:
    device = Device.CPU
  strategy, device = _get_strategy(device)
  
  logging.info(f"device={device}, distribution strategy={strategy}")
  
  BATCH_SIZE_PER_REPLICA = hp.get("BATCH_SIZE")
  NUM_EPOCHS = hp.get("NUM_EPOCHS")
  
  n_replicas = strategy.num_replicas_in_sync
  GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * n_replicas
  
  # virtual epochs:
  TRAIN_STEPS_PER_EPOCH = math.ceil(hp.get("num_train") / GLOBAL_BATCH_SIZE)
  EVAL_STEPS_PER_EPOCH = math.ceil(hp.get("num_eval") / GLOBAL_BATCH_SIZE)
  
  # for run_fn, fn_args.transform_output is not None
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
  input_signature_raw = convert_feature_spec_to_tensor_spec(tf_transform_output.raw_feature_spec())
  input_signature_trans = convert_feature_spec_to_tensor_spec(tf_transform_output.transformed_feature_spec())
  del input_signature_raw[LABEL_KEY]
  del input_signature_trans[LABEL_KEY]

  try:
      _ = hp.get('input_dataset_element_spec_trans_ser')
  except Exception:
      hp.Fixed('input_dataset_element_spec_trans_ser',
          (base64.b64encode(
              pickle.dumps(input_signature_trans))).decode('utf-8'))
  
  train_dataset = input_fn(
    fn_args.train_files,
    fn_args.data_accessor,
    tf_transform_output,
    GLOBAL_BATCH_SIZE)
  
  eval_dataset = input_fn(
    fn_args.eval_files,
    fn_args.data_accessor,
    tf_transform_output,
    GLOBAL_BATCH_SIZE)
  
  #the model is built and compiled in strategy scope:
  model = _make_2tower_keras_model(hp)
  # model = _make_2tower_keras_model(hp, tf_transform_output)

  # Write logs to path
  tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=fn_args.model_run_dir, update_freq='epoch')
  
  stop_early = keras.callbacks.EarlyStopping(
    monitor=f'val_hit_rate', min_delta=1E-4, patience=3, mode="max",
    restore_best_weights=True)
  
  """
  checkpoint_dir = os.path.join(fn_args.serving_model_dir, 'checkpoint')
  filepath = os.path.join(
    checkpoint_dir, 'best_model_{epoch:02d}-{val_loss:.2f}'  # Using val_loss is common
  )
  tf.io.gfile.makedirs(checkpoint_dir)
  callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath, monitor='val_loss', verbose=1, mode='min', save_best_only=True,
    save_weights_only=False, save_freq='epoch')
  """
  history = model.fit(
    train_dataset,
    steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
    validation_data=eval_dataset,
    validation_steps=EVAL_STEPS_PER_EPOCH,
    epochs=NUM_EPOCHS,
    callbacks=[tensorboard_callback, stop_early], verbose=1)
  
  print(f'fit history.history={history.history}')
  
  #TODO: consider adding the vocabularies as assets:
  #    see https://www.tensorflow.org/api_docs/python/tf/saved_model/Asset
  
  """
  latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
  if latest_checkpoint:
    model.load_weights(latest_checkpoint)
    print(f"Loaded best weights from {latest_checkpoint}")
  """
  
  #from TFX codebase: https://github.com/tensorflow/tfx/blob/v1.16.0/tfx/examples/penguin/penguin_utils_base.py
  def _make_raw_serving_signatures(model, tf_transform_output: tft.TFTransformOutput):
    """Returns the serving signatures.

    Args:
      model: the model function to apply to the transformed features.
      tf_transform_output: The transformation to apply to the serialized
        tf.Example.

    Returns:
      The signatures to use for saving the mode. The 'serving_default' signature
      will be a concrete function that takes a batch of unspecified length of
      serialized tf.Example, parses them, transformes the features and
      then applies the model.
      Similarly, "serving_query" and "serving_candidate" signature take batches of
      unspecified length of serialized tf.Example, parse them, transformes them,
      then applies the embedding models.
      The 'transform_features' signature parses the
      example and transforms the features.
    """
    
    # We need to track the layers in the model in order to save it.
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    def serve_tf_examples_fn(serialized_tf_example):
      '''Returns the serving signature for input being raw examples such as
      inputs = tf.data.TFRecordDataset(examples_file_paths, compression_type="GZIP")
      where examples_file_paths was written by MovieLensSplitExampleGen
      '''
      raw_feature_spec = tf_transform_output.raw_feature_spec()
      try:
        raw_feature_spec.pop(LABEL_KEY)
      except KeyError as e:
        logging.error(f'ERROR: {e}')
      
      raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
      
      transformed_features = model.tft_layer(raw_features)
      outputs = model(inputs=transformed_features, training=False)
      return {'outputs': outputs}
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    def serve_query_tf_examples_fn(serialized_tf_example):
      '''
      Returns the serving signature query embeddings for input being raw examples, not yet transformed to features.
      '''
      raw_feature_spec = tf_transform_output.raw_feature_spec()
      try:
        raw_feature_spec.pop(LABEL_KEY)
      except KeyError as e:
        logging.error(f'ERROR: {e}')
      raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
      transformed_features = model.tft_layer(raw_features)
      outputs = model.query_model(inputs=transformed_features, training=False)
      return {'outputs': outputs}
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    def serve_candidate_tf_examples_fn(serialized_tf_example):
      '''
      Returns the serving signature candidate embeddings for input being raw examples, not yet transformed to features.
      '''
      raw_feature_spec = tf_transform_output.raw_feature_spec()
      try:
        raw_feature_spec.pop(LABEL_KEY)
      except KeyError as e:
        logging.error(f'ERROR: {e}')
      raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
      transformed_features = model.tft_layer(raw_features)
      outputs = model.candidate_model(inputs=transformed_features, training=False)
      return {'outputs': outputs}
    
    @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
      '''Returns the transformed_features to be fed as input to evaluator.  inputs are the raw
      examples from MovieLensSplitExampleGen
      '''
      raw_feature_spec = tf_transform_output.raw_feature_spec()
      logging.debug(f'transform_features_fn spec = {raw_feature_spec}')
      raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
      transformed_features = model.tft_layer(raw_features)
      logging.info('eval_transformed_features = %s',transformed_features)
      return transformed_features
    
    @tf.function(input_signature=[input_signature_raw])
    def serve_query_dict_fn(raw_features):
      '''
      given raw inputs dictionary of tensors, transforms the data and returns the outputs of query model on transformed data.
      '''
      transformed_features = model.tft_layer(raw_features)
      outputs = model.query_model(inputs=transformed_features, training=False)
      return {'outputs': outputs}
    
    @tf.function(input_signature=[input_signature_raw])
    def serve_candidate_dict_fn(raw_features):
      '''
      given raw inputs dictionary of tensors, transforms the data and returns the outputs of candidate model on transformed data.
      '''
      transformed_features = model.tft_layer(raw_features)
      outputs = model.candidate_model(inputs=transformed_features, training=False)
      return {'outputs': outputs}
    
    @tf.function(input_signature=[input_signature_raw])
    def serve_default_dict_fn(raw_features):
      transformed_features = model.tft_layer(raw_features)
      outputs = model(inputs=transformed_features, training=False)
      return {'outputs': outputs}
    
    return {
      'serving_default': serve_tf_examples_fn,
      'transform_features': transform_features_fn,
      'serving_candidate': serve_candidate_tf_examples_fn,
      'serving_query': serve_query_tf_examples_fn,
      "serving_default_dict": serve_default_dict_fn,
      "serving_query_dict" : serve_query_dict_fn,
      "serving_candidate_dict": serve_candidate_dict_fn
    }
  
  signatures = {}
  other_sigs = _make_raw_serving_signatures(model, tf_transform_output)
  signatures["transform_features"] = other_sigs["transform_features"]
  signatures["serving_default"] = other_sigs["serving_default"]
  signatures["serving_query"] = other_sigs["serving_query"]
  signatures["serving_candidate"] = other_sigs["serving_candidate"]
  signatures["serving_default_dict"] = other_sigs["serving_default_dict"]
  signatures["serving_query_dict"] = other_sigs["serving_query_dict"]
  signatures["serving_candidate_dict"] = other_sigs["serving_candidate_dict"]
  
  tf.saved_model.save(model, fn_args.serving_model_dir, signatures=signatures)
  
  #the model signatures expected as input are positional keywords ordere.
  # to see the epected order, use saved_model_cli show --dir <path_to_format-serving-dir> --all
  
  #loaded_saved_model = tf.saved_model.load(fn_args.serving_model_dir)
  #print(f'loaded SavedModel signatures: {loaded_saved_model.signatures}')
  #infer = loaded_saved_model.signatures["serving_default"]
  #print(f'infer.structured_outputs={infer.structured_outputs}')
  
  return model

# TFX Tuner will call this function.
def tuner_fn(fn_args) -> tfx.components.TunerFnResult:
  """Build the tuner using the KerasTuner API.

  expects hyperparameters or cutom_config, but not both
  
  fn_args = fn_args_utils.get_common_fn_args(input_dict, exec_properties,
    working_dir)
    where exec_properties are the PARAMETERS from the Tuner Spec
    and working_dir is from the Executor's get_tmp_dir()
    https://github.com/tensorflow/tfx/blob/master/tfx/types/standard_component_specs.py#L390

  Args:
    fn_args: Holds args as name/value pairs.
      - working_dir (supplied by software): working dir for tuning.
      - train_files (supplied by software, obtained from examples):
          List of file paths containing training tf.Example data.
      - eval_files (supplied by software, obtained from examples):
          List of file paths containing eval tf.Example data.
      - train_steps (from train_args):
          number of train steps.
      - eval_steps (from eval_args):
          number of eval steps.
      - schema_path (optional, supplied already by graph):
         schema of the input data.
      - transform_graph_path (required):
         transform graph produced by TFT.
      - model_path
      - custom_config (required):
          'n_users'
          'n_movies'
          'n_genres'
          'run_eagerly'

  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
  """
  # RandomSearch is a subclass of keras_tuner.Tuner which inherits from
  # BaseTuner.
  
  #FnArgs should be from tfx.components.trainer.fn_args_utils
  #print(f"TUNER_FN fn_args={fn_args}")
  logging.debug(f"Working directory: {fn_args.working_dir}")
  logging.debug(f"Training files: {fn_args.train_files}")
  logging.debug(f"Evaluation files: {fn_args.eval_files}")
  logging.debug(f"Transform graph path: {fn_args.transform_graph_path}")
  logging.debug(f"data_accessor: {fn_args.data_accessor}")
  logging.debug(f"Hyperparameters: {fn_args.hyperparameters}")
  logging.debug(f"Custom config: {fn_args.custom_config}")
  
  #fn_args.transform_output is None so use transform_graph_ath instead
  
  if fn_args.hyperparameters:
    hp = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
  else:
    hp = get_default_hyperparameters(fn_args.custom_config)
  
  ## because _make_2tower_keras_model needs these specs for signatures, we store them as fixed hyperpareters
  # Also oote that the tuner method needs to use fn_args.transform_graph_path
  transform_graph = tft.TFTransformOutput(fn_args.transform_graph_path)
  input_signature_raw = convert_feature_spec_to_tensor_spec(transform_graph.raw_feature_spec())
  input_signature_trans = convert_feature_spec_to_tensor_spec(transform_graph.transformed_feature_spec())
  del input_signature_raw[LABEL_KEY]
  del input_signature_trans[LABEL_KEY]
  
  hp.Fixed('input_dataset_element_spec_raw_ser',
      (base64.b64encode(pickle.dumps(input_signature_raw))).decode('utf-8'))
  hp.Fixed('input_dataset_element_spec_trans_ser',
      (base64.b64encode(pickle.dumps(input_signature_trans))).decode('utf-8'))
    
  d = hp.get("device")
  if d == "GPU":
    device = Device.GPU
  elif d == "TPU":
    device = Device.TPU
  else:
    device = Device.CPU
  strategy, device = _get_strategy(device)
  
  BATCH_SIZE_PER_REPLICA = hp.get("BATCH_SIZE")
  NUM_EPOCHS = hp.get("NUM_EPOCHS")
  
  n_replicas = strategy.num_replicas_in_sync
  GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * n_replicas
 
  # virtual epochs:
  TRAIN_STEPS_PER_EPOCH = math.ceil(hp.get("num_train") / GLOBAL_BATCH_SIZE)
  EVAL_STEPS_PER_EPOCH = math.ceil(hp.get("num_eval") / GLOBAL_BATCH_SIZE)
  
  #tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
  train_dataset = input_fn(
    fn_args.train_files,
    fn_args.data_accessor,
    transform_graph,
    GLOBAL_BATCH_SIZE)
  
  eval_dataset = input_fn(
    fn_args.eval_files,
    fn_args.data_accessor,
    transform_graph,
    GLOBAL_BATCH_SIZE)
  
  # the objective must be must be a name that appears in the logs
  # returned by the model.fit() method during training.
  #val_logs has keys 'val_loss' and 'val_compile_metrics'
  '''
  tuner = keras_tuner.RandomSearch(
    _make_2tower_keras_model,
    max_trials=hp.get('MAX_TUNE_TRIALS'),
    executions_per_trial=hp.get('EXECUTIONS_PER_TRIAL'),
    overwrite=True,
    hyperparameters=hp,
    allow_new_entries=False,
    objective=keras_tuner.Objective(f'val_hit_rate', 'max'),
    directory=fn_args.working_dir,
    project_name='movie_lens_2t_tuning_r')
  '''
  
  '''
  tuner = keras_tuner.Hyperband(
    _make_2tower_keras_model,
    objective=keras_tuner.Objective(f'val_hit_rate', 'max'),
    max_epochs=8,
    factor=4,
    hyperband_iterations=3,
    overwrite=True,
    hyperparameters=hp,
    allow_new_entries=False,
    directory=fn_args.working_dir,
    project_name='movie_lens_2t_tuning_hb')
  '''
  
  tuner = keras_tuner.BayesianOptimization(
      _make_2tower_keras_model,
      objective=keras_tuner.Objective(f'val_hit_rate', 'max'),
      hyperparameters=hp,
      #alpha=1e-4, # 1e-3 for more exploration. alpha >= (change we want to detect)**2, so for batchsize 1024, alpha ~ (0.004)**2
      alpha=1e-2,
      beta=5.0, #defaut 2.6;  4.0 for more exploration.  sqrt(alpha) = 1./batch_size
      num_initial_points=20, #30
      max_trials=50, #should be 2 to 3 times num_initial_points
      allow_new_entries=False,
      directory=fn_args.working_dir,
      project_name='movie_lens_2t_tuning_bayesian')
  
  return tfx.components.TunerFnResult(
    tuner=tuner,
    fit_kwargs={
      'x': train_dataset,
      'validation_data': eval_dataset,
      'steps_per_epoch': TRAIN_STEPS_PER_EPOCH,
      'validation_steps': EVAL_STEPS_PER_EPOCH
    })
