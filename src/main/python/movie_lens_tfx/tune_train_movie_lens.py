# from

# some code is adapted from https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils_base.py
# and related files
# they have co Copyright 2020 Google LLC. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
from typing import List, Tuple
import enum
import os
import keras_tuner
import tensorflow as tf
import tf_keras as keras
import tensorflow_transform as tft
from tensorboard.plugins.hparams.api import hparams
# from tensorflow.python.ops.gen_experimental_dataset_ops import save_dataset
from tfx import v1 as tfx

from tfx_bsl.public import tfxio

from absl import logging

logging.set_verbosity(logging.DEBUG)
logging.set_stderrthreshold(logging.DEBUG)

TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
LOSS_FN = keras.losses.MeanSquaredError()
METRIC_FN = keras.metrics.RootMeanSquaredError()

FEATURE_KEYS = [
  'age', 'occupation', 'genres', 'hr', 'weekday', 'hr_wk', 'month'
]
_LABEL_KEY = 'rating'

class Device(enum.Enum):
  CPU = "CPU"
  GPU = "GPU"
  TPU = "TPU"

package = "ttdnn"

@keras.saving.register_keras_serializable(package=package)
class UserModel(keras.Model):
  # for init from a load, arguments are present for the compositional instance members too
  def __init__(self, max_user_id: int, embed_out_dim: int = 32,
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
    
    self.user_embedding = keras.Sequential([
      keras.layers.Embedding(self.max_user_id + 1, embed_out_dim),
      keras.layers.Flatten(data_format='channels_last'),
      ], name="user_emb")
    
    # numerical, dist between items matters
    self.age_embedding = None
    if self.feature_acronym.find("a") > -1:
      self.age_embedding = keras.Sequential([
        keras.layers.Flatten(),
        # shape is (batch_size, 1)
      ], name="age_emb")
    
    # numerical
    self.hr_wk_embedding = None
    if self.feature_acronym.find("h") > -1:
      self.hr_wk_embedding = keras.Sequential([
        keras.layers.Flatten(),
        # shape is (batch_size, 1)
      ], name="hr_wk_emb")
      
    if self.feature_acronym.find("m") > -1:
      self.month_embedding = keras.Sequential([
        keras.layers.Flatten(),
        # shape is (batch_size, 1)
      ], name="month_emb")
      
    # categorical
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
    total_length = self.embed_out_dim  # for user_id embedding
    if self.age_embedding:
      _shape = self.age_embedding.compute_output_shape(
        input_shape['age'])
      total_length += _shape[1]
    if self.hr_wk_embedding:
      _shape = self.hr_wk_embedding.compute_output_shape(
        input_shape['hr_wk'])
      total_length += _shape[1]
    if self.month_embedding:
      _shape = self.month_embedding.compute_output_shape(
        input_shape['month'])
      total_length += _shape[1]
    if self.occupation_embedding:
      _shape = self.occupation_embedding.compute_output_shape(
        input_shape['occupation'])
      total_length += _shape[1]
    if self.gender_embedding:
      _shape = self.gender_embedding.compute_output_shape(
        input_shape['gender'])
      total_length += _shape[1]
    return None, total_length
    # return (input_shape['movie_id'][0], total_length)
    # return self.user_embedding.compute_output_shape(input_shape['movie_id'])
  
  def call(self, inputs, **kwargs):
    # Take the input dictionary, pass it through each input layer,
    # and concatenate the result.
    # arrays are: 'user_id',  'gender', 'age_group', 'occupation','movie_id', 'rating'
    # print(f'call {self.name} type={type(inputs)}\n')
    results = []
    results.append(self.user_embedding(inputs['user_id']))
    if self.age_embedding:
      results.append(self.age_embedding(inputs['age']))
    if self.hr_wk_embedding:
      results.append(self.hr_wk_embedding(inputs['hr_wk']))
    if self.month_embedding:
      results.append(self.month_embedding(inputs['month']))
    if self.occupation_embedding:
      results.append(self.occupation_embedding(inputs['occupation']))
    if self.gender_embedding:
      results.append(self.gender_embedding(inputs['gender']))
    return keras.layers.Concatenate()(results)
  
  def get_config(self):
    config = super(UserModel, self).get_config()
    config.update({"max_user_id": self.max_user_id,
                   "embed_out_dim": self.embed_out_dim,
                   "feature_acronym": self.feature_acronym,
                   })
    return config

@keras.saving.register_keras_serializable(package=package)
class MovieModel(keras.Model):
  """
  NOTE: the movie_ids are expected to be already unique and represented by range [1, n_movies] and dtype np.int32.
    No integerlookup to rewrite to smaller number of ids is used here because ratings.dat uses 96% of the
    movies.dat ids.
  """
  
  # for init from a load, arguments are present for the compositional instance members too
  def __init__(self, n_movies: int, n_genres: int,
               embed_out_dim: int = 32, incl_genres: bool = True,
               **kwargs):
    super(MovieModel, self).__init__(**kwargs)
    
    self.embed_out_dim = embed_out_dim
    self.n_movies = n_movies
    self.n_genres = n_genres
    self.incl_genres = incl_genres
    # out_dim = int(np.sqrt(in_dim)) ~ 64
    
    self.movie_embedding = keras.Sequential([
      keras.layers.Embedding(self.n_movies + 1, self.embed_out_dim),
      keras.layers.Flatten(data_format='channels_last'),
      ], name="movie_emb")
    
    # def custom_standardize(self, data):
    #    return tf.strings.regex_replace(data, r",", " ")
    
    # r = tf.strings.split(inputs['genres'], ",")
    # ragint = tf.strings.to_number(r, out_type=tf.int32)
    if self.incl_genres:
      self.genres_embedding = keras.Sequential([
        # input is a string without a constrained length
        # output of lambda layer is a ragged string tensor
        # output of categoricalencoding is an array of length n_genres
        keras.layers.Lambda(lambda x: tf.strings.to_number(
          tf.strings.split(x, ","), out_type=tf.int32)),
        keras.layers.CategoryEncoding(num_tokens=self.n_genres,
                                      output_mode="multi_hot",
                                      sparse=False),
        # keras.layers.TextVectorization(output_mode='multi_hot',
        #    vocabulary=[str(i) for i in range(1,len(genres)+1)],
        #    standardize=custom_standardize)
        ##followed by tf.cast(the_output, tf.float32) needed for textvec output
      ], name="genres_emb")
  
  def build(self, input_shape):
    # print(f'build {self.name} Movie input_shape={input_shape}\n')
    self.movie_embedding.build(input_shape['movie_id'])
    if self.incl_genres:
      self.genres_embedding.build(input_shape['genres'])
    self.built = True
  
  def compute_output_shape(self, input_shape):
    # print(f'compute_output_shape {self.name} input_shape={input_shape}\n')
    # This is invoked after build by CandidateModel
    total_length = self.embed_out_dim
    if self.incl_genres:
      total_length += self.n_genres
    return None, total_length
    
  def call(self, inputs, **kwargs):
    # Take the input dictionary, pass it through each input layer,
    # and concatenate the result.
    # print(f'call {self.name} type={type(inputs)}, kwargs={kwargs}\n')
    # print(f'    spec={inputs.element_spec}\n')
    results = [self.movie_embedding(
      inputs['movie_id'])]  # shape is (batch_size, x, out_dim)
    if self.incl_genres:
      results.append(self.genres_embedding(inputs['genres']))
    return keras.layers.Concatenate()(results)
  
  def get_config(self):
    # updating super config stomps over existing key names, so if need separate values one would need
    # to use some form of package and class name in keys or uniquely name the keys to avoid collision
    config = super(MovieModel, self).get_config()
    config.update({"n_movies": self.n_movies, "n_genres": self.n_genres,
                   "embed_out_dim": self.embed_out_dim,
                   'incl_genres': self.incl_genres
                   })
    return config


# TODO: add hyper-parameter "temperature" after L2Norm
@keras.saving.register_keras_serializable(package=package)
class QueryModel(keras.Model):
  """Model for encoding user queries."""
  
  # for init from a load, arguments are present for the compositional instance members too
  def __init__(self, n_users: int, layer_sizes: list,
               embed_out_dim: int = 32,
               reg: keras.regularizers.Regularizer = None,
               drop_rate: float = 0., feature_acronym: str = "",
               **kwargs):
    """Model for encoding user queries.

            Args:
      layer_sizes:
        A list of integers where the i-th entry represents the number of units
        the i-th layer contains.
    """
    super(QueryModel, self).__init__(**kwargs)
    
    self.embedding_model = UserModel(max_user_id=n_users,
                                     embed_out_dim=embed_out_dim,
                                     feature_acronym=feature_acronym)
    
    self.dense_layers = keras.Sequential(name="dense_query")
    # Use the ReLU activation for all but the last layer.
    for layer_size in layer_sizes[:-1]:
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
    
    self.reg = reg
    
    self.n_users = n_users
    self.feature_acronym = feature_acronym
    self.embed_out_dim = embed_out_dim
    self.layer_sizes = layer_sizes
    self.drop_rate = drop_rate
  
  def build(self, input_shape):
    # print(f'build {self.name} input_shape={input_shape}\n')
    self.embedding_model.build(input_shape)
    input_shape_2 = self.embedding_model.compute_output_shape(
      input_shape)
    # print(f'{self.name} input_shape_2 = {input_shape_2}\n')
    self.dense_layers.build(input_shape_2)
    # print(f'{self.name} output shape={self.dense_layers.compute_output_shape(input_shape_2)}\n')
    # print(f'build {self.name} Query {self.embed_out_dim}, {self.layer_sizes}, => {self.dense_layers.compute_output_shape(input_shape_2)}\n')
    self.built = True
  
  def compute_output_shape(self, input_shape):
    # print(f'compute_output_shape {self.name} input_shape={input_shape}, {input_shape['user_id'][0]}, {self.layer_sizes[-1:]}\n')
    # This is invoked after build by TwoTower
    # return self.output_shapes[0]
    return None, self.layer_sizes[-1]
    # return (input_shape['user_id'][0], self.layer_sizes[-1])
  
  def call(self, inputs, **kwargs):
    # inputs should contain columns: 
    #  '''['user_id', 'gender', 'age_group', 'occupation','movie_id', 'rating_all']'''
    # print(f'call {self.name} type={type(inputs)}\n')
    feature_embedding = self.embedding_model(inputs, **kwargs)
    res = self.dense_layers(feature_embedding)
    return res
  
  def get_config(self):
    config = super(QueryModel, self).get_config()
    config.update({"n_users": self.n_users,
                   "embed_out_dim": self.embed_out_dim,
                   "drop_rate": self.drop_rate,
                   "layer_sizes": self.layer_sizes,
                   "feature_acronym": self.feature_acronym,
                   "reg": keras.saving.serialize_keras_object(self.reg),
                   })
    return config
  
  @classmethod
  def from_config(cls, config):
    for key in ["reg"]:
      config[key] = keras.saving.deserialize_keras_object(config[key])
    return cls(**config)


# TODO: add hyper-parameter "temperature" after L2Norm
@keras.saving.register_keras_serializable(package=package)
class CandidateModel(keras.Model):
  """Model for encoding candidate features."""
  
  # for init from a load, arguments are present for the compositional instance members too
  def __init__(self, n_movies: int, n_genres: int, layer_sizes,
               embed_out_dim: int = 32,
               reg: keras.regularizers.Regularizer = None,
               drop_rate: float = 0., incl_genres: bool = True,
               **kwargs):
    """Model for encoding candidate features.

    Args:
      layer_sizes:
        A list of integers where the i-th entry represents the number of units
        the i-th layer contains.
    """
    super(CandidateModel, self).__init__(**kwargs)
    
    self.embedding_model = MovieModel(n_movies=n_movies,
                                      n_genres=n_genres,
                                      embed_out_dim=embed_out_dim,
                                      incl_genres=incl_genres)
    
    self.dense_layers = keras.Sequential(name="dense_candidate")
    # Use the ReLU activation for all but the last layer.
    for layer_size in layer_sizes[:-1]:
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
    
    self.reg = reg
    
    self.n_movies = n_movies
    self.n_genres = n_genres
    self.incl_genres = incl_genres
    self.embed_out_dim = embed_out_dim
    self.drop_rate = drop_rate
    self.layer_sizes = layer_sizes
  
  def build(self, input_shape):
    # print(f'build {self.name} input_shape={input_shape}\n')
    self.embedding_model.build(input_shape)
    input_shape_2 = self.embedding_model.compute_output_shape(
      input_shape)
    self.dense_layers.build(input_shape_2)
    self.built = True
  
  def compute_output_shape(self, input_shape):
    # print(f'compute_output_shape {self.name} input_shape={input_shape}\n')
    # This is invoked after build by TwoTower
    # return (input_shape['movie_id'][0], self.layer_sizes[-1])
    return None, self.layer_sizes[-1]
  
  def call(self, inputs, **kwargs):
    # inputs should contain columns "movie_id", "genres"
    # print(f'call {self.name} type ={type(inputs)}\ntype ={inputs}\n')
    feature_embedding = self.embedding_model(inputs, **kwargs)
    res = self.dense_layers(feature_embedding)
    # returns an np.ndarray wrapped in a tensor if inputs is tensor, else not wrapped
    return res
  
  def get_config(self):
    config = super(CandidateModel, self).get_config()
    config.update({"n_movies": self.n_movies, "n_genres": self.n_genres,
                   "embed_out_dim": self.embed_out_dim,
                   "drop_rate": self.drop_rate,
                   "layer_sizes": self.layer_sizes,
                   "reg": keras.saving.serialize_keras_object(self.reg),
                   "incl_genres": self.incl_genres
                   })
    return config
  
  @classmethod
  def from_config(cls, config):
    for key in ["reg"]:
      config[key] = keras.saving.deserialize_keras_object(config[key])
    return cls(**config)


@keras.saving.register_keras_serializable(package=package)
class TwoTowerDNN(keras.Model):
  """
  a Two-Tower DNN model that accepts input containing: user, context, and item information along with 
  a label for training.
  
  a sigmoid is used to provide logistic regression predictions of the rating.

  when use_bias_corr is true, the Yi et al. paper is followed to calculate the item sampling probability
  within a mini-batch which is then used to correct probabities and the batch loss sum.

  the number of layers is controlled by a list of their sizes in layer_sizes.
  """
  
  # for init from a load, arguments are present for the compositional instance members too
  def __init__(self, n_users: int, n_movies: int, n_genres: int,
               layer_sizes: list, embed_out_dim: int,
               reg: keras.regularizers.Regularizer = None,
               drop_rate: float = 0, feature_acronym: str = "",
               use_bias_corr: bool = False,
               incl_genres: bool = True, **kwargs):
    super(TwoTowerDNN, self).__init__(**kwargs)
    
    self.query_model = QueryModel(n_users=n_users,
                                  layer_sizes=layer_sizes,
                                  embed_out_dim=embed_out_dim, reg=reg,
                                  drop_rate=drop_rate,
                                  feature_acronym=feature_acronym,
                                  **kwargs)
    
    self.candidate_model = CandidateModel(n_movies=n_movies,
                                          n_genres=n_genres,
                                          layer_sizes=layer_sizes,
                                          embed_out_dim=embed_out_dim,
                                          reg=reg, drop_rate=drop_rate,
                                          incl_genres=incl_genres,
                                          **kwargs)
    
    # elementwise multiplication:
    self.dot_layer = keras.layers.Dot(axes=1)
    self.sigmoid_layer = keras.layers.Activation(
      keras.activations.sigmoid)
    
    self.reg = reg
    
    self.n_users = n_users
    self.n_movies = n_movies
    self.n_genres = n_genres
    self.incl_genres = incl_genres
    self.layer_sizes = layer_sizes
    self.use_bias_corr = use_bias_corr
    self.feature_acronym = feature_acronym
    self.embed_out_dim = embed_out_dim
    self.drop_rate = drop_rate
    
    if self.use_bias_corr:
      self.item_prob_layer = keras.layers.Lambda(
        lambda x: tf.keras.ops.log(tf.keras.ops.clip(1. / x, 1e-6, 1.0)))
      self.softmax_layer = keras.layers.Softmax()
      self.log_layer = keras.layers.Lambda(lambda x: tf.keras.ops.log(x))
      self.mult_layer = keras.layers.Multiply()
      # self.final_loss_bc_layer = keras.losses.Loss(name=None, reduction="mean", dtype=None)
      self.final_loss_bc_layer = keras.layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=0))
  
  def call(self, inputs, **kwargs):
    """['user_id', 'gender', 'age_group', 'occupation','movie_id', 'rating']"""
    user_vector = self.query_model(inputs, **kwargs)
    movie_vector = self.candidate_model(inputs, **kwargs)
    s = self.dot_layer([user_vector, movie_vector])
    s = self.sigmoid_layer(s)
    return s
  
  def build(self, input_shape):
    # print(f'build {self.name} TWOTOWER input_shape={input_shape}\n')
    self.query_model.build(input_shape)
    self.candidate_model.build(input_shape)
    self.built = True
  
  def compute_output_shape(self, input_shape):
    # (batch_size,)  a scalar for each row in batch
    # return input_shape['user_id']
    return (None,)
  
  @keras.saving.register_keras_serializable(package=package,
                                            name="calc_item_probability_inverse")
  # in non-eager mode, keras attempts to draw a graph if annotated w/ tf.function
  @tf.function(autograph=True, reduce_retracing=True)
  def calc_item_probability_inverse(self, x):
    """
    given the batch x['movie_id'] tensor, this method returns an item probability vector.

    Args:
        x: the batch tensor of array for 'movie_id'
    Returns:
        tensor array of 'B' for the given batch movie_ids following Yi et al. paper.
    """
    # tf.keras.backend.eval is deprecated, but can be replaced with
    # @tf.function
    # def evaluate_tensor(tensor):
    #     return tensor
    # result = evaluate_tensor(tensor)
    alpha = tf.keras.backend.eval(self.optimizer.learning_rate)
    if len(tf.shape(x).numpy()) == 0:
      _batch_size = 1
    else:
      _batch_size = tf.shape(x)[0]
    A = tf.lookup.experimental.MutableHashTable(key_dtype=tf.int32,
                                                value_dtype=tf.float32,
                                                default_value=0.)
    B = tf.lookup.experimental.MutableHashTable(key_dtype=tf.int32,
                                                value_dtype=tf.float32,
                                                default_value=0.)
    for i, movie_id in enumerate(x):
      t = i + 1
      b = B.lookup(movie_id, dynamic_default_values=0.)
      a = A.lookup(movie_id, dynamic_default_values=0.)
      c = (1. - alpha) * b + (alpha * (t - a))
      B.insert(keys=movie_id, values=c)
      A.insert(keys=movie_id, values=t)
      if i == _batch_size - 1: break
    BB = B.export()[1]
    return BB
  
  def train_step(self, batch, **kwargs):
    x, y = batch
    with tf.GradientTape() as tape:
      if not self.use_bias_corr:
        y_pred = self.call(x, training=True)  # Forward pass
        loss = self.compute_loss(y=y, y_pred=y_pred)
      else:
        # following Yi et al, and a small portion of what's handled in tfrecommenders Retrieval layer
        # https://www.tensorflow.org/recommenders/examples/basic_retrieval
        
        # TODO: assert self.optimizer is SGD so that the loss is compat w/ gradient
        
        BB = self.calc_item_probability_inverse(x['movie_id'])
        log_p = self.item_prob_layer(BB)
        
        user_vector = self.query_model(x, **kwargs)
        movie_vector = self.candidate_model(x, **kwargs)
        score = self.dot_layer([user_vector, movie_vector])
        score_c = score - log_p
        p_batch = self.softmax_layer(score_c)
        y_pred = p_batch
        
        log_p_batch = self.log_layer(p_batch)
        loss_batch = self.mult_layer([y, -log_p_batch])
        loss = self.final_loss_bc_layer(loss_batch)
    
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    
    # Update metrics
    for metric in self.metrics:
      if metric.name == "loss":
        metric.update_state(loss)
      else:
        metric.update_state(y, y_pred)
    
    # Return a dict mapping metric names to current value.
    return {m.name: m.result() for m in self.metrics}
  
  def test_step(self, data):
    x, y = data
    y_pred = self(x, training=False)
    loss = self.compute_loss(y=y, y_pred=y_pred)
    for metric in self.metrics:
      if metric.name == "loss":
        metric.update_state(loss)
      else:
        metric.update_state(y, y_pred)
    # Return a dict mapping metric names to current value.
    # Note that it will include the loss (tracked in self.metrics).
    return {m.name: m.result() for m in self.metrics}
  
  def get_config(self):
    config = super(TwoTowerDNN, self).get_config()
    config.update({"n_users": self.n_users, "n_movies": self.n_movies,
                   "n_genres": self.n_genres,
                   "embed_out_dim": self.embed_out_dim,
                   "drop_rate": self.drop_rate,
                   "layer_sizes": self.layer_sizes,
                   "use_bias_corr": self.use_bias_corr,
                   "feature_acronym": self.feature_acronym,
                   "reg": keras.saving.serialize_keras_object(self.reg),
                   "incl_genres": self.incl_genres
                   })
    return config
  
  @classmethod
  def from_config(cls, config):
    for key in ["reg"]:
      config[key] = keras.saving.deserialize_keras_object(config[key])
    return cls(**config)


# https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/examples/penguin/penguin_utils_base.py#L98
def _input_fn(file_pattern: List[str],
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
    tfxio.TensorFlowDatasetOptions(
      batch_size=batch_size, label_key=_LABEL_KEY),
    tf_transform_output.transformed_metadata.schema).repeat().prefetch(
    tf.data.AUTOTUNE)


def _make_2tower_keras_model(
  hp: keras_tuner.HyperParameters) -> tf.keras.Model:
  # TODO: consider change to read from the transformed schema
  input_shapes = {}
  for element in FEATURE_KEYS:
    input_shapes[element] = (None,)
    # input_shapes[element] = (batch_size,)
  
  model = TwoTowerDNN(
    n_users=hp.get("user_id_max") + 1,
    n_movies=hp.get("movie_id_max") + 1,
    n_genres=hp.get("n_genres"),
    layer_sizes=hp.get('layer_sizes'),
    embed_out_dim=hp.get('embed_out_dim'),
    reg=None, drop_rate=0.1,
    feature_acronym=hp.get("feature_acronym"),
    use_bias_corr=hp.get('use_bias_corr'))
  
  model.build(input_shapes)
  
  optimizer = keras.optimizers.Adam(
    learning_rate=hp.get('learning_rate'))
  
  # LOSS:
  # can use Ordinal Logistic Regression for classification into ratings categories
  # or MSE: when loss=MSE, choose RMSE as eval metric, but they are affected by outliers
  #  so consider MAE
  #  so consider MAE
  model.compile(
    loss=LOSS_FN,
    optimizer=optimizer,
    metrics=[METRIC_FN, keras.metrics.MeanAbsoluteError()],
    run_eagerly=hp.get("run_eagerly")
  )
  # TO debug, user run_eagerly=False
  return model


def get_default_hyperparameters(
  custom_config: tfx.components.FnArgs.custom_config) -> keras_tuner.HyperParameters:
  """Returns hyperparameters for building Keras model."""
  hp = keras_tuner.HyperParameters()
  # Defines search space.
  hp.Choice('lr', [1e-4], default=1e-4)
  hp.Choice("regl2", values=[None, 0.001, 0.01], default=None)
  hp.Float("drop_rate", min_value=0.1, max_value=0.5, default=0.5)
  hp.Choice("embed_out_dim", values=[32], default=32)
  hp.Choice("layer_sizes", values=[[32]], default=[32])
  # ahmos for "age", "hr_wk", "month", "occupation", "gender"
  hp.Fixed("feature_acronym", value="h")
  hp.Boolean("incl_genres", default=True)
  hp.Fixed('num_epochs', value=10)
  hp.Fixed('batch_size', TRAIN_BATCH_SIZE)
  hp.Boolean("use_bias_corr", default=False)
  hp.Fixed('user_id_max', value=custom_config["user_id_max"])
  hp.Fixed('movie_id_max', custom_config["movie_id_max"])
  hp.Fixed('n_genres', custom_config["n_genres"])
  hp.Fixed('run_eagerly', False)
  return hp


# TFX Trainer will call this function.
def _get_strategy(device: Device) -> Tuple[
  tf.distribute.Strategy, Device]:
  strategy = None
  if device == Device.GPU:
    try:
      device_physical = tf.config.list_physical_devices('GPU')[0]
      strategy = tf.distribute.MirroredStrategy(
        devices=[device_physical])  # or [device_physical.name]
      tf.config.optimizer.set_jit(False)
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
    device = Device.TPU
  return strategy, device


# https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/examples/penguin/penguin_utils_base.py#L43
def _make_serving_signatures(model,
                             tf_transform_output: tft.TFTransformOutput):
  """Returns the serving signatures.

  Args:
    model: the model function to apply to the transformed features.
    tf_transform_output: The transformation to apply to the serialized
      tf.Example.

  Returns:
    The signatures to use for saving the mode. The 'serving_default' signature
    will be a concrete function that takes a batch of unspecified length of
    serialized tf.Example, parses them, transformes the features and
    then applies the model. The 'transform_features' signature will parses the
    example and transforms the features.
  """
  
  # We need to track the layers in the model in order to save it.
  # TODO(b/162357359): Revise once the bug is resolved.
  model.tft_layer = tf_transform_output.transform_features_layer()
  
  @tf.function(input_signature=[
    tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
  ])
  def serve_tf_examples_fn(serialized_tf_example):
    """Returns the output to be used in the serving signature."""
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    # Remove label feature since these will not be present at serving time.
    raw_feature_spec.pop(_LABEL_KEY)
    raw_features = tf.io.parse_example(serialized_tf_example,
                                       raw_feature_spec)
    transformed_features = model.tft_layer(raw_features)
    logging.info('serve_transformed_features = %s',
                 transformed_features)
    
    outputs = model(transformed_features)
    # TODO(b/154085620): Convert the predicted labels from the model using a
    # reverse-lookup (opposite of transform.py).
    return {'outputs': outputs}


# https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/examples/chicago_taxi_pipeline/taxi_utils.py#L128
def _get_transform_features_signature(model, tf_transform_output):
  """Returns a serving signature that accepts `tensorflow.Example`."""
  model.tft_layer_eval = tf_transform_output.transform_features_layer()
  
  @tf.function(
    input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ]
  )
  def transform_features_fn(serialized_tf_example):
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    raw_features = tf.io.parse_example(serialized_tf_example,
                                       raw_feature_spec)
    transformed_features = model.tft_layer_eval(raw_features)
    logging.info('eval_transformed_features = %s', transformed_features)
    return transformed_features
  
  return transform_features_fn


# https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/docs/tutorials/tfx/recommenders.ipynb#L851
# https://github.com/tensorflow/tfx/blob/master/docs/tutorials/tfx/recommenders.ipynb
def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example and applies TFT."""
  try:
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
      """Returns the output to be used in the serving signature."""
      try:
        feature_spec = tf_transform_output.raw_feature_spec()
        parsed_features = tf.io.parse_example(serialized_tf_examples,
                                              feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        result = model(transformed_features)
      except BaseException as err:
        raise BaseException(
          '######## ERROR IN serve_tf_examples_fn:\n{}\n###############'.format(
            err))
      return result
  except BaseException as err2:
    raise BaseException(
      '######## ERROR IN _get_serve_tf_examples_fn:\n{}\n###############'.format(
        err2))
  
  return serve_tf_examples_fn


def run_fn(fn_args: tfx.components.FnArgs):
  """Train the model based on given args.

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
          'user_id_max'
          'movie_id_max'
          'n_genres'
          'run_eagerly'

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
      'user_id_max'
      'movie_id_max'
      'n_genres'
      'run_eagerly'

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
  
  # not sure if outputs query_model and candidate_model are passed to this.
  for attr_name in dir(fn_args):
    # Filter out built-in methods and private attributes
    if not attr_name.startswith('__') and not callable(
      getattr(fn_args, attr_name)):
      attr_value = getattr(fn_args, attr_name)
      logging.debug(f"{attr_name}: {attr_value}")
  
  if not fn_args.hyperparameters:
    raise ValueError('hyperparameters must be provided')
  
  logging.debug(
    f'fn_args.serving_model_dir={fn_args.serving_model_dir}')
  logging.debug(f'fn_args.train_files={fn_args.train_files}')
  logging.debug(f'fn_args.eval_files={fn_args.eval_files}')
  logging.debug(f'fn_args.train_steps={fn_args.train_steps}')
  logging.debug(f'fn_args.eval_steps={fn_args.eval_steps}')
  logging.debug(f"data_accessor: {fn_args.data_accessor}")
  logging.debug(f"Hyperparameters: {fn_args.hyperparameters}")
  
  query_model_serving_dir = os.path.join(fn_args.serving_model_dir,
                                         'query_model')
  candidate_model_serving_dir = os.path.join(fn_args.serving_model_dir,
                                             'candidate_model')
  
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
  
  train_dataset = _input_fn(
    fn_args.train_files,
    fn_args.data_accessor,
    tf_transform_output,
    TRAIN_BATCH_SIZE)
  
  eval_dataset = _input_fn(
    fn_args.eval_files,
    fn_args.data_accessor,
    tf_transform_output,
    EVAL_BATCH_SIZE)
  
  hp = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
  
  logging.info('HyperParameters for training: %s' % hp.get_config())
  
  if "device" in fn_args.custom_config:
    d = fn_args.custom_config["device"]
    if d == "GPU":
      device = Device.GPU
    elif d == "TPU":
      device = Device.TPU
    else:
      device = Device.CPU
  else:
    device = Device.CPU
  strategy, device = _get_strategy(device)
  
  with strategy.scope():
    model = _make_2tower_keras_model(hp)
    # model = _make_2tower_keras_model(hp, tf_transform_output)
  
  # Write logs to path
  # tensorboard_callback = tf.keras.callbacks.TensorBoard(
  #    log_dir=fn_args.model_run_dir, update_freq='epoch')
  
  stop_early = tf.keras.callbacks.EarlyStopping(
    monitor=f'val_{LOSS_FN.name}', patience=3)
  
  model.fit(
    train_dataset,
    steps_per_epoch=fn_args.train_steps,
    validation_data=eval_dataset,
    validation_steps=fn_args.eval_steps,
    callbacks=[stop_early])
  
  # save all 3 models
  # return the two twoer model
  model.save(fn_args.serving_model_dir, save_format='tf')
  # should handle saving decorated methods too, that is,
  # those decorated with @keras.saving.register_keras_serializable(package=package)
  
  model.query_model.save(query_model_serving_dir, save_format='tf')
  model.candidate_model.save(candidate_model_serving_dir,
                             save_format='tf')
  
  # should be able to use model.save without creating a signature profile
  #  but if have troubles with this, use one of these 2:
  # signatures = _make_serving_signatures(model, tf_transform_output)
  # signatures = {
  #  'serving_default':
  #    _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
  #      tf.TensorSpec( shape=[None], dtype=tf.string, name='examples')),
  #  'transform_features':
  #    _get_transform_features_signature(model, tf_transform_output),
  # }
  # tf.saved_model.save(model, fn_args.serving_model_dir, signatures=signatures)
  
  return model


# TFX Tuner will call this function.
def tuner_fn(
  fn_args: tfx.components.FnArgs) -> tfx.components.TunerFnResult:
  """Build the tuner using the KerasTuner API.

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
          'user_id_max'
          'movie_id_max'
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
  
  logging.debug(f"Working directory: {fn_args.working_dir}")
  logging.debug(f"Training files: {fn_args.train_files}")
  logging.debug(f"Evaluation files: {fn_args.eval_files}")
  logging.debug(f"Transform graph path: {fn_args.transform_graph_path}")
  logging.debug(f"data_accessor: {fn_args.data_accessor}")
  logging.debug(f"Hyperparameters: {fn_args.hyperparameters}")
  logging.debug(f"Custom config: {fn_args.custom_config}")
  
  if fn_args.hyperparameters:
    hp = keras_tuner.HyperParameters.from_config(
      fn_args.hyperparameters)
  else:
    hp = get_default_hyperparameters(fn_args.custom_config)
  
  # the objective must be must be a name that appears in the logs
  # returned by the model.fit() method during training.
  tuner = keras_tuner.RandomSearch(
    _make_2tower_keras_model,
    max_trials=6,
    hyperparameters=hp,
    allow_new_entries=False,
    objective=keras_tuner.Objective(f'val_{LOSS_FN.name}', 'min'),
    # objective=keras_tuner.Objective('val_loss', 'min'),
    directory=fn_args.working_dir,
    project_name='movie_lens_2t_tuning')
  
  transform_graph = tft.TFTransformOutput(fn_args.transform_graph_path)
  
  train_dataset = _input_fn(
    fn_args.train_files,
    fn_args.data_accessor,
    transform_graph,
    TRAIN_BATCH_SIZE)
  
  eval_dataset = _input_fn(
    fn_args.eval_files,
    fn_args.data_accessor,
    transform_graph,
    EVAL_BATCH_SIZE)
  
  return tfx.components.TunerFnResult(
    tuner=tuner,
    fit_kwargs={
      'x': train_dataset,
      'validation_data': eval_dataset,
      'steps_per_epoch': fn_args.train_steps,
      'validation_steps': fn_args.eval_steps
    })
