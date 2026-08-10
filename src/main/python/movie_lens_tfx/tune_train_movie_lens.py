# from
import base64
import pickle
import numpy as np
import abc
import time
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
from tensorflow_transform import common_types
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
logging.set_verbosity(logging.INFO)
logging.set_stderrthreshold(logging.INFO)

'''
builds pipelines for training a TwoTowerDNN model to train Query and Candidate
embedding models.  The training is optimized using Contrastive Learning for a
Listwise Discriminative Model.

The run_fn defines the model, compile, fit and signatures.
The tuner_fn specifies that the custom metric "val_composite_ndcg_20" should be used
to decide which model is best.
'''

DEFAULT_BATCH_SIZE = 1024
DEFAULT_NUM_EPOCHS = 20
DEFAULT_NUM_EXAMPLES = 100000

#NOTE: could be improved by writing the headers to a file in the Transform stage and reading them here:
LABEL_KEY = 'rating'
N_GENRES = 18
N_AGE_GROUPS = 7

package = "ttdnn"

# https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/examples/penguin/penguin_utils_base.py#L98
def input_fn(file_pattern: List[str], data_accessor: tfx.components.DataAccessor,
    tf_transform_output: tft.TFTransformOutput, batch_size:int, is_train:bool,
) -> tf.data.Dataset:
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
  if is_train:
      return (data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size,
        shuffle=True, shuffle_buffer_size=2000, label_key=LABEL_KEY),
        tf_transform_output.transformed_metadata.schema)
        .repeat()
        .prefetch(tf.data.AUTOTUNE))
  return (data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size,
        shuffle=False, label_key=LABEL_KEY),
        tf_transform_output.transformed_metadata.schema)
        .prefetch(tf.data.AUTOTUNE))

def _make_query_model(n_users : int, layer_sizes : list,
    regl2 : float, drop_rate : float,
    feature_acronym : str, **kwargs) :
    
    @keras.utils.register_keras_serializable(package=package)
    class CyclicalEncoding(keras.layers.Layer):
        def __init__(self, max_val, name='cyc_enc', **kwargs):
            super().__init__(name=name, **kwargs)
            self.max_val = max_val
        
        def call(self, inputs):
            radians = 2.0 * np.pi * tf.cast(inputs, tf.float32) / tf.cast(self.max_val, tf.float32)
            return tf.concat([tf.sin(radians), tf.cos(radians)], axis=-1)
        
        def get_config(self):
            config = super(CyclicalEncoding, self).get_config()
            config.update({"max_val": self.max_val})
    
    @keras.utils.register_keras_serializable(package=package)
    class QueryBiasConcatenationLayer(keras.layers.Layer):
        """Appends a constant 1.0 dimension to the query vector for MIPS bias folding."""
        
        def __init__(self, name='query_bias_concat', **kwargs):
            super(QueryBiasConcatenationLayer, self).__init__(name=name, **kwargs)
        
        def call(self, inputs):
            res = inputs
            batch_size = tf.shape(res)[0]
            ones = tf.ones(shape=(batch_size, 1), dtype=res.dtype)
            return tf.concat([res, ones], axis=-1)
        
        def compute_output_shape(self, input_shape):
            # input_shape is (batch_size, dense_dim)
            return (input_shape[0], input_shape[1] + 1)
        
    @keras.utils.register_keras_serializable(package=package)
    class UserModel(keras.Model):
        # for init from a load, arguments are present for the compositional instance members too
        def __init__(self, max_user_id: int,
                feature_acronym: str = "",
                name='user_model', **kwargs):
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
            super(UserModel, self).__init__(name=name, **kwargs)
            
            # NOTE: it is up to the using component to filter for OOV values
            #      to avoid using this incorrectly
            # output dimension calculated following advice in
            # https://developers.googleblog.com/introducing-tensorflow-feature-columns/
            user_embed_out_dim = round(max_user_id ** 0.25)  # 9
            user_emb = keras.Sequential([
                keras.layers.Embedding(max_user_id + 1, user_embed_out_dim),
                keras.layers.Flatten(data_format='channels_last'),
            ], name="user_emb")
            
            # ordinal, dist between items matters
            age_emb = None
            if feature_acronym.find("a") > -1:
                age_embed_out_dim = 7
                age_emb = keras.Sequential([
                    keras.layers.Dense(age_embed_out_dim, activation='swish',
                        kernel_initializer='he_normal', use_bias=False, name='age_emb_dense'),
                    keras.layers.Flatten(data_format='channels_last'),
                ], name="age_emb")
            
            # ordinal, dist between items matters.
            yr_z_emb = None
            if feature_acronym.find("y") > -1:
                yr_embed_out_dim = round(50 ** 0.25)  # 3
                yr_z_emb = keras.Sequential([
                    keras.layers.Dense(yr_embed_out_dim, activation='swish',
                        kernel_initializer='he_normal', use_bias=False,
                        kernel_regularizer=keras.regularizers.l2(1e-3), name='yr_z_emb_dense'),
                    keras.layers.Flatten(data_format='channels_last'),
                ], name="yr_z_emb")
            
            # numerical, cyclical
            hr_wk_emb = None
            if feature_acronym.find("h") > -1:
                hr_wk_emb = keras.Sequential([
                    CyclicalEncoding(max_val=24 * 7),
                    keras.layers.Flatten(data_format='channels_last'),
                ], name="hr_wk_emb")
            
            # numerical, cyclical
            month_emb = None
            if feature_acronym.find("m") > -1:
                month_emb = keras.Sequential([
                    CyclicalEncoding(max_val=12),
                    keras.layers.Flatten(data_format='channels_last'),
                ], name="month_emb")
            
            # categorical, nominal, order doesn't matter
            occupation_emb = None
            if feature_acronym.find("o") > -1:
                occupation_emb = keras.Sequential([
                    keras.layers.CategoryEncoding(num_tokens=21, output_mode="one_hot", name="one_hot_occ"),
                    keras.layers.Flatten(data_format='channels_last'),
                ], name="occupation_emb")
            
            # categorical
            gender_emb = None
            if feature_acronym.find("s") > -1:
                gender_emb = keras.Sequential([
                    keras.layers.CategoryEncoding(num_tokens=2, output_mode="one_hot", name="gender_on_hot"),
                    keras.layers.Flatten(data_format='channels_last'),
                ], name="gender_emb")
            
            self.feature_acronym = feature_acronym
            self.max_user_id = max_user_id
            self.user_emb = user_emb
            self.age_emb = age_emb
            self.yr_z_emb = yr_z_emb
            self.hr_wk_emb = hr_wk_emb
            self.month_emb = month_emb
            self.occupation_emb = occupation_emb
            self.gender_emb = gender_emb
        
        def build(self, input_shape):
            # print(f'build {self.name} input_shape={input_shape}\n')
            self.user_emb.build(input_shape['user_id'])
            if self.age_emb:
                self.age_emb.build(input_shape['age'])
            if self.yr_z_emb:
                self.yr_z_emb.build(input_shape['yr_z'])
            if self.hr_wk_emb:
                self.hr_wk_emb.build(input_shape['hr_wk'])
            if self.month_emb:
                self.month_emb.build(input_shape['month'])
            if self.occupation_emb:
                self.occupation_emb.build(input_shape['occupation'])
            if self.gender_emb:
                self.gender_emb.build(input_shape['gender'])
            self.built = True
        
        def compute_output_shape(self, input_shape):
            # print(f'compute_output_shape {self.name} input_shape={input_shape}\n')
            # This is invoked after build by QueryModel.
            _shape = self.user_emb.compute_output_shape(
                input_shape['user_id'])
            total_length = _shape[-1]
            if self.age_emb:
                _shape = self.age_emb.compute_output_shape(
                    input_shape['age'])
                total_length += _shape[-1]
            if self.yr_z_emb:
                _shape = self.yr_z_emb.compute_output_shape(
                    input_shape['yr_z'])
                total_length += _shape[-1]
            if self.hr_wk_emb:
                _shape = self.hr_wk_emb.compute_output_shape(
                    input_shape['hr_wk'])
                total_length += _shape[-1]
            if self.month_emb:
                _shape = self.month_emb.compute_output_shape(
                    input_shape['month'])
                total_length += _shape[-1]
            if self.occupation_emb:
                _shape = self.occupation_emb.compute_output_shape(
                    input_shape['occupation'])
                total_length += _shape[-1]
            if self.gender_emb:
                _shape = self.gender_emb.compute_output_shape(
                    input_shape['gender'])
                total_length += _shape[-1]
            return None, total_length
            # return (input_shape['movie_id'][0], total_length)
            # return self.user_emb.compute_output_shape(input_shape['movie_id'])
        
        def call(self, inputs, **kwargs):
            # Take the input dictionary, pass it through each input layer,
            # and concatenate the result.
            # arrays are: 'user_id',  'gender', 'age_group', 'occupation','movie_id', 'rating'
            # print(f'call {self.name} type={type(inputs)}\n')
            # tf.print(inputs)
            results = []
            results.append(self.user_emb(inputs['user_id']))
            if self.age_emb:
                results.append(self.age_emb(inputs['age']))
            if self.yr_z_emb:
                results.append(self.yr_z_emb(inputs['yr_z']))
            if self.hr_wk_emb:
                results.append(self.hr_wk_emb(inputs['hr_wk']))
            if self.month_emb:
                results.append(self.month_emb(inputs['month']))
            if self.occupation_emb:
                results.append(self.occupation_emb(inputs['occupation']))
            if self.gender_emb:
                results.append(self.gender_emb(inputs['gender']))
            res = keras.layers.Concatenate()(results)
            # logging.debug(f'call {self.name} SHAPE ={res.shape}')
            # tf.print('CALL', self.name, ' shape=', res.shape)
            return res
        
        def get_config(self):
            config = super(UserModel, self).get_config()
            config.update({"max_user_id": self.max_user_id,
                "feature_acronym": self.feature_acronym,
            })
            return config
    
    @keras.utils.register_keras_serializable(package=package)
    class QueryModel(keras.Model):
        """Model for encoding user queries."""
        
        # for init from a load, arguments are present for the compositional instance members too
        def __init__(self, n_users: int,
                layer_sizes: list,
                regl2: float = 0.0,
                drop_rate: float = 0., feature_acronym: str = "",
                name='query_model', **kwargs):
            """Model for encoding user queries.
    
                    Args:
              layer_sizes:
                A list of integers where the i-th entry represents the number of units
                the i-th layer contains.
            """
            super(QueryModel, self).__init__(name=name, **kwargs)
            
            self.user_model = UserModel(max_user_id=n_users,
                feature_acronym=feature_acronym)
            if isinstance(layer_sizes, str):
                layer_sizes = json.loads(layer_sizes)
            
            self.dense_query = keras.Sequential(name="dense_query")
            reg = None
            # Use the ReLU activation for all but the last layer.
            for layer_size in layer_sizes[:-1]:
                if regl2 > 0.0:
                    reg = keras.regularizers.l2(regl2)
                # TODO: consider changing order to: Dense, LayerNorm, Activation(elu), Dropout
                self.dense_query.add(
                    keras.layers.Dense(layer_size,
                        activation="elu",
                        kernel_regularizer=reg,
                        kernel_initializer="glorot_normal",
                        use_bias=False, name=f'dense_{layer_size}'))
                # self.dense_query.add(keras.layers.BatchNormalization())
                self.dense_query.add(keras.layers.LayerNormalization())
                self.dense_query.add(keras.layers.Dropout(drop_rate))
            
            for layer_size in layer_sizes[-1:]:
                self.dense_query.add(keras.layers.Dense(layer_size,
                    kernel_initializer="glorot_normal", use_bias=False, name=f'_layers'))
                    
            # removing the normalization layer to allow the models to use dot product instead
            # of cosine similarity for more personalized ANN searches that use the magnitudes
            # in addition to the directions
            
            self.query_bias_concat_layer = QueryBiasConcatenationLayer()
            
            self.regl2 = regl2
            self.n_users = n_users
            self.feature_acronym = feature_acronym
            self.layer_sizes = layer_sizes
            self.drop_rate = drop_rate
        
        def build(self, input_shape):
            # print(f'build {self.name} input_shape={input_shape}\n')
            self.user_model.build(input_shape)
            input_shape_2 = self.user_model.compute_output_shape(input_shape)
            self.dense_query.build(input_shape_2)
            dense_out_shape = self.dense_query.compute_output_shape(input_shape_2)
            self.query_bias_concat_layer.build(dense_out_shape)
            self.built = True
        
        def compute_output_shape(self, input_shape):
            # print(f'compute_output_shape {self.name} input_shape={input_shape}, {input_shape['user_id'][0]}, {self.layer_sizes[-1:]}\n')
            # This is invoked after build by TwoTower
            # return self.output_shapes[0]
            input_shape_2 = self.user_model.compute_output_shape(input_shape)
            dense_out_shape = self.dense_query.compute_output_shape(input_shape_2)
            output_shape = self.query_bias_concat_layer.compute_output_shape(dense_out_shape)
            return output_shape
            # return None, self.layer_sizes[-1]
            # return (input_shape['user_id'][0], self.layer_sizes[-1])
        
        def call(self, inputs, **kwargs):
            # inputs should contain columns:
            # print(f'call {self.name} type={type(inputs)}, inputs={inputs}\n')
            feature_embedding = self.user_model(inputs, **kwargs)
            res = self.dense_query(feature_embedding)
            # tf.print('CALL', self.name, ' shape=', res.shape)
            # Append 1.0 to the query vector to multiply against the item bias in CandidateModel calll
            return self.query_bias_concat_layer(res)
        
        def get_config(self):
            config = super(QueryModel, self).get_config()
            config.update({"n_users": self.n_users,
                "drop_rate": self.drop_rate,
                "layer_sizes": self.layer_sizes,
                "feature_acronym": self.feature_acronym,
                "regl2": self.regl2,
            })
            return config
   
    return QueryModel(n_users=n_users,
                                    layer_sizes=layer_sizes,
                                    regl2=regl2,
                                    drop_rate=drop_rate,
                                    feature_acronym=feature_acronym,
                                    **kwargs)
    
def _make_candidate_model(n_movies : int, movies_offset : int,
        n_genres : int, layer_sizes : List,
        regl2 : float, drop_rate : float, incl_genres : bool, **kwargs):
    
    @keras.utils.register_keras_serializable(package=package)
    class CandidateBiasConcatenationLayer(keras.layers.Layer):
        """Encapsulates bias reshaping and concatenation for MIPS folding."""
        def __init__(self, n_movies : int, movies_offset: int, name='candidate_bias_concat', **kwargs):
            super(CandidateBiasConcatenationLayer, self).__init__(name=name, **kwargs)
            self.n_movies = n_movies
            self.movies_offset = movies_offset
            self.item_bias = tf.keras.layers.Embedding(
                input_dim=n_movies,
                output_dim=1,
                embeddings_initializer='zeros'
            )
            
        def call(self, inputs):
            res, movie_ids = inputs
            shifted_ids = movie_ids - self.movies_offset
            safe_ids = tf.clip_by_value(shifted_ids, 0, self.n_movies - 1)
            bias = self.item_bias(safe_ids)
            bias_reshaped = tf.reshape(bias, [-1, 1])
            return tf.concat([res, bias_reshaped], axis=-1)
        
        def build(self, input_shape):
            res_shape, movie_ids_shape = input_shape
            self.item_bias.build(movie_ids_shape)
            self.built = True
            
        def compute_output_shape(self, input_shape):
            res_shape, movie_ids_shape = input_shape
            # Total dimension increases by 1 to account for the folded bias
            return (res_shape[0], res_shape[1] + 1)
        
        def get_config(self):
            # updating super config stomps over existing key names, so if need separate values one would need
            # to use some form of package and class name in keys or uniquely name the keys to avoid collision
            config = super(CandidateBiasConcatenationLayer, self).get_config()
            config.update(
                {"n_movies": self.n_movies, "movies_offset" : self.movies_offset
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
        def __init__(self, n_movies: int, movies_offset: int, n_genres: int,
                incl_genres: bool = True,
                name='movie_model', **kwargs):
            super(MovieModel, self).__init__(name=name, **kwargs)
            
            self.n_movies = n_movies
            self.movies_offset = movies_offset
            self.n_genres = n_genres
            self.incl_genres = incl_genres
            # out_dim = int(np.sqrt(in_dim)) ~ 64
            
            # NOTE: it is up to the using component to filter for OOV values
            #      to avoid using this incorrectly
            movie_embed_out_dim = round(self.n_movies ** 0.25)  # 8
            self.movie_emb = keras.Sequential([
                keras.layers.Embedding(
                    self.n_movies,
                    movie_embed_out_dim,
                    embeddings_initializer="glorot_normal"),
                keras.layers.Flatten(data_format='channels_last'),
            ], name="movie_emb")
            
            genres_emb = None
            if self.incl_genres:
                # expand to embed_out_dim for concatenation
                genres_embed_out_dim = 8  # 18
                genres_emb = keras.Sequential([
                    keras.layers.Dense(genres_embed_out_dim, use_bias=False),
                    keras.layers.Flatten(data_format='channels_last'),
                ], name="genres_emb")
            self.genres_emb = genres_emb
        
        def build(self, input_shape):
            # tf.print("build", self.name, "input_shape=:", input_shape)
            # tf.print(f"OUTPUT shapes:", self.movie_emb.compute_output_shape( input_shape['movie_id']))
            self.movie_emb.build(input_shape['movie_id'])
            if self.incl_genres:
                self.genres_emb.build(input_shape['genres'])
                # tf.print(self.genres_embedding.compute_output_shape(input_shape['genres']))
            self.built = True
        
        def compute_output_shape(self, input_shape):
            # print(f'compute_output_shape {self.name} input_shape={input_shape}\n')
            # This is invoked after build by CandidateModel
            _shape = self.movie_emb.compute_output_shape(
                input_shape['movie_id'])
            total_length = _shape[-1]
            if self.incl_genres:
                _shape = self.genres_emb.compute_output_shape(
                    input_shape['genres'])
                total_length += _shape[-1]
            return None, total_length
        
        def call(self, inputs, **kwargs):
            # Take the input dictionary, pass it through each input layer,
            # and concatenate the result.
            # print(f'call {self.name} type={type(inputs)}, kwargs={kwargs}\n')
            # print(f'    spec={inputs.element_spec}\n')
            
            shifted_ids = inputs['movie_id'] - self.movies_offset
            
            # 2. Safety guard: Clip IDs to valid range [0, n_movies - 1]
            # This prevents crashes if an unseen offset ID is passed in serving
            safe_ids = tf.clip_by_value(shifted_ids, 0, self.n_movies - 1)
            
            x = tf.cast(safe_ids, dtype=tf.int32)
            results = [self.movie_emb(x)]
            # shape is (batch_size, x, out_dim)
            if self.incl_genres:
                results.append(self.genres_emb(inputs['genres']))
            # tf.print('concatenate shapes:', [r.shape for r in results])
            res = keras.layers.Concatenate(axis=-1)(results)
            # tf.print('call result,shape=', res.shape)
            # logging.debug(f'call {self.name} SHAPE ={res.shape}')
            # tf.print('CALL', self.name, ' shape=', res.shape)
            return res
        
        def get_config(self):
            # updating super config stomps over existing key names, so if need separate values one would need
            # to use some form of package and class name in keys or uniquely name the keys to avoid collision
            config = super(MovieModel, self).get_config()
            config.update(
                {"n_movies": self.n_movies,
                    "movies_offset": self.movies_offset,
                    "n_genres": self.n_genres,
                    'incl_genres': self.incl_genres
                })
            return config
    
    @keras.utils.register_keras_serializable(package=package)
    class CandidateModel(keras.Model):
        """Model for encoding candidate features."""
        
        # for init from a load, arguments are present for the compositional instance members too
        def __init__(self, n_movies: int, movies_offset: int, n_genres: int,
                layer_sizes,
                regl2: float = 0.0,
                drop_rate: float = 0., incl_genres: bool = True,
                name='candidate_model', **kwargs):
            """Model for encoding candidate features.
    
            Args:
              layer_sizes:
                A list of integers where the i-th entry represents the number of units
                the i-th layer contains.
              movies_offset: is num_users + 1 where num_uses is the number of users in the catalog.
            """
            super(CandidateModel, self).__init__(name=name, **kwargs)
            
            self.movie_model = MovieModel(n_movies=n_movies,
                movies_offset=movies_offset,
                n_genres=n_genres,
                incl_genres=incl_genres, name="movie_model")
            
            self.dense_candidate = keras.Sequential(name="dense_candidate")
            if isinstance(layer_sizes, str):
                layer_sizes = json.loads(layer_sizes)
            reg = None
            # Use the ReLU activation for all but the last layer.
            for layer_size in layer_sizes[:-1]:
                if regl2 > 0.0:
                    reg = keras.regularizers.l2(regl2)
                self.dense_candidate.add(
                    keras.layers.Dense(layer_size,
                        activation="elu",
                        kernel_regularizer=reg,
                        kernel_initializer="glorot_normal",
                        use_bias=False, name=f'{layer_size}'
                    ))
                # self.dense_query.add(keras.layers.BatchNormalization())
                self.dense_candidate.add(keras.layers.LayerNormalization())
                # self.dense_query.add(keras.activations.elu())
                self.dense_candidate.add(keras.layers.Dropout(drop_rate))
            
            for layer_size in layer_sizes[-1:]:
                self.dense_candidate.add(keras.layers.Dense(layer_size,
                    kernel_initializer="glorot_normal", use_bias=False, name=f'_layers'))
                    
            self.bias_concat_layer = CandidateBiasConcatenationLayer(n_movies=n_movies, movies_offset=movies_offset)
            
            # removing the noramlization layers to allow the models to use dot product instead
            # of cosine similarity for more personalized ANN searches that use the magnitudes
            # in addition to the directions
            
            self.regl2 = regl2
            self.n_movies = n_movies
            self.movies_offset = movies_offset
            self.n_genres = n_genres
            self.incl_genres = incl_genres
            self.drop_rate = drop_rate
            self.layer_sizes = layer_sizes
        
        def build(self, input_shape):
            # print(f'build {self.name} input_shape={input_shape}\n')
            self.movie_model.build(input_shape)
            input_shape_2 = self.movie_model.compute_output_shape(input_shape)
            self.dense_candidate.build(input_shape_2)
            dense_out_shape = self.dense_candidate.compute_output_shape(input_shape_2)
            self.bias_concat_layer.build((dense_out_shape, input_shape['movie_id']))
            self.built = True
        
        def compute_output_shape(self, input_shape):
            # print(f'compute_output_shape {self.name} input_shape={input_shape}\n')
            # This is invoked after build by TwoTower
            input_shape_2 = self.movie_model.compute_output_shape(input_shape)
            dense_out_shape = self.dense_candidate.compute_output_shape(input_shape_2)
            #bias_out_shape = (dense_out_shape[0], 1)
            output_shape = self.bias_concat_layer.compute_output_shape((dense_out_shape, input_shape['movie_id']))
            return output_shape
        
        def call(self, inputs, **kwargs):
            # inputs should contain columns "movie_id", "genres"
            # logging.debug(f'call {self.name} type ={type(inputs)}\ntype ={inputs}\n')
            feature_embedding = self.movie_model(inputs, **kwargs)
            # tf.print('invoked movie_emb.  shape=', feature_embedding.shape)
            res = self.dense_candidate(feature_embedding)
            # returns an np.ndarray wrapped in a tensor if inputs is tensor, else not wrapped
            # logging.debug(f'CALL {self.name} SHAPE ={res.shape}\n')
            # tf.print('CALL', self.name, ' shape=', res.shape)
            return self.bias_concat_layer((res, inputs['movie_id']))
        
        def get_config(self):
            config = super(CandidateModel, self).get_config()
            config.update(
                {"n_movies": self.n_movies,
                    "movies_offset": self.movies_offset,
                    "n_genres": self.n_genres,
                    "drop_rate": self.drop_rate,
                    "layer_sizes": self.layer_sizes,
                    "regl2": self.regl2,
                    "incl_genres": self.incl_genres
                })
            return config
    
    return CandidateModel(n_movies=n_movies, movies_offset=movies_offset,
                                            n_genres=n_genres,
                                            layer_sizes=layer_sizes,
                                            regl2=regl2,
                                            drop_rate=drop_rate,
                                            incl_genres=incl_genres,
                                            **kwargs)
    
def _make_2tower_keras_model(hp: keras_tuner.HyperParameters) -> tf.keras.Model:
  
  #input_dataset_element_spec_raw = hp.get("input_dataset_element_spec_raw_ser")
  #input_dataset_element_spec_raw = pickle.loads(base64.b64decode(input_dataset_element_spec_raw.encode('utf-8')))
  
  logging.info("_make_2tower_keras_model")
  
  input_dataset_element_spec_trans = hp.get("input_dataset_element_spec_trans_ser")
  input_dataset_element_spec_trans = pickle.loads(base64.b64decode(input_dataset_element_spec_trans.encode('utf-8')))
  
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
         layer_sizes: list,
         regl2: float = 0.0,
         drop_rate: float = 0,
         feature_acronym: str = "",
         use_bias_corr: bool = True,
         bias_corr_alpha: float=0.1,
         log_q_correction_factor: float=0.5,
         incl_genres: bool = True,
         temperature:float=1.0, name='twotowerdnn', **kwargs):
      super(TwoTowerDNN, self).__init__(name=name, **kwargs)
      
      self.query_model = _make_query_model(n_users=n_users,
                                    layer_sizes=layer_sizes,
                                    regl2=regl2,
                                    drop_rate=drop_rate,
                                    feature_acronym=feature_acronym,
                                    **kwargs)
      
      self.candidate_model = _make_candidate_model(n_movies=n_movies, movies_offset=movies_offset,
                                            n_genres=n_genres,
                                            layer_sizes=layer_sizes,
                                            regl2=regl2,
                                            drop_rate=drop_rate,
                                            incl_genres=incl_genres,
                                            **kwargs)
      
      if isinstance(layer_sizes, str):
          layer_sizes = json.loads(layer_sizes)
      
      # only used while inspecting table_B for threshold for dataset
      self.calc_table_B_diagnostic = False
      
      self.dot_layer = keras.layers.Dot(axes=1, name='dot_layer')
      #to use HeuristicLambdaLoss, train with the positives of the dataset splits
      #self.loss_function = HeuristicLambdaLoss()
      #to use LambdaSoftmaxLoss, train with full dataset splits
      #self.loss_function = LambdaSoftmaxLoss(temperature = temperature)
      self.mean_loss_metric = keras.metrics.Mean(name="mean_loss")
      self.mrr_k_metric = MeanReciprocalRankAtK(k=20)
      self.ndcg_k_metric = NDCGAtKForInBatchNegatives(k=20)
      self.recall_k_metric = RecallAtKForInBatchNegatives(k=20)
      self.in_batch_hit_rate_metric = InBatchHitRate()
      
      self.regl2 = regl2
      
      # 1.0 : Full popularity bias
      # 0.5 : A balanced space that allows the latent tail to emerge
      # 0.0 : Heavy popularity penalty.
      self.log_q_correction_factor = log_q_correction_factor
      
      self.n_users = n_users
      self.n_movies = n_movies
      self.movies_offset = movies_offset
      self.n_genres = n_genres
      self.incl_genres = incl_genres
      self.layer_sizes = layer_sizes
      self.feature_acronym = feature_acronym
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
          # B holds the exponential moving average of the delta t step gap, that is,
          # the average time in steps for a movie to be seen again.
          # a low value means the movie is seen frequently (i.e. a popular movie),
          # while a high value means the movie is not seen very often.
          self.table_B = tf.lookup.experimental.MutableHashTable(
              key_dtype=tf.int32, value_dtype=tf.float32, default_value=1.0)
          self.global_step = tf.Variable(0., trainable=False,
              dtype=tf.float32)
      else:
          self.table_B = None
      self.ndcg_k_composite_metric = NDCGAtKComposite(k=20, use_composite=self.use_bias_corr)
    
    @property
    def metrics(self):
        # OVERRIDE to workaround tf.keras handling of validation metrics
        # It tells the model: "When you finish an epoch, pull results from these two."
        if self.use_bias_corr:
            return [self.mean_loss_metric, self.in_batch_hit_rate_metric, self.mrr_k_metric, self.recall_k_metric,
                self.ndcg_k_metric, self.ndcg_k_composite_metric,
                self.ndcg_k_composite_metric.ndcg_head,
                self.ndcg_k_composite_metric.ndcg_torso,
                self.ndcg_k_composite_metric.ndcg_tail,
                ]
        else:
            return [self.mean_loss_metric, self.in_batch_hit_rate_metric, self.mrr_k_metric, self.recall_k_metric,
                self.ndcg_k_metric, self.ndcg_k_composite_metric]
        
    def call(self, inputs):
      """
      compute the dot product  score for the user data to movie data.
      
      score = <q, c> + b_m where q is query embedding and c is candidate embedding and b_m is a
          movie bias term that is internally added by dedicating the last element in q to be value 1.0
          and the last element in  c to be the learned b_m.
      NOTE that the q and c embeddings are not normalized so the result is dot product, not cosine similarity.
      NOTE that the resulting trained embeddings from the trained QueryModel and trained CandidateModel
      will have embedding vector magnitudes and direction which helps improve personalized ANN searches.
      
      Args:
         inputs: transformed features
      Returns:
          dot product score for the user data to movie data
      """
      #logging.debug(f'call {self.name} inputs={inputs}\n')
      
      
      
      user_vector = self.query_model(inputs)
      movie_vector = self.candidate_model(inputs)
      #tf.print('U,V SHAPES: ', user_vector.shape, movie_vector.shape)
      s = self.dot_layer([user_vector, movie_vector])
      return s
   
    def build(self, input_shape):
      #tf.print("TwoTowerDNN build input_shape=", input_shape)
      normalized_input_shape = {
        k: tf.TensorShape(v) if not isinstance(v, tf.TensorShape) else v
        for k, v in input_shape.items()
      }
      #tf.print("TwoTowerDNN build normalized_nput_shape=", normalized_input_shape)
      #print(f'build {self.name} input_shape={input_shape}\n')
      # logging.debug(f'build {self.name} input_shape={input_shape}\n')
      if not self.query_model.built:
        self.query_model.build(normalized_input_shape)
      if not self.candidate_model.built:
        self.candidate_model.build(normalized_input_shape)
      s0 = self.query_model.compute_output_shape(normalized_input_shape)
      s1 = self.candidate_model.compute_output_shape(normalized_input_shape)
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
        """frequency estimation logic from Yi et al."""
        self.global_step.assign_add(1.0)
        t = self.global_step
        
        movie_ids_int = tf.cast(movie_ids, tf.int32)
        movie_ids_flat = tf.reshape(movie_ids_int, [-1])
        
        last_t = self.table_A.lookup(movie_ids_flat)
        B_old = self.table_B.lookup(movie_ids_flat)
        
        delta_t = tf.cast(t - last_t, tf.float32)
        
        B_new = (1.0 - self.bias_corr_alpha) * B_old + self.bias_corr_alpha * delta_t
        
        self.table_A.insert(movie_ids_flat, tf.fill(tf.shape(movie_ids_flat), t))
        self.table_B.insert(movie_ids_flat, B_new)
        
        # Remove tf.maximum(B_new, 1.0).
        # If B_new is 0.2 (appears 5 times per batch), p_i correctly becomes 5.0
        return 1.0 / tf.maximum(B_new, 1e-6)
    
    def in_batch_softmax_loss_function(self, y_true, logits):
        """
        y_true: The original ratings (used as sample weights) [Batch]
        logits: The [Batch, Batch] matrix after Log-Q correction and Temperature
        """
        batch_size = tf.shape(logits)[0]
        labels = tf.range(batch_size)
        
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        
        return tf.reduce_mean(loss)
    
    def train_step(self, batch):
        x, y = batch  # y is typically not used in pure In-Batch Softmax (identity matrix is the target)
        movie_ids = x['movie_id']
        beta_correction = self.log_q_correction_factor
        with tf.GradientTape() as tape:
            user_embeddings = self.query_model(x)  # [Batch, Dim]
            movie_embeddings = self.candidate_model(x)  # [Batch, Dim]
            
            # Compute ALL-TO-ALL Similarity (In-Batch Softmax)
            # scores[i, j] is similarity between user i and movie j
            # this is [batch_size X batch_size] and the diagonal is the dot product
            raw_logits = tf.matmul(user_embeddings, movie_embeddings, transpose_b=True)
            logits = raw_logits / self.temperature
            
            pre_logit_max = tf.reduce_max(logits)
            pre_logit_min = tf.reduce_min(logits)
            pre_logit_mean = tf.reduce_mean(logits)

            if self.use_bias_corr:
                # Get frequency corrections
                p_i = self._update_frequencies(movie_ids)
                # Allow p_i to act as expected count, which can be > 1.0 for blockbusters
                # Do not clip the upper bound to 1.0, only protect against log(0)
                log_q = tf.math.log(tf.maximum(p_i, 1e-6))
                # Apply Log-Q correction to columns (the candidate side)
                # Broad-casting log_q across the batch
                #logits = logits - tf.expand_dims(log_q, axis=0)
                logits = logits - (beta_correction * tf.expand_dims(log_q, axis=0))
            
            #logits is [batch_size x batch_size] to use non-diagonal elements as negatives for the
            #diagonal elements.
            # but to avoid counting the same user's other ratings as negative and to avoid counting
            # the same movie rated by the user as having more than one rating, masks are made to
            # assure those elements are not considered in the rank calculations that are in the
            # in batch loss and metric functions
            #shapes: batch_Size x batch_size
            user_mask = tf.equal(tf.expand_dims(x['user_id'], 0), tf.expand_dims(x['user_id'], 1))
            movie_mask = tf.equal(tf.expand_dims(movie_ids, 0), tf.expand_dims(movie_ids, 1))
            user_mask = tf.squeeze(user_mask, axis=-1)
            movie_mask = tf.squeeze(movie_mask, axis=-1)
            mask = tf.logical_or(user_mask, movie_mask)
            # if diagonal got masked, unmask it
            batch_size = tf.shape(logits)[0]
            identity_mask = tf.eye(batch_size, dtype=tf.bool)
            mask = tf.logical_and(mask, tf.logical_not(identity_mask))
            masked_logits = tf.where(mask, tf.constant(-1e9, dtype=logits.dtype), logits)
            masked_raw_logits = tf.where(mask, tf.constant(-1e9, dtype=logits.dtype), raw_logits)
            loss = self.in_batch_softmax_loss_function(y, masked_logits)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.mean_loss_metric.update_state(loss)
        
        labels = tf.range(batch_size)
        
        self.mrr_k_metric.update_state(labels, masked_raw_logits, sample_weight=y)
        self.ndcg_k_metric.update_state(labels, masked_raw_logits, sample_weight=y)
        self.recall_k_metric.update_state(labels, masked_raw_logits, sample_weight=y)
        self.in_batch_hit_rate_metric(y_true=labels, y_pred=masked_raw_logits, sample_weight=y)
        
        self.ndcg_k_composite_metric.update_state(
            y_true=labels,
            y_pred=masked_raw_logits,
            movie_ids=movie_ids,
            table_b=self.table_B,
            sample_weight=y
        )
        
        output = {m.name: m.result() for m in self.metrics}
        #DEBUG: =====
        output.update({
            "pre_logit_max": pre_logit_max,
            "pre_logit_min": pre_logit_min,
            "pre_logit_mean": pre_logit_mean,
            "logit_max" : tf.reduce_max(logits),
            "logit_min" : tf.reduce_min(logits),
            "logit_mean" : tf.reduce_mean(logits),
        })
        return output
    
    def test_step(self, data):
        x, y = data
        movie_ids = x['movie_id']
        user_embeddings = self.query_model(x, training=False)
        movie_embeddings = self.candidate_model(x, training=False)
        #in test and eval, do not divide by temperature
        raw_logits = tf.matmul(user_embeddings, movie_embeddings, transpose_b=True)
        logits = raw_logits / self.temperature
        batch_size = tf.shape(logits)[0]
        
        # Define Ranking Labels
        user_mask = tf.equal(tf.expand_dims(x['user_id'], 0),
            tf.expand_dims(x['user_id'], 1))
        movie_mask = tf.equal(tf.expand_dims(x['movie_id'], 0),
            tf.expand_dims(x['movie_id'], 1))
        user_mask = tf.squeeze(user_mask, axis=-1)
        movie_mask = tf.squeeze(movie_mask, axis=-1)
        mask = tf.logical_or(user_mask, movie_mask)
        #if diagonal got masked, unmask it
        identity_mask = tf.eye(batch_size, dtype=tf.bool)
        mask = tf.logical_and(mask, tf.logical_not(identity_mask))
        masked_logits = tf.where(mask, tf.constant(-1e9, dtype=logits.dtype),
            logits)
        masked_raw_logits = tf.where(mask, tf.constant(-1e9, dtype=raw_logits.dtype),
            raw_logits)
        #loss = self.loss_function(y, logits)
        loss = self.in_batch_softmax_loss_function(y, masked_logits)
        
        labels = tf.range(batch_size)
        self.mean_loss_metric.update_state(loss)
        self.mrr_k_metric.update_state(labels, masked_raw_logits, sample_weight=y)
        self.ndcg_k_metric.update_state(labels, masked_raw_logits, sample_weight=y)
        self.recall_k_metric.update_state(labels, masked_raw_logits, sample_weight=y)
        self.in_batch_hit_rate_metric.update_state(y_true=labels, y_pred=masked_raw_logits, sample_weight=y)
        
        #def update_state(self, y_true, y_pred, sample_weight=None, table_b = None, movie_ids=None)
        self.ndcg_k_composite_metric.update_state(
            y_true=labels,
            y_pred=masked_raw_logits,
            table_b=self.table_B,
            movie_ids=movie_ids,
            sample_weight=y
        )
        
        output = {m.name: m.result() for m in self.metrics}
        # DEBUG: =====
        logit_max = tf.reduce_max(logits)
        logit_min = tf.reduce_min(logits)
        logit_mean = tf.reduce_mean(logits)
        output.update({
            "logit_max": logit_max,
            "logit_min": logit_min,
            "logit_mean": logit_mean
        })
        return output
    
    #do not invoke this from within a tf_function because tensors won't have numpy
    def inspect_table_B_distribution(self, table_b, bins: int = 10,
            max_bar_width: int = 35):
        """
        Exports values from a MutableHashTable and prints percentile stats
        and a terminal-friendly ASCII histogram.
        """
        import numpy as np
        # Export all keys and values from the table
        keys, values = table_b.export()
        
        # onvert to numpy for statistical analysis
        b_values = values.numpy()
        
        if len(b_values) == 0:
            print("table_B is currently empty. Run a training pass first.")
            return
        
        print(f"Total unique items in table_B: {len(b_values)}")
        print("=" * 60)
        
        # Basic Stats
        print(f"Min value:  {np.min(b_values):.2f}")
        print(f"Max value:  {np.max(b_values):.2f}")
        print(f"Mean value: {np.mean(b_values):.2f}")
        print("-" * 60)
        
        # Percentiles
        percentiles = [20, 50, 75, 80, 85, 90, 95, 99]
        calc_percentiles = np.percentile(b_values, percentiles)
        
        for p, val in zip(percentiles, calc_percentiles):
            print(f"{p:>2}th percentile: {val:>8.2f}")
        
        print("-" * 60)
        print(f"Recommendation: For a 20% head cutoff, set head_threshold = {calc_percentiles[0]:.2f}")
        print(f"Recommendation: For an 80% tail cutoff, set b_threshold = {calc_percentiles[3]:.2f}")
        print("=" * 60)
        
        # Terminal ASCII Histogram
        counts, bin_edges = np.histogram(b_values, bins=bins)
        max_count = max(counts) if max(counts) > 0 else 1
        
        print("\nASCII Histogram (B_new Distribution):")
        print("Range of B_new           | Count   | Distribution")
        print("-" * 60)
        
        for i in range(len(counts)):
            low = bin_edges[i]
            high = bin_edges[i + 1]
            count = counts[i]
            
            # Calculate bar length relative to max count
            bar_len = int((count / max_count) * max_bar_width)
            bar = "█" * bar_len
            
            # Print formatted row
            range_str = f"[{low:>7.1f} - {high:>7.1f})"
            print(f"{range_str} | {count:>7d} | {bar}")
        print("=" * 60)
    
    def fit(self, *args, **kwargs):
        history = super().fit(*args, **kwargs)
        
        # 2. Run diagnostic hook at the end of the fit call
        if self.calc_table_B_diagnostic:
            self.inspect_table_B_distribution(self.table_B)

        try:
            tf.print("num_epochs=", len(history.history['mean_loss']))
        except Exception:
            pass
        
        return history
    
    def get_config(self):
      config = super(TwoTowerDNN, self).get_config()
      config.update({"n_users": self.n_users, "n_movies": self.n_movies,
        "movies_offset" : self.movies_offset,
        "n_genres": self.n_genres,
        "drop_rate": self.drop_rate,
        "layer_sizes": self.layer_sizes,
        "use_bias_corr": self.use_bias_corr,
        "feature_acronym": self.feature_acronym,
        "regl2": self.regl2,
        "incl_genres": self.incl_genres,
        "bias_corr_alpha": self.bias_corr_alpha,
        "log_q_correction_factor": self.log_q_correction_factor,
        "temperature": self.temperature,
        })
      return config
    
    #override needed for keras_tuner utils.save_json
    # if TensorShape remains in return from model.get_build_config(), there is error here:
    # https://github.com/keras-team/keras-tuner/blob/48f671490201f6b873e4d27dee8df6f406256ca4/keras_tuner/engine/tuner.py#L237
    # this override removes the TensorShapes and then a complementary fix in the def build method puts the TensorShapes back in.
    def get_build_config(self):
        build_config = super().get_build_config()
        # If input_shape is a dict of TensorShapes, convert them to tuples/lists recursively:
        if "input_shape" in build_config:
            original_shapes = build_config["input_shape"]
            if isinstance(original_shapes, dict):
                build_config["input_shape"] = {
                    k: tuple(v.as_list()) if hasattr(v, "as_list") and v.rank is not None else v
                    for k, v in original_shapes.items()
                }
            elif hasattr(original_shapes, "as_list"):
                build_config["input_shape"] = tuple(original_shapes.as_list())
        return build_config
  
  @keras.utils.register_keras_serializable(package=package)
  class HeuristicLambdaLoss(keras.losses.Loss):
      """
      a rank-weighted InfoNCE loss.   focuses the model's gradient on pairs
      where the positive item is currently at a lower rank than negatives.

      softmax cross entropy loss treats the retrieval as one-vs-all classification
      which ignores the magnitude of the error in rank.

      LambdaRank (Burgess from MS) created to work around non-differentiable
      NDCG.  It uses a differentiable smooth loss function like Softmax,
      multiplies the gradient of it by the change in ndcg from the swapping
      of item positions.
      
      Heuristic Lambda Rank uses that as a weight to push the item
      towards its correct rank.  It is an O(N=batch_size) algorithm
      while LambdaRank is O(N^2).
      It avoids the vanishing gradient problem.

      Heuristic Lambda Rank is commonly seen in systems like Two-Tower Retrieval,
      especially with large datasets.  It trains the bi-encoders to produce
      a topk similarity list that is rank aware.
      NOTE: when training, the datasets should include all ratings, 1-5.
      """
      def __init__(self, name="heuristic_lambda", **kwargs):
          super(HeuristicLambdaLoss, self).__init__(name=name, **kwargs)
      
      def call(self, y_true, y_pred):
          """
          y_true: [batch_size]
          y_pred:  [batch_size, batch_size].  the result of matmul Q_embedd, C_embed^T / temperature
          """
          logits = y_pred
          batch_size = tf.shape(logits)[0]
          pos_indices = tf.range(tf.shape(logits)[0])
          pos_scores = tf.linalg.diag_part(logits)[:, tf.newaxis]
          
          #current rank of pos item in its row.
          # use a differentiable approx or stop gradient for the rank
          is_greater_equal = tf.cast(logits >= (pos_scores - 1e-6), tf.float32)
          ranks = tf.reduce_sum(is_greater_equal, axis=1) - 1.0
          ranks = tf.stop_gradient(tf.maximum(ranks, 0.0))
          
          #heuristic rank: 1/(log2(1 + rank)
          # compute stop gradient to avoid backprop thru rank logic
          weights = 1.0 / tf.math.log1p(ranks + 1.0)

          #cross entropy scaled by weights
          loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pos_indices, logits=logits)
          relevance = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
          loss = loss * weights * relevance
          return tf.reduce_mean(loss)
      
      def get_config(self):
          config = super(HeuristicLambdaLoss, self).get_config()
          return config

  @keras.utils.register_keras_serializable(package=package)
  class LambdaSoftmaxLoss(keras.losses.Loss):
      """
      Listwise loss that weights the softmax cross-entropy by the
      potential Delta NDCG of the positive item...a hybrid of LambdaRank and Softmax Cross-Entropy
      """
      def __init__(self, name="lambda_softmax", **kwargs):
          super().__init__(name=name, **kwargs)
      
      def call(self, y_true, y_pred):
          """
          y_true: [batch_size]
          y_pred:  [batch_size, batch_size].  the result of matmul Q_embedd, C_embed^T / temperature
          """
          logits = y_pred
          batch_size = tf.shape(logits)[0]
          
          # the positive score (diagonal).  [batch_size, 1]
          pos_scores = tf.linalg.diag_part(logits)[:, tf.newaxis]
          
          # Calculate current Rank (1-based)
          # Count how many items in the batch have a higher score than the positive
          is_greater_equal = tf.cast(logits >= (pos_scores - 1e-6), tf.float32)
          ranks = tf.reduce_sum(is_greater_equal, axis=1)
          ranks = tf.stop_gradient(ranks)
          
          # Calculate Relevance Gain (using the ratings y)
          # Standard NDCG gain formula: (2^rel - 1)
          # If ratings are 0-1, this still works; if 1-5, it scales significantly.
          relevance = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
          gain = tf.math.pow(2.0, relevance) - 1.0
          
          # Calculate Delta NDCG weight
          # We calculate the benefit of moving the item from its current rank to Rank 1
          log_2 = tf.math.log(tf.cast(2, ranks.dtype))
          current_discount = 1.0 / (tf.math.log(ranks + 1.0)/log_2)
          ideal_discount = 1.0# 1.0 / (tf.math.log(1.0 + 1.0)/log_2) #1.0
          delta_ndcg = (gain * (ideal_discount - current_discount)) + 1.0
          
          # 6. Weighted Cross Entropy
          pos_indices = tf.range(batch_size) #index of the "1" positive in the row of the batch.  it's the column in Indentity matrix where 1 is.
          ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
              labels=pos_indices, logits=logits)
          
          # Applying delta_ndcg as a per-example weight
          return tf.reduce_mean(ce_loss * delta_ndcg)
          
      def get_config(self):
          config = super(LambdaSoftmaxLoss, self).get_config()
          return config
  
  @keras.utils.register_keras_serializable(package=package)
  class InBatchHitRate(keras.metrics.Metric):
      def __init__(self, name="hit_rate", **kwargs):
          super(InBatchHitRate, self).__init__(name=name, **kwargs)
          self.hits = self.add_weight(name="total_hits",
              initializer="zeros")
          self.count = self.add_weight(name="total_count",
              initializer="zeros")
      
      def update_state(self, y_true, y_pred, sample_weight=None):
          """
          y_true: Ignored here (internally generated as tf.range), or used for weights
          y_pred: The [Batch, Batch] logits matrix
          sample_weight: normalized ratings (y) from the dataset
          """
          batch_size = tf.shape(y_pred)[0]
          
          targets = tf.range(batch_size, dtype=tf.int32)
          
          # Find the predicted index (the movie with the highest similarity)
          # y_pred shape: [Batch, Batch] -> preds shape: [Batch]
          preds = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
          
          # 4. Compare: [Batch] boolean vector
          is_correct = tf.equal(preds, targets)
          is_correct = tf.cast(is_correct, tf.float32)
          
          weights = sample_weight if sample_weight is not None else y_true
          if weights is not None:
              weights = tf.cast(tf.reshape(weights, [-1]), tf.float32)
              is_correct = is_correct * weights
              self.count.assign_add(tf.reduce_sum(weights))
          else:
              self.count.assign_add(tf.cast(batch_size, tf.float32))
          self.hits.assign_add(tf.reduce_sum(is_correct))
      
      def result(self):
          return tf.math.divide_no_nan(self.hits, self.count)
      
      def reset_state(self):
          self.hits.assign(0.0)
          self.count.assign(0.0)
  
  @keras.utils.register_keras_serializable(package=package)
  class MeanReciprocalRankAtK(keras.metrics.Metric):
      """
      """
      def __init__(self, name="mrr", k:int=10,**kwargs):
          name = f"{name}_{k}"
          super(MeanReciprocalRankAtK, self).__init__(name=name, **kwargs)
          self.k = k
          self.mrr_sum = self.add_weight(name="mrr_sum", initializer="zeros")
          self.count = self.add_weight(name="count", initializer="zeros")

      def update_state(self, y_true, y_pred, sample_weight=None):
          """
          y_true: Ignored here (internally generated as tf.range), or used for weights.  use an identity
                  matrix with same shape as y_pred
          y_pred: The [Batch, Batch] logits matrix result of matmul Q_embedd * C_embed^T
          sample_wieght: use the ground truth labels here, i.e. y
          """
          pos_scores = tf.linalg.diag_part(y_pred)[:, tf.newaxis]
          is_greater_equal = tf.cast(y_pred >= (pos_scores - 1e-6), tf.float32)
          ranks = tf.reduce_sum(is_greater_equal, axis=1)
          reciprocal_rank = 1.0 / ranks
          k = tf.cast(self.k, tf.float32)
          reciprocal_rank = tf.where(ranks <= k, reciprocal_rank, 0.0)
          
          weights = sample_weight if sample_weight is not None else y_true
          weights = tf.cast(tf.reshape(weights, [-1]), tf.float32)
          
          self.mrr_sum.assign_add(tf.reduce_sum(tf.multiply(reciprocal_rank, weights)))
          self.count.assign_add(tf.reduce_sum(weights))
         
      def result(self):
          return tf.math.divide_no_nan(self.mrr_sum, self.count)
      
      def reset_state(self):
          self.mrr_sum.assign(0.0)
          self.count.assign(0.0)
          
      def get_config(self):
          config = super(MeanReciprocalRankAtK, self).get_config()
          config.update({
              "k": self.k
          })
          return config
      
  @keras.utils.register_keras_serializable(package=package)
  class NDCGAtKForInBatchNegatives(keras.metrics.Metric):
      """
      """
      def __init__(self, name="ndcg", k: int = 20, **kwargs):
          name = f"{name}_{k}"
          super(NDCGAtKForInBatchNegatives, self).__init__(name=name, **kwargs)
          self.k = k
          self.ndcg_sum = self.add_weight(name="ndcg_sum", initializer="zeros")
          self.count = self.add_weight(name="count", initializer="zeros")
      
      def update_state(self, y_true, y_pred, sample_weight=None):
          """
          y_true: Ignored here (internally generated as tf.range), or used for weights.  use an identity
                  matrix with same shape as y_pred
          y_pred: The [Batch, Batch] logits matrix result of matmul Q_embedd * C_embed^T
          weights: the actual ground truth, y_true vector should be set here
          """
          #[batch_size, 1]
          pos_scores = tf.linalg.diag_part(y_pred)[:, tf.newaxis]
          is_greater_equal = tf.cast(y_pred >= (pos_scores - 1e-6), tf.float32)
          ranks = tf.reduce_sum(is_greater_equal,  axis=1)  # where relevance is >= diagonal
          log2_rank = tf.math.log(ranks + 1.0) / tf.math.log(2.0)
          k = tf.cast(self.k, tf.float32)
          relevant_mask = tf.cast(ranks <= k, tf.float32)
          # NDCG = (1 / log2(rank + 1)) / IDCG. Since IDCG is 1.0 here:
          ndcg = (1.0 / log2_rank) * relevant_mask
          
          if sample_weight is not None:
              weights = tf.cast(sample_weight, tf.float32)
              weights = tf.reshape(weights, [-1])
              self.ndcg_sum.assign_add(tf.reduce_sum(ndcg * weights))
              self.count.assign_add(tf.reduce_sum(weights))
          else:
              self.ndcg_sum.assign_add(tf.reduce_sum(ndcg))
              self.count.assign_add(tf.cast(tf.shape(y_pred)[0], tf.float32))
      
      def result(self):
          return tf.math.divide_no_nan(self.ndcg_sum, self.count)
      
      def reset_state(self):
          self.ndcg_sum.assign(0.0)
          self.count.assign(0.0)
      
      def get_config(self):
          config = super().get_config()
          config.update({
              "k": self.k
          })
          return config
  
  @keras.utils.register_keras_serializable(package=package)
  class NDCGAtKForInBatchPart(keras.metrics.Metric, abc.ABC):
    """
    Computes NDCG@K restricted strictly to rows where the true positive
    candidate item is a tail item based on the streaming table_B state.
    """
    def __init__(self, head_torso_tail_idx:int =2, name:str ="ndcg", k: int = 20, **kwargs):
        if head_torso_tail_idx < 0 or head_torso_tail_idx > 2:
            raise ValueError("head_torso_tail_idx must be >= 0 and <= 2")
        name = self.get_name(name, k)
        super(NDCGAtKForInBatchPart, self).__init__(name=name, **kwargs)
        self.k = k
        self.head_torso_tail_idx=head_torso_tail_idx
        self.ndcg_sum = self.add_weight(name="ndcg_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
    
    @abc.abstractmethod
    def get_name(self, name:str, k:int):
        raise NotImplementedError("This method must be implemented by the inheritor.")
    
    @abc.abstractmethod
    def get_part_mask(self, b_values):
        raise NotImplementedError("This method must be implemented by the inheritor.")
    
    def update_state(self, labels, logits, movie_ids,
            table_b:tf.lookup.experimental.MutableHashTable, sample_weight=None):
        # 1. Identify Tail Items
        movie_ids_flat = tf.cast(tf.reshape(movie_ids, [-1]), tf.int32)
        b_values = table_b.lookup(movie_ids_flat)
        
        part_mask = self.get_part_mask(b_values)
        
        # 2. Efficient Ranking (Aligned with NDCGAtKForInBatchNegatives)
        pos_scores = tf.linalg.diag_part(logits)[:, tf.newaxis]
        is_greater_equal = tf.cast(logits >= (pos_scores - 1e-6), tf.float32)
        ranks = tf.reduce_sum(is_greater_equal, axis=1)
        
        k = tf.cast(self.k, tf.float32)
        relevant_mask = tf.cast(ranks <= k, tf.float32)
        log2_rank = tf.math.log(ranks + 1.0) / tf.math.log(2.0)
        
        # NDCG for ALL items in batch
        dcg = (1.0 / log2_rank) * relevant_mask
        
        # 3. Filter accumulator to include only tail rows
        weights = tf.cast(part_mask, tf.float32)
        if sample_weight is not None:
            weights = weights * tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
        
        self.ndcg_sum.assign_add(tf.reduce_sum(dcg * weights))
        self.count.assign_add(tf.reduce_sum(weights))
    
    def result(self):
        return tf.math.divide_no_nan(self.ndcg_sum, self.count)
    
    def reset_state(self):
        self.ndcg_sum.assign(0.0)
        self.count.assign(0.0)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "k": self.k,
            "head_torso_tail_idx": self.head_torso_tail_idx,
        })
        return config
  
  @keras.utils.register_keras_serializable(package=package)
  class NDCGAtKForInBatchHead(NDCGAtKForInBatchPart):
      """
      Computes NDCG@K restricted strictly to rows where the true positive
      candidate item is a head item based on the streaming table_B state.
      """
      
      def __init__(self,  b_threshold_head=7.92, name:str="ndcg", k: int = 20, **kwargs):
          super().__init__(head_torso_tail_idx=0, name=name, k=k, **kwargs)
          self.b_threshold_head = b_threshold_head
      
      def get_name(self, name:str,  k:int):
          return f"{name}_head_{k}"
      
      def get_part_mask(self, b_values):
          return b_values <= self.b_threshold_head
      
      def get_config(self):
          config = super().get_config()
          config.update({
              "b_threshold_head": self.b_threshold_head,
          })
          return config
  
  @keras.utils.register_keras_serializable(package=package)
  class NDCGAtKForInBatchTorso(NDCGAtKForInBatchPart):
      """
      Computes NDCG@K restricted strictly to rows where the true positive
      candidate item is a torse item based on the streaming table_B state.
      """
      
      def __init__(self, b_threshold_head=7.92, b_threshold_tail=12.21, name: str = "ndcg",
              k: int = 20, **kwargs):
          super().__init__(head_torso_tail_idx=1, name=name, k=k, **kwargs)
          self.b_threshold_head = b_threshold_head
          self.b_threshold_tail = b_threshold_tail
      
      def get_name(self, name: str, k: int):
          return f"{name}_torso_{k}"
      
      def get_part_mask(self, b_values):
            return tf.logical_and(
                b_values > self.b_threshold_head,
                b_values < self.b_threshold_tail
            )
          
      def get_config(self):
          config = super().get_config()
          config.update({
              "b_threshold_head": self.b_threshold_head,
              "b_threshold_tail": self.b_threshold_tail,
          })
          return config
  
  @keras.utils.register_keras_serializable(package=package)
  class NDCGAtKForInBatchTail(NDCGAtKForInBatchPart):
      """
      Computes NDCG@K restricted strictly to rows where the true positive
      candidate item is a tail item based on the streaming table_B state.
      """
      
      def __init__(self, b_threshold_tail=12.21, name: str = "ndcg", k: int = 20, **kwargs):
          super().__init__(head_torso_tail_idx=2, name=name, k=k, **kwargs)
          self.b_threshold_tail = b_threshold_tail
      
      def get_name(self, name: str, k: int):
          return f"{name}_tail_{k}"
      
      def get_part_mask(self, b_values):
          return  b_values >= self.b_threshold_tail
      
      def get_config(self):
          config = super().get_config()
          config.update({
              "b_threshold_tail": self.b_threshold_tail,
          })
          return config
      
  @keras.utils.register_keras_serializable(package=package)
  class NDCGAtKComposite(keras.metrics.Metric):
    def __init__(self, b_threshold_head=7.92, b_threshold_tail=12.21, name="composite_ndcg",
            k: int = 20, use_composite:bool=True,
            w_head:float=0.3, w_torso:float=0.5, w_tail:float=0.2, **kwargs):
        """
        Args:
            b_threshold_head (float): b_table values less than this are the head of the distribution frequently picked movies but few unique movies.
            b_threshold_tail (float): b_table values greater than this are the tail of the ditstribution rarly picked and are few in number of unique movies.
            k (int): top k to use in NDCG metrics
            use_composite (bool): if False, the standard NDCG metrics is used, else the composite is used.
            The composite is w_head * NDCG_head + w_torso * NDCG_torso + w_tail * NDCG_tail.
        """
        name = f"{name}_{k}"
        super(NDCGAtKComposite, self).__init__(name=name, **kwargs)
        self.b_threshold_head = b_threshold_head
        self.b_threshold_tail = b_threshold_tail
        self.w_head = w_head
        self.w_torso = w_torso
        self.w_tail = w_tail
        self.k = k
        self.use_composite = use_composite
        if use_composite:
            self.ndcg_head = NDCGAtKForInBatchHead(b_threshold_head=b_threshold_head, k=k)
            self.ndcg_torso = NDCGAtKForInBatchTorso(b_threshold_head=b_threshold_head, b_threshold_tail=b_threshold_tail, k=k)
            self.ndcg_tail = NDCGAtKForInBatchTail(b_threshold_tail=b_threshold_tail, k=k)
        else:
            self.ndcg = NDCGAtKForInBatchNegatives(k=k)
    
    def update_state(self, y_true, y_pred, sample_weight=None,
            table_b:tf.lookup.experimental.MutableHashTable = None, movie_ids=None):
        if self.use_composite:
            self.ndcg_head.update_state(y_true, y_pred, movie_ids, table_b, sample_weight)
            self.ndcg_torso.update_state(y_true, y_pred, movie_ids, table_b, sample_weight)
            self.ndcg_tail.update_state(y_true, y_pred, movie_ids, table_b, sample_weight)
        else:
            self.ndcg.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        if self.use_composite:
            s0 = self.w_head * self.ndcg_head.result()
            s1 = self.w_torso * self.ndcg_torso.result()
            s2 = self.w_tail * self.ndcg_tail.result()
            return s0 + s1 + s2
        else:
            return self.ndcg.result()
        
    def reset_state(self):
        if self.use_composite:
            self.ndcg_head.reset_state()
            self.ndcg_torso.reset_state()
            self.ndcg_tail.reset_state()
        else:
            self.ndcg.reset_state()
        
    def get_config(self):
        # Fix: super(NDCGAtKComposite).get_config() throws an error in python.
        # Use super().get_config()
        config = super().get_config()
        config.update({
            "b_threshold_head" : self.b_threshold_head,
            "b_threshold_tail" : self.b_threshold_tail,
            "w_head" : self.w_head,
            "w_torso" : self.w_torso,
            "w_tail" : self.w_tail,
            "k" : self.k,
            "use_composite" : self.use_composite,
        })
        return config
    
  @keras.utils.register_keras_serializable(package=package)
  class RecallAtKForInBatchNegatives(keras.metrics.Metric):
      """
      same as hit_rate_at_k for in batch negatives
      """
      def __init__(self, name="recall", k: int = 100, **kwargs):
          name = f"{name}_{k}"
          super(RecallAtKForInBatchNegatives, self).__init__(name=name, **kwargs)
          self.k = k
          self.hits = self.add_weight(name="hits", initializer="zeros")
          self.count = self.add_weight(name="count", initializer="zeros")
      
      def update_state(self, y_true, y_pred, sample_weight=None):
          """
          y_true: Ignored here (internally generated as tf.range), or used for weights.  use an identity
                  matrix with same shape as y_pred
          y_pred: The [Batch, Batch] logits matrix result of matmul Q_embedd * C_embed^T
          """
          #[batch_Size, 1]
          pos_scores = tf.linalg.diag_part(y_pred)[:, tf.newaxis]
          is_greater_equal = tf.cast(y_pred >= (pos_scores - 1e-6), tf.float32)
          ranks = tf.reduce_sum(is_greater_equal, axis=1)
          k = tf.cast(self.k, tf.float32)
          is_hit = tf.cast(ranks <= k, tf.float32)
          
          if sample_weight is not None:
              w = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
          elif y_true is not None and tf.rank(y_true) == 1:
              # Likely a vector of ratings
              w = tf.cast(y_true, tf.float32)
          else:
              # Unweighted: every row in the batch counts as 1
              w = tf.ones_like(is_hit)
          
          self.hits.assign_add(tf.reduce_sum(tf.multiply(is_hit, w)))
          self.count.assign_add(tf.reduce_sum(w))
      
      def result(self):
          return tf.math.divide_no_nan(self.hits, self.count)
      
      def reset_state(self):
          self.hits.assign(0.0)
          self.count.assign(0.0)
      
      def get_config(self):
          config = super(RecallAtKForInBatchNegatives, self).get_config()
          config.update({
              "k": self.k
          })
          return config
  
  # use strategy
  strategy, device = _get_strategy()
  
  #METRICS_FN_LIST = [InBatchHitRate(name="hit_rate")]
  #tf.keras.metrics.SparseCategoricalAccuracy(name="acc")
  
  with strategy.scope():
    model = TwoTowerDNN(
      n_users=hp.get("n_users") + 1,
      n_movies=hp.get("n_movies") + 1,
      movies_offset = hp.get("n_users") + 1,
      n_genres=hp.get("n_genres"),
      layer_sizes=hp.get('layer_sizes'),
      regl2=hp.get('regl2'),
      drop_rate=hp.get('drop_rate'),
      feature_acronym=hp.get("feature_acronym"),
      use_bias_corr=hp.get('use_bias_corr'),
      bias_corr_alpha = hp.get('bias_corr_alpha'),
      log_q_correction_factor = hp.get('log_q_correction_factor'),
      incl_genres=hp.get('incl_genres'),
      temperature=hp.get('temperature'),
    )
    
    # call model once to trace methods.
    fake_trans_ds = create_fake_transformed_batch(input_dataset_element_spec_trans)
    
    build_input_shapes = {
        feat: tensor.shape for feat, tensor in fake_trans_ds.items()
    }
    
    # initialize the variables, rooted at these models so that variable names do not include parent in definition
    model.query_model.build(build_input_shapes)
    model.candidate_model.build(build_input_shapes)
    model.build(build_input_shapes)
    
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
        #run_eagerly=True,
    )
  
  model.summary(print_fn=logging.info)
  print('MODEL SUMMARY:') 
  trainable_count = sum([keras.backend.count_params(w) for w in model.trainable_weights])
  non_trainable_count = sum([keras.backend.count_params(w) for w in model.non_trainable_weights])
  print(f"Total Trainable Params: {trainable_count}")
  print(f"Total Non-Trainable Params: {non_trainable_count}")
  #model.summary(expand_nested=True)
  
  return model

def get_default_hyperparameters(custom_config) -> keras_tuner.HyperParameters:
  """Returns hyperparameters for building Keras model."""
  #print(f'get_default_hyperparameters: custom_config={custom_config}')
  use_best_as_fixed = False
  
  hp = keras_tuner.HyperParameters()
  # Defines search space.
  
  if not use_best_as_fixed:
      hp.Float('learning_rate', 1e-4, 1e-3, sampling='log')
      hp.Float('weight_decay', 1e-4, 1e-2, sampling='log')
      hp.Float('drop_rate', min_value=0.1, max_value=0.3, default=0.5)
      hp.Float('log_q_correction_factor', min_value=0.1, max_value=1.0, default=0.5)
  else:
      hp.Fixed('learning_rate', 0.0001026)
      hp.Fixed('weight_decay', 0.00016785)
      hp.Fixed('drop_rate', 0.11754)
      #TODO: edit when have best value:
      hp.Fixed('log_q_correction_factor',value=0.5)

  #let AdamW weight decay handle the regularization, so set regl2 to 0:
  #hp.Float('regl2', 1e-5, 1e-2, sampling="log")
  hp.Fixed('regl2', 0.0)
  #layers_sizes is a list of ints, so encode each list as a string, choices can only be int,float,bool,str
  #the last layer in layer_sizes is the query and candidate embedding models' output dimensions-1
  #hp.Choice("layer_sizes", values=[json.dumps([16-1])], default=json.dumps([16-1]))
  hp.Fixed("layer_sizes", value=json.dumps([24 - 1]))
  #hp.Fixed("layer_sizes", value=json.dumps([46, 24 - 1]))
  #hp.Fixed("layer_sizes", value=json.dumps([64-1, 32-1]))
  # ahmos for "age", "hr_wk", "month", "occupation", "gender"
  hp.Fixed("feature_acronym", custom_config.get("feature_acronym", "h"))
  hp.Fixed("incl_genres", custom_config["incl_genres"])
  hp.Fixed('BATCH_SIZE', custom_config.get("BATCH_SIZE", DEFAULT_BATCH_SIZE))
  hp.Fixed('NUM_EPOCHS', custom_config.get("NUM_EPOCHS", DEFAULT_NUM_EPOCHS))
  #use_bias_corr = hp.Choice("use_bias_corr", values=[True, False], default=True)
  use_bias_corr = hp.Fixed("use_bias_corr", value=True)
  #if batch_size=1024, max temp should be about 0.2;  if batch_size is 2048, temp max ~ 0.4
  if use_bias_corr:
      if not use_best_as_fixed:
          hp.Choice("bias_corr_alpha", values=[0.01, 0.05, 0.1], default=0.05)
          hp.Float('temperature', 0.05, 0.15, step=0.01)
      else:
          hp.Fixed("bias_corr_alpha", 0.01)
          hp.Fixed('temperature', 0.1)
  else:
      hp.Choice("bias_corr_alpha", values=[0.1], default=0.05)  # 0.01, 0.05, 0.1
      hp.Fixed("temperature", value=1.0)
  hp.Fixed('n_users', value=custom_config["n_users"])
  hp.Fixed('n_movies', custom_config["n_movies"])
  hp.Fixed('n_genres', custom_config["n_genres"])
  hp.Fixed('run_eagerly', custom_config.get("run_eagerly", False))
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
def _get_strategy() -> Tuple[tf.distribute.Strategy, str]:
    if tf.config.list_physical_devices('TPU'):
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
                tpu='local')
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
            logging.info("Hardware auto-detected: TPU")
            return strategy, "TPU"
        except Exception as ex:
            logging.error(f"TPU detected but failed to initialize: {ex}")
            
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # MirroredStrategy handles both single-GPU and multi-GPU configurations automatically
            strategy = tf.distribute.MirroredStrategy()
            logging.info(f"Hardware auto-detected: {len(gpus)} GPU(s)")
            return strategy, "GPU"
        except Exception as ex:
            logging.error(f"GPU detected but strategy failed: {ex}")
    #NOTE a multihost strategy should use  tf.distribute.MultiWorkerMirroredStrategy
    # Fallback to default CPU strategy
    strategy = tf.distribute.get_strategy()
    logging.info("Hardware auto-detected: CPU fallback")
    return strategy, "CPU"

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

def create_input_shapes_from_spec(transformed_feature_spec : Dict[str, common_types.FeatureSpecType])\
        -> Dict[str, tf.TensorShape]:
    
    input_shapes = {}
    
    for name, spec in transformed_feature_spec.items():
        if isinstance(spec, tf.io.FixedLenFeature):
            # FixedLenFeature.shape describes the feature without the batch dimension.
            # We add [None] to indicate the batch size is variable.
            input_shapes[name] = tf.TensorShape([None] + list(spec.shape))
        
        elif isinstance(spec, tf.io.VarLenFeature):
            # VarLenFeature (Sparse) data usually has a dynamic shape.
            # [None, None] is the standard representation for a
            # variable-length sequence in a batch.
            input_shapes[name] = tf.TensorShape([None, None])
        
        elif isinstance(spec, tf.io.SparseFeature):
            # Less common, but handled similarly to VarLen
            input_shapes[name] = tf.TensorShape([None, None])
    
    return input_shapes

def get_stop_early_callback():
    # use patience=3 with batch_size 1024, and patience=5 with batch_size 2048
    # for val_ndcg_20 and batch_size=2056, min_delta should be 0.005 (random)
    # for val_mean_loss, min_delta=0.015 and patience=3
    # for cal_composite_ndcg_20
    # we use val_composite_ndcg_20 in order to better recommend items to the tail users
    # by including them in the NDCG score.
    # note that the val_composite_ndcg_20 peaks well before ndcg_20 and the other metrics,
    #  because those are maximized by popularity.
    return keras.callbacks.EarlyStopping(
        monitor=f'val_composite_ndcg_20', min_delta=0.0002, patience=3, mode="max",
        start_from_epoch=1,
        restore_best_weights=True)

@keras.utils.register_keras_serializable(package=package)
class MinimumThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_composite_ndcg_20',
            min_threshold:float = 2*0.005,
            start_epoch=1, patience=3):
        """
        Args:
            monitor: Metric key in validation logs.
            min_threshold: Minimum metric value expected.
            start_epoch: Warmup period (1-indexed) before pruning takes effect.
        """
        super().__init__()
        self.monitor = monitor
        self.min_threshold = min_threshold
        self.start_epoch = start_epoch
        self.patience = patience
        self.fail_count : int = 0
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_val = logs.get(self.monitor)
        
        if current_val is None or epoch < self.start_epoch:
            return
        
        # Check rule after warmup period
        if current_val > self.min_threshold:
            self.fail_count = 0
        else:
            self.fail_count += 1
            if self.fail_count >= self.patience:
                print(
                    f"\n[Pruned Trial] Epoch {epoch}: {self.monitor} ({current_val:.4f}) "
                    f"failed to meet threshold ({self.min_threshold:.4f}). Halting trial."
                )
                self.model.stop_training = True
                # Bayesian Optimization learns from this low score, but for another tuner,
                # one might want to abort the whole trial, and in that case, use:
                #   raise kt.errors.FailedTrialError("score is too low by epoch {epoch}...")
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "monitor": self.monitor,
            "min_threshold": self.min_threshold,
            "start_epoch": self.start_epoch,
            "patience" : self.patience,
        })
        return config


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
          
    fn_args.serving_model_dir: where the model will be saved to.
    NOTE that a hackish workaround to save the Query and Candidate embedding models
          separately for serving inference has been added to the run_fn in tune_train_movie_lens.py
          to create sibling directories called serving_query_model and serving_candidate_model.

    fn_args.hyperparameters (required) : keras_tuner.HyperParameters with keys
      'lr'
      "regl2"
      "drop_rate"
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
  logging.info("run_fn")
  for attr_name in dir(fn_args):
    # Filter out built-in methods and private attributes
    if not attr_name.startswith('__') and not callable(
      getattr(fn_args, attr_name)):
      attr_value = getattr(fn_args, attr_name)
      logging.debug(f"{attr_name}: {attr_value}")
  
  if fn_args.hyperparameters:
      logging.info("hp from fn_args.hyperparameters")
      hp = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
  elif fn_args.custom_config:
      logging.info("hp from custom_config")
      hp = get_default_hyperparameters(fn_args.custom_config)
  else:
      raise ValueError('hyperparameters must be provided')
  
  print('HyperParameters for training: %s' % hp.get_config())
  
  strategy, device = _get_strategy()
  
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
  _cand_keys = {'movie_id', 'genres'}
  input_signature_raw_candidate = {k: input_signature_raw[k] for k in _cand_keys}
  input_signature_raw_query = {k:v for k,v in input_signature_raw.items() if k not in _cand_keys}
  
  ## TODO: is input_dataset_element_spec_trans needed here since we have input_signature_trans?
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
    GLOBAL_BATCH_SIZE, is_train=True)
  
  eval_dataset = input_fn(
    fn_args.eval_files,
    fn_args.data_accessor,
    tf_transform_output,
    GLOBAL_BATCH_SIZE, is_train=False)
  
  #the model is built and compiled in strategy scope:
  logging.info("create 2Tower from run_fn")
  model = _make_2tower_keras_model(hp)
  # model = _make_2tower_keras_model(hp, tf_transform_output)

  # Write logs to path
  tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=fn_args.model_run_dir, update_freq='epoch')
  
  stop_early = get_stop_early_callback()
  
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
  logging.info("fit model")
  history = model.fit(
    train_dataset,
    steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
    validation_data=eval_dataset,
    validation_steps=EVAL_STEPS_PER_EPOCH,
    epochs=NUM_EPOCHS,
    callbacks=[tensorboard_callback, stop_early], verbose=1)
  
  print(f'fit history.history={history.history}')
  total_epochs_run = len(history.history['val_mean_loss'])
  print(f"total_epochs_run={total_epochs_run}", flush=True)
  logging.info(f"total_epochs_run={total_epochs_run}")
  
  #TODO: consider adding the vocabularies as assets:
  #    see https://www.tensorflow.org/api_docs/python/tf/saved_model/Asset
  
  """
  latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
  if latest_checkpoint:
    model.load_weights(latest_checkpoint)
    print(f"Loaded best weights from {latest_checkpoint}")
  """
  
  #create the query and candidate saved models
  from pathlib import Path
  path = Path(fn_args.serving_model_dir)
  serving_query_dir = str( path.parent / "serving_query_model")
  serving_candidate_dir = str( path.parent / "serving_candidate_model")
  
  build_input_shapes : Dict[str, tf.TensorShape] = create_input_shapes_from_spec(
      transformed_feature_spec = tf_transform_output.transformed_feature_spec())
  
  #create new query and candidate trained models without any parent references in thier variable or computation graphs:
  trained_query_weights = model.query_model.get_weights()
  trained_candidate_weights = model.candidate_model.get_weights()
  tf.keras.backend.clear_session()
  
  query_model = _make_query_model(n_users=model.n_users,
      layer_sizes=model.layer_sizes,
      regl2=model.regl2,
      drop_rate=model.drop_rate,
      feature_acronym=model.feature_acronym)
  query_model.build(input_shape=build_input_shapes)
  query_model.set_weights(trained_query_weights)
  
  print('QUERY MODEL SUMMARY:')
  query_model.summary()
  
  candidate_model = _make_candidate_model(n_movies=model.n_movies,
      movies_offset=model.movies_offset,
      n_genres=model.n_genres,
      layer_sizes=model.layer_sizes,
      regl2=model.regl2,
      drop_rate=model.drop_rate,
      incl_genres=model.incl_genres)
  candidate_model.build(input_shape=build_input_shapes)
  candidate_model.set_weights(trained_candidate_weights)
  
  print('CANDIDATE MODEL SUMMARY:')
  candidate_model.summary()
  
  tft_layer = tf_transform_output.transform_features_layer()
  raw_feature_spec = tf_transform_output.raw_feature_spec()
  
  def _parse_and_transform(raw_features, feature_spec, batch_size):
      complete_features = dict(raw_features)
      for key, spec in feature_spec.items():
          if key not in complete_features:
              if isinstance(spec, tf.io.FixedLenFeature):
                  shape = tf.concat([[batch_size], spec.shape], axis=0)
                  complete_features[key] = tf.zeros(shape=shape, dtype=spec.dtype)
              else:
                  complete_features[key] = tf.SparseTensor(
                      indices=tf.zeros([0, 2], dtype=tf.int64),
                      values=tf.zeros([0], dtype=spec.dtype),
                      dense_shape=tf.cast(tf.stack([batch_size, 0]), tf.int64))
      return tft_layer(complete_features)
  
  @tf.function
  def query_serve_fn(raw_features):
      batch_size = tf.shape(next(iter(raw_features.values())))[0]
      transformed_features = _parse_and_transform(raw_features, raw_feature_spec, batch_size)
      outputs = query_model(inputs=transformed_features, training=False)
      return {'outputs': outputs}
  
  @tf.function
  def candidate_serve_fn(raw_features):
      batch_size = tf.shape(next(iter(raw_features.values())))[0]
      transformed_features = _parse_and_transform(raw_features, raw_feature_spec, batch_size)
      outputs = candidate_model(inputs=transformed_features, training=False)
      return {'outputs': outputs}
  
  @tf.function
  def twotower_serve_dict_fn(raw_features):
      batch_size = tf.shape(next(iter(raw_features.values())))[0]
      transformed_features = _parse_and_transform(raw_features, raw_feature_spec, batch_size)
      outputs = model(inputs=transformed_features, training=False)
      return {'outputs': outputs}
  
  def get_relaxed_feature_spec(original_spec):
      """
      for the serialized example inputs for Candidate model or Query model, we want to be able to accept
      just the movie data and provide dummy values to fill the full joined user movie data.
      the dummy value is ignored by the candidate or query model
      :param original_spec:
      :return:
      """
      relaxed_spec = {}
      for key, spec in original_spec.items():
          if isinstance(spec, tf.io.FixedLenFeature):
              # We create a new spec with a default_value.
              # Use 0 for numbers and "" for strings.
              default = 0 if spec.dtype.is_integer or spec.dtype.is_floating else ""
              if key == 'timestamp':
                  default = -1
              relaxed_spec[key] = tf.io.FixedLenFeature(
                  shape=spec.shape,
                  dtype=spec.dtype,
                  default_value=default
              )
          else:
              # VarLenFeatures (SparseTensors) naturally handle missing keys
              # by returning an empty SparseTensor rather than crashing.
              relaxed_spec[key] = spec
      return relaxed_spec
  
  @tf.function
  def serve_query_tf_examples_fn(serialized_tf_example):
      '''
      Returns the serving signature query embeddings for input being raw examples, not yet transformed to features.
      '''
      relaxed_feature_spec = get_relaxed_feature_spec(raw_feature_spec)
      try:
          relaxed_feature_spec.pop(LABEL_KEY)
      except KeyError as e:
          logging.error(f'ERROR: {e}')
      raw_features = tf.io.parse_example(serialized_tf_example, relaxed_feature_spec)
      raw_timestamp = raw_features['timestamp']
      raw_features['timestamp'] = tf.where(
          tf.equal(raw_timestamp, -1),
          tf.cast(tf.timestamp(), tf.int64),
          raw_timestamp
      )
      transformed_features = tft_layer(raw_features)
      outputs = query_model(inputs=transformed_features, training=False)
      return {'outputs': outputs}
  
  @tf.function
  def serve_candidate_tf_examples_fn(serialized_tf_example):
      '''
      Returns the serving signature candidate embeddings for input being raw examples, not yet transformed to features.
      '''
      relaxed_feature_spec = get_relaxed_feature_spec(raw_feature_spec)
      try:
          relaxed_feature_spec.pop(LABEL_KEY)
      except KeyError as e:
          logging.error(f'ERROR: {e}')
      raw_features = tf.io.parse_example(serialized_tf_example,
          relaxed_feature_spec)
      transformed_features = tft_layer(raw_features)
      outputs = candidate_model(inputs=transformed_features, training=False)
      return {'outputs': outputs}
  
  def serve_twotower_tf_examples_fn(serialized_tf_example):
      '''Returns the serving signature for input being raw examples such as
      inputs = tf.data.TFRecordDataset(examples_file_paths, compression_type="GZIP")
      where examples_file_paths was written by MovieLensSplitExampleGen
      '''
      raw_feature_spec2 = raw_feature_spec.copy()
      try:
          raw_feature_spec2.pop(LABEL_KEY)
      except KeyError as e:
          logging.error(f'ERROR: {e}')
      raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec2)
      transformed_features = tft_layer(raw_features)
      outputs = model(inputs=transformed_features, training=False)
      return {'outputs': outputs}
  
  @tf.function
  def serve_query_tf_examples_fn(serialized_tf_example):
      '''
      Returns the serving signature query embeddings for input being raw examples, not yet transformed to features.
      '''
      relaxed_feature_spec = get_relaxed_feature_spec(raw_feature_spec)
      try:
          relaxed_feature_spec.pop(LABEL_KEY)
      except KeyError as e:
          logging.error(f'ERROR: {e}')
      raw_features = tf.io.parse_example(serialized_tf_example, relaxed_feature_spec)
      raw_timestamp = raw_features['timestamp']
      raw_features['timestamp'] = tf.where(
          tf.equal(raw_timestamp, -1),
          tf.cast(tf.timestamp(), tf.int64),
          raw_timestamp
      )
      transformed_features = tft_layer(raw_features)
      outputs = query_model(inputs=transformed_features, training=False)
      return {'outputs': outputs}
  
  @tf.function
  def serve_candidate_tf_examples_fn(serialized_tf_example):
      '''
      Returns the serving signature candidate embeddings for input being raw examples, not yet transformed to features.
      '''
      relaxed_feature_spec = get_relaxed_feature_spec(raw_feature_spec)
      try:
          relaxed_feature_spec.pop(LABEL_KEY)
      except KeyError as e:
          logging.error(f'ERROR: {e}')
      raw_features = tf.io.parse_example(serialized_tf_example, relaxed_feature_spec)
      transformed_features = tft_layer(raw_features)
      outputs = candidate_model(inputs=transformed_features, training=False)
      return {'outputs': outputs}
  
  @tf.function
  def transform_features_fn(serialized_tf_example):
      '''Returns the transformed_features to be fed as input to evaluator.  inputs are the raw
      examples from MovieLensSplitExampleGen
      '''
      raw_features = tf.io.parse_example(serialized_tf_example,raw_feature_spec)
      transformed_features = tft_layer(raw_features)
      return transformed_features
  
  ## ==== begin the export to QUERY saved model =======
  export_archive = keras.export.ExportArchive()
  export_archive.track(query_model)
  export_archive.track(tft_layer)

  export_archive.add_endpoint(
      name="serving_default",
      fn=query_serve_fn,
      input_signature=[input_signature_raw_query]
  )
  export_archive.add_endpoint(
      name="serving_default_examples",
      fn=serve_query_tf_examples_fn,
      input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')]
  )
  
  export_archive.write_out(serving_query_dir)
  
  logging.info(f"saved query model to {serving_query_dir}")
  
  ## ==== begin the export to CANDIDATE saved model =======
  export_archive = keras.export.ExportArchive()
  export_archive.track(candidate_model)
  export_archive.track(tft_layer)
  
  export_archive.add_endpoint(
      name="serving_default",
      fn=candidate_serve_fn,
      input_signature=[input_signature_raw_candidate]
  )
  export_archive.add_endpoint(
      name="serving_default_examples",
      fn=serve_candidate_tf_examples_fn,
      input_signature=[
          tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')]
  )
  export_archive.write_out(serving_candidate_dir)
  
  logging.info(f"saved candidate model to {serving_candidate_dir}")
  
  ## ==== begin the export to TWOTOWER all-in-one saved model =======
  export_archive = keras.export.ExportArchive()
  export_archive.track(model)
  export_archive.track(query_model)
  export_archive.track(candidate_model)
  export_archive.track(tft_layer)

  export_archive.add_endpoint(
      name="serving_default",
      fn=serve_twotower_tf_examples_fn,
      input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')]
  )
  export_archive.add_endpoint(
      name="serving_default_dict",
      fn=twotower_serve_dict_fn,
      input_signature=[input_signature_raw]
  )
  export_archive.add_endpoint(
      name="transform_features",
      fn=transform_features_fn,
      input_signature=[
          tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')]
  )
  export_archive.add_endpoint(
      name="serving_query",
      fn=serve_query_tf_examples_fn,
      input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')]
  )
  export_archive.add_endpoint(
      name="serving_query_dict",
      fn=query_serve_fn,
      input_signature=[input_signature_raw_query]
  )
  export_archive.add_endpoint(
      name="serving_candidate",
      fn=serve_candidate_tf_examples_fn,
      input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')]
  )
  export_archive.add_endpoint(
      name="serving_candidate_dict",
      fn=candidate_serve_fn,
      input_signature=[input_signature_raw_candidate]
  )
  
  export_archive.write_out(fn_args.serving_model_dir)
  
  logging.info(f"saved candidate model to {fn_args.serving_model_dir}")
  
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
    logging.info("hp from fn_args.hyperparameters")
    hp = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
  else:
    logging.info("hp from custom_config")
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
    
  strategy, device = _get_strategy()
  
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
      GLOBAL_BATCH_SIZE, is_train=True)
  
  eval_dataset = input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      transform_graph,
      GLOBAL_BATCH_SIZE, is_train=False)
  
  # the objective must be must be a name that appears in the logs
  # returned by the model.fit() method during training.
  #val_logs has keys 'val_loss' and 'val_compile_metrics'
  '''
  tuner = keras_tuner.RandomSearch(
    _make_2tower_keras_model,
    max_trials=10,
    executions_per_trial=1,
    overwrite=True,
    hyperparameters=hp,
    allow_new_entries=False,
    objective=keras_tuner.Objective(f'val_composite_ndcg_20', 'max'),
    directory=fn_args.working_dir,
    project_name='movie_lens_2t_tuning_r')
  '''
  
  '''
  tuner = keras_tuner.Hyperband(
    _make_2tower_keras_model,
    objective=keras_tuner.Objective(f'val_composite_ndcg_20', 'max'),
    max_epochs=8,
    factor=4,
    hyperband_iterations=3,
    overwrite=True,
    hyperparameters=hp,
    allow_new_entries=False,
    directory=fn_args.working_dir,
    project_name='movie_lens_2t_tuning_hb')
  '''
  
  #for max_trials=10, use alpha=1e-3, beta=2.3, num_initial_points=3
  #for max_trials=40 we use:
  #   num_initial_points = 13:
  #       allocates ~32% to random sampling.
  #       For a 5-dimensional search space, 13 points provide the Gaussian Process (GP)
  #       of the objective function with a solid, well-sampled baseline matrix
  #       (matrix of hyper-parm combinations) before it starts fitting its
  #       surrogate model.
  #    beta = 3.3:
  #        3.3 encourages higher exploration in the Upper Confidence Bound (UCB)
  #        acquisition function. we now have 27 guided trials remaining after the
  #        initialization phase, the algorithm has enough runway to explore
  #        distinct parameter pockets early on and still settle into exploiting
  #        the best regions before trial 40.
  #
  # alpha = 1e-3: Setting alpha = 1e-3 acts as a gentle noise-smoothing filter. Since validation metrics like ranking NDCG exhibit micro-variance between epochs, this prevents the GP from over-fitting to minor validation jitter.
  print("construct tuner BayesianOptimization")
  tuner = keras_tuner.BayesianOptimization(
      _make_2tower_keras_model,
      objective=keras_tuner.Objective('val_composite_ndcg_20', 'max'),
      hyperparameters=hp,
      alpha=1e-3,
      beta=3.3, #defaut 2.6;  4.0 for more exploration.
      num_initial_points=13, #30
      max_trials=40, #should be 2 to 3 times num_initial_points
      #TEMPORARY when fixing params:
      #num_initial_points=1, #30
      #max_trials=1, #should be 2 to 3 times num_initial_points
      allow_new_entries=False,
      directory=fn_args.working_dir,
      project_name='movie_lens_2t_tuning_bayesian')
  
  NUM_EPOCHS = hp.get("NUM_EPOCHS")
  
  stop_early = get_stop_early_callback()
  
  stop_threshold = MinimumThresholdCallback(
        monitor='val_composite_ndcg_20',
    )
  
  return tfx.components.TunerFnResult(
    tuner=tuner,
    fit_kwargs={
      'x': train_dataset,
      'validation_data': eval_dataset,
      'steps_per_epoch': TRAIN_STEPS_PER_EPOCH,
      'validation_steps': EVAL_STEPS_PER_EPOCH,
      'epochs' : NUM_EPOCHS,
      'callbacks' : [stop_early, stop_threshold],
    })
  