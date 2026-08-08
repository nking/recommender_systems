import math
import unittest

import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner
from tensorflow import TensorShape

from movie_lens_tfx.PipelineComponentsFactory import *
from movie_lens_tfx.tune_train_movie_lens import _make_2tower_keras_model, \
    get_default_hyperparameters, _make_candidate_model, _make_query_model, \
    create_input_shapes_from_spec

from helper import *

"""
this class is for use in debugging the model
"""
tf.config.run_functions_eagerly(True)

tf.get_logger().propagate = False
from absl import logging
logging.set_verbosity(logging.INFO)
logging.set_stderrthreshold(logging.INFO)
import warnings
# Suppress DeprecationWarnings from 3rd-party libraries
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*parse_example_dataset.*")

class EagerModelTrainTest(unittest.TestCase):

  def setUp(self):
      super().setUp()
      self.n_users = 6040
      self.n_movies = 3952
      self.n_genres = 18
      self.n_occupations = 21
      self.num_examples = 100
      self.MIN_EVAL_SIZE = 50
      self.name = 'test run for table_b'
      self.BATCH_SIZE = 50
      self.num_epochs = 1
    
  def test_model(self):
      
      #datasets are 100 examples
      
      custom_config = {
            'n_users': self.n_users,
            'n_movies': self.n_movies,
            'n_genres': self.n_genres,
            'feature_acronym': "ahosy",
            'run_eagerly': True,
            'incl_genres': True,
            'BATCH_SIZE': self.BATCH_SIZE,
            "NUM_EPOCHS": self.num_epochs,
            "num_examples": self.num_examples,
            "version": "0.001",
            "model_name": MODEL_NAME.USER_MOVIE.value,
            "input_dataset_element_spec_trans_ser" : "gASV/gIAAAAAAAB9lCiMA2FnZZSMInRlbnNvcmZsb3cucHl0aG9uLmZy"
                                                     "YW1ld29yay50ZW5zb3KUjApUZW5zb3JTcGVjlJOUjCh0ZW5zb3JmbG93L"
                                                     "nB5dGhvbi5mcmFtZXdvcmsudGVuc29yX3NoYXBllIwLVGVuc29yU2hhcG"
                                                     "WUk5RdlChoBYwJRGltZW5zaW9ulJOUToWUUpRoCksBhZRSlGWFlFKUjCJ"
                                                     "0ZW5zb3JmbG93LnB5dGhvbi5mcmFtZXdvcmsuZHR5cGVzlIwIYXNfZHR"
                                                     "5cGWUk5SMB2Zsb2F0MzKUhZRSlGgBh5RSlIwGZ2VuZGVylGgEaAddlChoCk"
                                                     "6FlFKUaApLAYWUUpRlhZRSlGgWaBmHlFKUjAZnZW5yZXOUaARoB12UKGgK"
                                                     "ToWUUpRoCksBhZRSlGgKSxKFlFKUZYWUUpRoFmgjh5RSlIwCaHKUaARoB12"
                                                     "UKGgKToWUUpRoCksBhZRSlGWFlFKUaBZoL4eUUpSMBWhyX3drlGgEaAddlCh"
                                                     "oCk6FlFKUaApLAYWUUpRlhZRSlGgWaDmHlFKUjAVtb250aJRoBGgHXZQoaApO"
                                                     "hZRSlGgKSwGFlFKUZYWUUpRoFmhDh5RSlIwIbW92aWVfaWSUaARoB12UKGgKT"
                                                     "oWUUpRoCksBhZRSlGWFlFKUaBZoTYeUUpSMCm9jY3VwYXRpb26UaARoB12UKGg"
                                                     "KToWUUpRoCksBhZRSlGWFlFKUaBZoV4eUUpSMC3NlY19pbnRvX3lylGgEaAdd"
                                                     "lChoCk6FlFKUaApLAYWUUpRlhZRSlGgWaGGHlFKUjAd1c2VyX2lklGgEaAddlC"
                                                     "hoCk6FlFKUaApLAYWUUpRlhZRSlGgWaGuHlFKUjAd3ZWVrZGF5lGgEaAddlCho"
                                                     "Ck6FlFKUaApLAYWUUpRlhZRSlGgWaHWHlFKUjAJ5cpRoBGgHXZQoaApOhZRSlGg"
                                                     "KSwGFlFKUZYWUUpRoFmh/h5RSlIwEeXJfepRoBGgHXZQoaApOhZRSlGgKSwGFlF"
                                                     "KUZYWUUpRoFmiJh5RSlHUu"
        }
      hp : keras_tuner.HyperParameters = get_default_hyperparameters(custom_config)
      hp.Fixed('input_dataset_element_spec_trans_ser', custom_config["input_dataset_element_spec_trans_ser"])
      
      model = _make_2tower_keras_model(hp)
      
      BATCH_SIZE_PER_REPLICA = self.BATCH_SIZE
      NUM_EPOCHS = self.num_epochs
      n_replicas = 1
      GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * n_replicas
      
      # virtual epochs:
      TRAIN_STEPS_PER_EPOCH = math.ceil(self.num_examples / GLOBAL_BATCH_SIZE)
      EVAL_STEPS_PER_EPOCH = math.ceil(self.num_examples / GLOBAL_BATCH_SIZE)
      
      train_path = os.path.join(get_project_dir(), 'src/test/resources/ml-1m/transformed_features_train.parquet')
      eval_path = os.path.join(get_project_dir(),
          'src/test/resources/ml-1m/transformed_features_val.parquet')
      train_dataset = load_parquet_as_tf_dataset(train_path)
      eval_dataset = load_parquet_as_tf_dataset(eval_path)
      
      history = model.fit(
          train_dataset,
          steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
          validation_data=eval_dataset,
          validation_steps=EVAL_STEPS_PER_EPOCH,
          epochs=NUM_EPOCHS,
          callbacks=[], verbose=1)
      
      print(f'fit history.history={history.history}')
      
      # create new query and candidate trained models without any parent references in thier variable or computation graphs:
      trained_query_weights = model.query_model.get_weights()
      trained_candidate_weights = model.candidate_model.get_weights()
      tf.keras.backend.clear_session()
      
      # DEBUG
      import numpy as np
      tf.print("trained_query_weights  len=:", len(trained_query_weights))
      for i, layer in enumerate(trained_query_weights):
          tf.print(i, " shape=", np.shape(layer))
      tf.print("trained_candidate_weights  len=:",
          len(trained_candidate_weights))
      for i, layer in enumerate(trained_candidate_weights):
          tf.print(i, " shape=", np.shape(layer))
      
      build_input_shapes = {'age': TensorShape([1, 1]),
          'gender': TensorShape([1, 1]), 'genres': TensorShape([1, 1, 18]),
          'hr': TensorShape([1, 1]), 'hr_wk': TensorShape([1, 1]),
          'month': TensorShape([1, 1]), 'movie_id': TensorShape([1, 1]),
          'occupation': TensorShape([1, 1]),
          'sec_into_yr': TensorShape([1, 1]),
          'user_id': TensorShape([1, 1]), 'weekday': TensorShape([1, 1]),
          'yr': TensorShape([1, 1]), 'yr_z': TensorShape([1, 1])}
      
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
      
def decode_genres(encoded_str):
    if isinstance(encoded_str, bytes):
        encoded_str = encoded_str.decode("utf-8")
    # Decode base64 -> bytes -> unpickle to original object (list or numpy array)
    return pickle.loads(base64.b64decode(encoded_str))

def load_parquet_as_tf_dataset(file_path: str, batch_size: int = 32,
        shuffle: bool = True):
    """
    Reads Parquet files matching a path pattern and returns a tf.data.Dataset.
    """
    
    df = pd.read_parquet(file_path)
    label = df['rating']
    df.drop(columns=['rating'], inplace=True)
    
    deserialized_genres = [decode_genres(x) for x in df["genres"]]
    genres_matrix = np.array(deserialized_genres, dtype=np.float32)
    
    if genres_matrix.ndim == 2 and genres_matrix.shape[1] == 18:
        genres_matrix = genres_matrix.reshape(-1, 1, 18)
        
    expected_sample_shape = (1, 18)
    actual_sample_shape = genres_matrix.shape[1:]
    assert (actual_sample_shape == expected_sample_shape
    ), f"Assertion Error: 'genres' sample shape must be {expected_sample_shape}, but got {actual_sample_shape}"
    
    float_cols = [col for col in df.columns if col != "genres"]
    feature_dict = {}
    for col in float_cols:
        col_array = df[col].to_numpy().astype("float32")
        col_array = col_array.reshape(-1, 1)
        feature_dict[col] = col_array
    feature_dict["genres"] = genres_matrix
    
    dataset = tf.data.Dataset.from_tensor_slices((feature_dict, label))
    
    dataset = dataset.shuffle(buffer_size=100).batch(50).prefetch(tf.data.AUTOTUNE)
    
    return dataset
      