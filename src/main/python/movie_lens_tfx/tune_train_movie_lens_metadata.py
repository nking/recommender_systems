"""train a movie_genres to movie_id metadata model"""
import base64
import pickle
# some code is adapted from https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils_base.py
# and related files
# they have co Copyright 2020 Google LLC. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
from typing import List, Tuple, Dict
import tensorflow as tf
import tensorflow.keras as keras
#import tf_keras as keras ## this fails
import enum
import os
import math
import json
import keras_tuner
import tensorflow_transform as tft
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

DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_EPOCHS = 20
DEFAULT_NUM_EXAMPLES = 100000

LOSS_FN = keras.losses.MeanSquaredError() #name=mean_squared_error
METRIC_FN = keras.metrics.RootMeanSquaredError()

MAX_TUNE_TRIALS_DEFAULT = 10

#NOTE: could be improved by writing the headers to a file in the Transform stage and reading them here:
FEATURE_KEYS = [
   'movie_id', 'genres'
]
LABEL_KEY = 'rating'
N_GENRES = 18

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


def _make_movie_metadata_model(hp: keras_tuner.HyperParameters) -> tf.keras.Model:
  # TODO: consider change to read from the transformed schema
  
  input_dataset_element_spec_ser = hp.get("input_dataset_element_spec_ser")
  input_dataset_element_spec = pickle.loads(base64.b64decode(input_dataset_element_spec_ser.encode('utf-8')))
  logging.debug(f'input_dataset_element_spec={input_dataset_element_spec}')
  # NOTE: tfx expected the models to subclass tf.keras.Model, not keras.Model
  
  _input_dataset_element_spec = {}
  for key, value in input_dataset_element_spec.items():
    _shape = [i for i in value.shape]
    _shape[0] = None
    _input_dataset_element_spec[key] = tf.TensorSpec(shape=_shape, dtype=value.dtype, name=key)
  input_dataset_element_spec = _input_dataset_element_spec
  
  # TODO: add hyper-parameter "temperature" after L2Norm
  @keras.utils.register_keras_serializable(package=package)
  class QueryModel(keras.Model):
    """Model for encoding user queries composed of multihot genres."""
    
    # for init from a load, arguments are present for the compositional instance members too
    def __init__(self, layer_sizes: list, n_genres: int=18,
                 embed_out_dim: int = 32,
                 reg: keras.regularizers.Regularizer = None,
                 drop_rate: float = 0., **kwargs):
      """Model for encoding genres queries.

      Args:
        layer_sizes:
          A list of integers where the i-th entry represents the number of units
          the i-th layer contains.
        n_genres:
        embed_out_dim:
        reg:
        drop_rate
      """
      super(QueryModel, self).__init__(**kwargs)
      
      self.embedding_model = keras.Sequential([
        keras.layers.Dense(embed_out_dim),
        keras.layers.Flatten(data_format='channels_last'),
      ], name="genres_emb")
      
      if isinstance(layer_sizes, str):
        layer_sizes = json.loads(layer_sizes)
      
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
      
      self.n_genres = n_genres
      self.embed_out_dim = embed_out_dim
      self.layer_sizes = layer_sizes
      self.drop_rate = drop_rate
    
    def build(self, input_shape):
      # print(f'build {self.name} input_shape={input_shape}\n')
      self.embedding_model.build(input_shape['genres'])
      input_shape_2 = self.embedding_model.compute_output_shape(input_shape['genres'])
      self.dense_layers.build(input_shape_2)
      self.built = True
    
    def compute_output_shape(self, input_shape):
      # print(f'compute_output_shape {self.name} input_shape={input_shape}, {input_shape['user_id'][0]}, {self.layer_sizes[-1:]}\n')
      # This is invoked after build by TwoTower
      # return self.output_shapes[0]
      input_shape_3 = self.dense_layers.compute_output_shape(
        self.embedding_model.compute_output_shape(input_shape['genres']))
      _shape_3 = [i for i in input_shape_3]
      _shape_3[0] = None
      return _shape_3
      # return None, self.layer_sizes[-1]
      # return (input_shape['user_id'][0], self.layer_sizes[-1])
    
    def call(self, inputs, **kwargs):
      # inputs should contain columns:
      # print(f'call {self.name} type={type(inputs)}\n')
      feature_embedding = self.embedding_model(inputs['genres'], **kwargs)
      res = self.dense_layers(feature_embedding)
      #tf.print('CALL', self.name, ' shape=', res.shape)
      return res
    
    def get_config(self):
      config = super(QueryModel, self).get_config()
      config.update({"n_genres": self.n_genres,
                     "embed_out_dim": self.embed_out_dim,
                     "layer_sizes": self.layer_sizes,
                     "reg": keras.utils.serialize_keras_object(
                       self.reg),
                     })
      return config
    
    @classmethod
    def from_config(cls, config):
      for key in ["reg"]:
        config[key] = keras.utils.deserialize_keras_object(config[key])
      return cls(**config)
  
  # TODO: add hyper-parameter "temperature" after L2Norm
  @keras.utils.register_keras_serializable(package=package)
  class CandidateModel(keras.Model):
    """Model for encoding candidate features, that is, movie_ids."""
    
    # for init from a load, arguments are present for the compositional instance members too
    def __init__(self, n_movies: int, layer_sizes,
                 embed_out_dim: int = 32,
                 reg: keras.regularizers.Regularizer = None,
                 drop_rate: float = 0., **kwargs):
      """Model for encoding candidate features.

      Args:
        layer_sizes:
          A list of integers where the i-th entry represents the number of units
          the i-th layer contains.
      """
      super(CandidateModel, self).__init__(**kwargs)
      
      self.embedding_model = keras.Sequential([
        keras.layers.Embedding(n_movies + 1, embed_out_dim),
        keras.layers.Flatten(data_format='channels_last'),
      ], name="movie_emb")
      
      self.dense_layers = keras.Sequential(name="dense_candidate")
      if isinstance(layer_sizes, str):
        layer_sizes = json.loads(layer_sizes)
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
      self.embed_out_dim = embed_out_dim
      self.drop_rate = drop_rate
      self.layer_sizes = layer_sizes
      
    def build(self, input_shape):
      # print(f'build {self.name} input_shape={input_shape}\n')
      self.embedding_model.build(input_shape['movie_id'])
      input_shape_2 = self.embedding_model.compute_output_shape(input_shape['movie_id'])
      self.dense_layers.build(input_shape_2)
      self.built = True
    
    def compute_output_shape(self, input_shape):
      # print(f'compute_output_shape {self.name} input_shape={input_shape}\n')
      # This is invoked after build by TwoTower
      input_shape_3 = self.dense_layers.compute_output_shape(
        self.embedding_model.compute_output_shape(input_shape['movie_id']))
      _shape_3 = [i for i in input_shape_3]
      _shape_3[0] = None
      return _shape_3
      # return None, self.layer_sizes[-1]
    
    def call(self, inputs, **kwargs):
      # inputs should contain columns "movie_id", "genres"
      # logging.debug(f'call {self.name} type ={type(inputs)}\ntype ={inputs}\n')
      feature_embedding = self.embedding_model(inputs['movie_id'], **kwargs)
      #tf.print('invoked movie_emb.  shape=', feature_embedding.shape)
      res = self.dense_layers(feature_embedding)
      # returns an np.ndarray wrapped in a tensor if inputs is tensor, else not wrapped
      # logging.debug(f'CALL {self.name} SHAPE ={res.shape}\n')
      #tf.print('CALL', self.name, ' shape=', res.shape)
      return res
    
    def get_config(self):
      config = super(CandidateModel, self).get_config()
      config.update(
        {"n_movies": self.n_movies,
         "embed_out_dim": self.embed_out_dim,
         "drop_rate": self.drop_rate,
         "layer_sizes": self.layer_sizes,
         "reg": keras.utils.serialize_keras_object(self.reg),
         })
      return config
    
    @classmethod
    def from_config(cls, config):
      for key in ["reg"]:
        config[key] = keras.utils.deserialize_keras_object(config[key])
      return cls(**config)
  
  @keras.utils.register_keras_serializable(package=package)
  class MetadataDNN(keras.Model):
    """
    a T2-Tower DNN model that accepts input containing: 'genres', 'movie_id' and 'rating' where 'rating' is the label.

    a sigmoid is used to provide logistic regression predictions of the rating.

    the number of layers is controlled by a list of their sizes in layer_sizes.
    """
    
    # for init from a load, arguments are present for the compositional instance members too
    def __init__(self,  n_movies: int, n_genres: int,
                 layer_sizes: list, embed_out_dim: int,
                 reg: keras.regularizers.Regularizer = None,
                 drop_rate: float = 0, **kwargs):
      super(MetadataDNN, self).__init__(**kwargs)
      
      if isinstance(layer_sizes, str):
        layer_sizes = json.loads(layer_sizes)
      
      self.query_model = QueryModel(layer_sizes=layer_sizes, n_genres=n_genres,
                                    embed_out_dim=embed_out_dim,
                                    reg=reg,
                                    drop_rate=drop_rate,
                                    **kwargs)
      
      self.candidate_model = CandidateModel(n_movies=n_movies,
                                            layer_sizes=layer_sizes, embed_out_dim=embed_out_dim,
                                            reg=reg,
                                            drop_rate=drop_rate,
                                            **kwargs)
      
      # elementwise multiplication:
      self.dot_layer = keras.layers.Dot(axes=1)
      self.sigmoid_layer = keras.layers.Activation(keras.activations.sigmoid)
      
      self.reg = reg
      
      self.n_movies = n_movies
      self.n_genres = n_genres
      self.layer_sizes = layer_sizes
      self.embed_out_dim = embed_out_dim
      self.drop_rate = drop_rate
    
    @tf.function(input_signature=[input_dataset_element_spec])
    def call(self, inputs):
      #logging.debug(f'call {self.name} inputs={inputs}\n')
      user_vector = self.query_model(inputs)
      movie_vector = self.candidate_model(inputs)
      #tf.print('U,V SHAPES: ', user_vector.shape, movie_vector.shape)
      s = self.dot_layer([user_vector, movie_vector])
      s = self.sigmoid_layer(s)
      #tf.print('CALL', self.name, ' shape=', s.shape, ' type=', type(s))
      return s
      
    @tf.function(input_signature=[input_dataset_element_spec])
    def serve_query_model(self, inputs):
      """A dedicated function to trace and serve the trained Query Model."""
      return self.query_model(inputs)  #
    
    @tf.function(input_signature=[input_dataset_element_spec])
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
      self.sigmoid_layer.build(s2)
      self.built = True
    
    def compute_output_shape(self, input_shape):
      # (batch_size,)  a scalar for each row in batch
      # return input_shape['user_id']
      s0 = self.query_model.compute_output_shape(input_shape)
      s1 = self.candidate_model.compute_output_shape(input_shape)
      s2 = self.dot_layer.compute_output_shape([s0, s1])
      s3 = self.sigmoid_layer.compute_output_shape(s2)
      _shape_3 = [i for i in s3]
      _shape_3[0] = None
      return _shape_3
      # return (None,)
    
    def train_step(self, batch, **kwargs):
      x, y = batch
      with tf.GradientTape() as tape:
        y_pred = self.call(x)  # Forward pass
        loss = self.compute_loss(y=y, y_pred=y_pred)
      
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
      y_pred = self(x, training=False) #self.predict or self.evaluate?
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
      config = super(MetadataDNN, self).get_config()
      config.update({"n_movies": self.n_movies,
                     "n_genres": self.n_genres,
                     "embed_out_dim": self.embed_out_dim,
                     "drop_rate": self.drop_rate,
                     "layer_sizes": self.layer_sizes,
                     "reg": keras.utils.serialize_keras_object(self.reg),
                     })
      return config
    
    @classmethod
    def from_config(cls, config):
      for key in ["reg"]:
        config[key] = keras.utils.deserialize_keras_object(config[key])
      return cls(**config)
  
  # use strategy
  d = hp.get("device")
  if d == "GPU":
    device = Device.GPU
  elif d == "TPU":
    device = Device.TPU
  else:
    device = Device.CPU
  strategy, device = _get_strategy(device)
  
  with strategy.scope():
    model = MetadataDNN(
      n_movies=hp.get("movie_id_max") + 1,
      n_genres=hp.get("n_genres"),
      layer_sizes=hp.get('layer_sizes'),
      embed_out_dim=hp.get('embed_out_dim'),
      reg=None, drop_rate=0.1,
    )
  
    input_shapes = {}
    # input_shapes[element] = (batch_size,)
    for element in FEATURE_KEYS:
      if element == "genres":
        input_shapes[element] = (None, 1, N_GENRES)
      else:
        input_shapes[element] = (None, 1)
    
    model.build(input_shapes)
    
    optimizer = keras.optimizers.Adam(learning_rate=hp.get('learning_rate'))
    
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


def get_default_hyperparameters(custom_config, input_element_spec) -> keras_tuner.HyperParameters:
  """Returns hyperparameters for building Keras model."""
  #print(f'get_default_hyperparameters: custom_config={custom_config}')
  hp = keras_tuner.HyperParameters()
  # Defines search space.
  hp.Choice('learning_rate', [1e-4], default=1e-4)
  hp.Choice("regl2", values=[0.0, 0.001, 0.01], default=None)
  hp.Float("drop_rate", min_value=0.1, max_value=0.5, default=0.5)
  hp.Choice("embed_out_dim", values=[32], default=32)
  #layers_sizes is a list of ints, so encode each list as a string, chices can only be int,float,bool,str
  hp.Choice("layer_sizes", values=[json.dumps([32])], default=json.dumps([32]))
  # ahmos for "age", "hr_wk", "month", "occupation", "gender"
  hp.Fixed('BATCH_SIZE', custom_config.get("BATCH_SIZE", DEFAULT_BATCH_SIZE))
  hp.Fixed('NUM_EPOCHS', custom_config.get("NUM_EPOCHS", DEFAULT_NUM_EPOCHS))
  hp.Fixed('movie_id_max', custom_config["movie_id_max"])
  hp.Fixed('n_genres', custom_config["n_genres"])
  hp.Fixed('run_eagerly', custom_config["run_eagerly"])
  hp.Fixed('device', custom_config.get("device", 'CPU'))
  hp.Fixed('MAX_TUNE_TRIALS', custom_config.get("MAX_TUNE_TRIALS", MAX_TUNE_TRIALS_DEFAULT))
  hp.Fixed('input_dataset_element_spec_ser', (base64.b64encode(pickle.dumps(input_element_spec))).decode('utf-8'))
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
          'user_id_max'
          'movie_id_max'
          'n_genres'
          'run_eagerly'
          'device'

    fn_args.hyperparameters (required) : keras_tuner.HyperParameters with keys
      'lr'
      "regl2"
      "drop_rate"
      "embed_out_dim"
      "layer_sizes"
      'num_epochs'
      'batch_size'
      'movie_id_max'
      'n_genres'
      'run_eagerly'
      'device'
  """
  logging.debug(f"run_fn fn_args type={type(fn_args)}")
  # not sure if outputs query_model and candidate_model are passed to this.
  for attr_name in dir(fn_args):
    # Filter out built-in methods and private attributes
    if not attr_name.startswith('__') and not callable(
      getattr(fn_args, attr_name)):
      attr_value = getattr(fn_args, attr_name)
      logging.debug(f"{attr_name}: {attr_value}")
  
  if not fn_args.hyperparameters:
    raise ValueError('hyperparameters must be provided')
  
  transform_graph = tft.TFTransformOutput(fn_args.transform_graph_path)
  
  if fn_args.hyperparameters:
    hp = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
  else:
    tmp_dataset = input_fn(fn_args.train_files, fn_args.data_accessor,
                           transform_graph, DEFAULT_BATCH_SIZE)
    
    def extract_features(features, label):
      return features
    
    x = tmp_dataset.map(extract_features)
    hp = get_default_hyperparameters(fn_args.custom_config, x.element_spec)
  
  logging.info('HyperParameters for training: %s' % hp.get_config())
  
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
  
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
  
  input_signature_raw = convert_feature_spec_to_tensor_spec(tf_transform_output.raw_feature_spec())
  del input_signature_raw[LABEL_KEY]
  logging.debug(f"input_signature_raw={input_signature_raw}")
  
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
  model = _make_movie_metadata_model(hp)

  # Write logs to path
  tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=fn_args.model_run_dir, update_freq='epoch')
  
  stop_early = keras.callbacks.EarlyStopping(
    monitor=f'val_loss', min_delta=1E-4, patience=3)
  
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
  
  input_element_spec = train_dataset.element_spec[0]
  
  call_sig = model.call.get_concrete_function(
    input_element_spec
  )
  
  query_sig = model.serve_query_model.get_concrete_function(
    input_element_spec
  )
  
  candidate_sig = model.serve_candidate_model.get_concrete_function(
    input_element_spec
  )
  
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
      where examples_file_paths was written by MovieLensExampleGen
      '''
      raw_feature_spec = tf_transform_output.raw_feature_spec()
      try:
        raw_feature_spec.pop(LABEL_KEY)
      except KeyError as e:
        logging.error(f'ERROR: {e}')
      logging.debug(f'default_serving: LABEL_KEY={LABEL_KEY}, raw_feature_spec={raw_feature_spec}')
      
      raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
      
      transformed_features = model.tft_layer(raw_features)
      outputs = model(inputs=transformed_features)
      logging.debug(f'default_serving: have outputs')
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
      outputs = model.query_model(inputs=transformed_features)
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
      outputs = model.candidate_model(inputs=transformed_features)
      return {'outputs': outputs}
    
    @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
      '''Returns the transformed_features to be fed as input to evaluator.  inputs are the raw
      examples from MovieLensExampleGen
      '''
      raw_feature_spec = tf_transform_output.raw_feature_spec()
      print('transform_features_fn spec = {raw_feature_spec}')
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
      "serving_query_dict": serve_query_dict_fn,
      "serving_candidate_dict": serve_candidate_dict_fn
    }
  
  signatures = {
    'serving_twotower_transformed': call_sig,
    'serving_query_transformed': query_sig,
    'serving_candidate_transformed': candidate_sig,
  }
  other_sigs = _make_raw_serving_signatures(model, tf_transform_output)
  signatures["transform_features"] = other_sigs["transform_features"]
  signatures["serving_default"] = other_sigs["serving_default"]
  signatures["serving_query"] = other_sigs["serving_query"]
  signatures["serving_candidate"] = other_sigs["serving_candidate"]
  signatures["serving_default_dict"] = other_sigs["serving_default_dict"]
  signatures["serving_query_dict"] = other_sigs["serving_query_dict"]
  signatures["serving_candidate_dict"] = other_sigs["serving_candidate_dict"]
  
  """
  #this isn't necessary.  BulkInferrer still has the same problems finding saved model variables
  import numpy as np
  serialized_batch = tf.constant(np.array(
    [b'\n\xa2\x01\n\x1d\n\x06genres\x12\x13\n\x11\n\x0fAction|Thriller\n\x0f\n\x06gender\x12\x05\n\x03\n\x01M\n\x0c\n\x03age\x12\x05\x1a\x03\n\x01-\n\x0f\n\x06rating\x12\x05\x1a\x03\n\x01\x04\n\x12\n\x08movie_id\x12\x06\x1a\x04\n\x02\x8c\x08\n\x16\n\ttimestamp\x12\t\x1a\x07\n\x05\x8a\xac\xbe\xd2\x03\n\x13\n\noccupation\x12\x05\x1a\x03\n\x01\x07\n\x10\n\x07user_id\x12\x05\x1a\x03\n\x01\x04',
     b'\n\xa9\x01\n\x0c\n\x03age\x12\x05\x1a\x03\n\x01-\n\x16\n\ttimestamp\x12\t\x1a\x07\n\x05\xf4\xab\xbe\xd2\x03\n$\n\x06genres\x12\x1a\n\x18\n\x16Action|Sci-Fi|Thriller\n\x0f\n\x06rating\x12\x05\x1a\x03\n\x01\x05\n\x0f\n\x06gender\x12\x05\n\x03\n\x01M\n\x13\n\noccupation\x12\x05\x1a\x03\n\x01\x07\n\x10\n\x07user_id\x12\x05\x1a\x03\n\x01\x04\n\x12\n\x08movie_id\x12\x06\x1a\x04\n\x02\xd8\t']))
  try:
    # Call the serving function with dummy data forces the graph to trace and initialize all variables
    # within the context of the signature.  NLK: The fit function creates trace only for training, not serving.
    print("Forcing model tracing with dummy data...")
    INPUT_KEY = "serialized_tf_example"# list(signatures["serving_default"].structured_input_signature[1].keys())[0]
    _ = signatures["serving_default"](**{INPUT_KEY: serialized_batch})
    print(f"Tracing complete. Variables should be initialized.  outputs={_}")
  except Exception as e:
    print(f"Warning: Failed to trace with dummy data. Error: {e}")
  """
  
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
  
  #FnArgs should be from tfx.components.trainer.fn_args_utils
  logging.debug(f"run_fn fn_args type={type(fn_args)}")
  logging.debug(f"Working directory: {fn_args.working_dir}")
  logging.debug(f"Training files: {fn_args.train_files}")
  logging.debug(f"Evaluation files: {fn_args.eval_files}")
  logging.debug(f"Transform graph path: {fn_args.transform_graph_path}")
  logging.debug(f"data_accessor: {fn_args.data_accessor}")
  logging.debug(f"Hyperparameters: {fn_args.hyperparameters}")
  logging.debug(f"Custom config: {fn_args.custom_config}")
  
  transform_graph = tft.TFTransformOutput(fn_args.transform_graph_path)
  
  if fn_args.hyperparameters:
    hp = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
  else:
    tmp_dataset = input_fn(fn_args.train_files,fn_args.data_accessor,
      transform_graph, DEFAULT_BATCH_SIZE)
    def extract_features(features, label):
      return features
    x = tmp_dataset.map(extract_features)
    hp = get_default_hyperparameters(fn_args.custom_config, x.element_spec)
    
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
  tuner = keras_tuner.RandomSearch(
    _make_movie_metadata_model,
    max_trials=hp.get("MAX_TUNE_TRIALS"),
    hyperparameters=hp,
    allow_new_entries=False,
    objective=keras_tuner.Objective(f'val_loss', 'min'),
    # objective=keras_tuner.Objective('val_loss', 'min'),
    directory=fn_args.working_dir,
    project_name='movie_lens_metadata_tuning')
  
  return tfx.components.TunerFnResult(
    tuner=tuner,
    fit_kwargs={
      'x': train_dataset,
      'validation_data': eval_dataset,
      'steps_per_epoch': TRAIN_STEPS_PER_EPOCH,
      'validation_steps': EVAL_STEPS_PER_EPOCH
    })
