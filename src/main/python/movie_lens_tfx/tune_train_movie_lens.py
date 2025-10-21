#from

#some code is adapted from https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils_base.py
# and related files
# they have co Copyright 2020 Google LLC. All Rights Reserved.
#Licensed under the Apache License, Version 2.0 (the "License");
from typing import Dict, Text, Any, List, Tuple
import enum
import os
import keras_tuner
import tensorflow as tf
import tf_keras as keras
import tensorflow_transform as tft
from tensorboard.plugins.hparams.api import hparams
from tensorflow.python.ops.gen_experimental_dataset_ops import \
  save_dataset
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

#https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/examples/penguin/penguin_utils_base.py#L98
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
      tf_transform_output.transformed_metadata.schema).repeat().prefetch(tf.data.AUTOTUNE)

def _make_2tower_keras_model(hp: keras_tuner.HyperParameters) -> tf.keras.Model:
  #TODO: consider change to read from the transformed schema
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

  optimizer = keras.optimizers.Adam(learning_rate=hp.get('learning_rate'))

  #LOSS:
  #can use Ordinal Logistic Regression for classification into ratings categories
  #or MSE: when loss=MSE, choose RMSE as eval metric, but they are affected by outliers
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

def get_default_hyperparameters(custom_config: tfx.components.FnArgs.custom_config)-> keras_tuner.HyperParameters:
  """Returns hyperparameters for building Keras model."""
  hp = keras_tuner.HyperParameters()
  # Defines search space.
  hp.Choice('lr', [1e-4], default=1e-4)
  hp.Choice("regl2", values=[None, 0.001, 0.01], default=None)
  hp.Float("drop_rate", min_value=0.1, max_value=0.5, default=0.5)
  hp.Choice("embed_out_dim", values=[32], default=32)
  hp.Choice("layer_sizes", values=[[32]], default=[32])
  #ahos for "Age_group", "hrs_bin", "occupation", "gender"
  hp.Fixed("feature_acronym", value="h")
  hp.Boolean("incl_genres", default=True)
  hp.Fixed('num_epochs', value=10)
  hp.Fixed('batch_size', TRAIN_BATCH_SIZE)
  hp.Boolean("use_bias_corr", default=False)
  hp.Fixed('user_id_max', value=custom_config["user_id_max"])
  hp.Fixed('movie_id_max', custom_config["movie_id_max"])
  hp.Fixed('n_genres',  custom_config["n_genres"])
  hp.Fixed('run_eagerly', False)
  return hp

# TFX Trainer will call this function.
def _get_strategy(device : Device) -> Tuple[tf.distribute.Strategy, Device]:
  strategy = None
  if device == Device.GPU:
    try:
      device_physical = tf.config.list_physical_devices('GPU')[0]
      strategy = tf.distribute.MirroredStrategy(devices=[device_physical])  # or [device_physical.name]
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
        #device_physical = tf.config.list_physical_devices('TPU')[0]
        #tf.config.set_visible_devices(device_physical, 'TPU')
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
        # instantiate a distribution strategy
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        # https://www.kaggle.com/docs/tpu
        # TPU v3-8 on Kaggle has 8 cores.  increase batch_size MXU is not near 100% in TPU monitor
        # e.g. batch_size = 16 * strategy.num_replicas_in_sync
        tf.config.optimizer.set_jit(False)
        logging.info(f"TPU is available and set as default device:{tpu.master()}")
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

#https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/examples/penguin/penguin_utils_base.py#L43
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
    raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
    transformed_features = model.tft_layer(raw_features)
    logging.info('serve_transformed_features = %s', transformed_features)

    outputs = model(transformed_features)
    # TODO(b/154085620): Convert the predicted labels from the model using a
    # reverse-lookup (opposite of transform.py).
    return {'outputs': outputs}

#https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/examples/chicago_taxi_pipeline/taxi_utils.py#L128
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
    raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
    transformed_features = model.tft_layer_eval(raw_features)
    logging.info('eval_transformed_features = %s', transformed_features)
    return transformed_features

  return transform_features_fn

#https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/docs/tutorials/tfx/recommenders.ipynb#L851
#https://github.com/tensorflow/tfx/blob/master/docs/tutorials/tfx/recommenders.ipynb
def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example and applies TFT."""
  try:
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
      """Returns the output to be used in the serving signature."""
      try:
        feature_spec = tf_transform_output.raw_feature_spec()
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        result = model(transformed_features)
      except BaseException as err:
        raise BaseException('######## ERROR IN serve_tf_examples_fn:\n{}\n###############'.format(err))
      return result
  except BaseException as err2:
      raise BaseException('######## ERROR IN _get_serve_tf_examples_fn:\n{}\n###############'.format(err2))

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

  #not sure if outputs query_model and candidate_model are passed to this.
  for attr_name in dir(fn_args):
    # Filter out built-in methods and private attributes
    if not attr_name.startswith('__') and not callable(getattr(fn_args, attr_name)):
      attr_value = getattr(fn_args, attr_name)
      logging.debug(f"{attr_name}: {attr_value}")

  if not fn_args.hyperparameters:
    raise ValueError('hyperparameters must be provided')

  logging.debug(f'fn_args.serving_model_dir={fn_args.serving_model_dir}')
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
    #model = _make_2tower_keras_model(hp, tf_transform_output)

  # Write logs to path
  #tensorboard_callback = tf.keras.callbacks.TensorBoard(
  #    log_dir=fn_args.model_run_dir, update_freq='epoch')

  stop_early = tf.keras.callbacks.EarlyStopping(
    monitor=f'val_{LOSS_FN.name}', patience=3)

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[stop_early])

  #save all 3 models
  # return the two twoer model
  model.save(fn_args.serving_model_dir, save_format='tf')
  #should handle saving decorated methods too, that is,
  #those decorated with @keras.saving.register_keras_serializable(package=package)

  model.query_model.save(query_model_serving_dir, save_format='tf')
  model.candidate_model.save(candidate_model_serving_dir, save_format='tf')

  #should be able to use model.save without creating a signature profile
  #  but if have troubles with this, use one of these 2:
  # signatures = _make_serving_signatures(model, tf_transform_output)
  # signatures = {
  #  'serving_default':
  #    _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
  #      tf.TensorSpec( shape=[None], dtype=tf.string, name='examples')),
  #  'transform_features':
  #    _get_transform_features_signature(model, tf_transform_output),
  #}
  #tf.saved_model.save(model, fn_args.serving_model_dir, signatures=signatures)

  return model

# TFX Tuner will call this function.
def tuner_fn(fn_args: tfx.components.FnArgs) -> tfx.components.TunerFnResult:
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

  #the objective must be must be a name that appears in the logs
  # returned by the model.fit() method during training.
  tuner = keras_tuner.RandomSearch(
      _make_2tower_keras_model,
      max_trials=6,
      hyperparameters=hp,
      allow_new_entries=False,
      objective=keras_tuner.Objective(f'val_{LOSS_FN.name}','min'),
      #objective=keras_tuner.Objective('val_loss', 'min'),
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