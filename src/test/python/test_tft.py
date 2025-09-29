import tensorflow as tf
import tensorflow_transform as tft
import apache_beam as beam

# Define the raw data and its schema
RAW_DATA_FEATURE_SPEC = {
  'x': tf.io.FixedLenFeature([], tf.float32),
  'y': tf.io.FixedLenFeature([], tf.string)
}
RAW_DATA = [
  {'x': 10.0, 'y': 'apple'},
  {'x': 20.0, 'y': 'orange'},
  {'x': 30.0, 'y': 'apple'},
  {'x': 40.0, 'y': 'apple'}
]


# Define the preprocessing function
def preprocessing_fn(inputs):
  """Preprocesses input data by normalizing 'x' and vocabulary-izing 'y'."""
  # Scale 'x' using the min and max observed across the entire dataset
  x = inputs['x']
  x_scaled = tft.scale_by_min_max(x)

  # Vocabulary-ize 'y' to convert strings to integer IDs
  y = inputs['y']
  y_vocab = tft.compute_and_apply_vocabulary(y)

  return {'x_scaled': x_scaled, 'y_vocab': y_vocab}
