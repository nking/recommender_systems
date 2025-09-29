# code to use as a tft test.  
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
import apache_beam as beam
import numpy as np

# Define the preprocessing function
def preprocessing_fn(inputs):
  x = inputs['x']
  # Scale the input 'x' to have a mean of 0 and variance of 1
  x_centered = x - tft.mean(x)
  return {
      'x_centered': x_centered
  }

# Define your input data
input_data = [
    {'x': np.array([1.0], dtype=np.float32)},
    {'x': np.array([2.0], dtype=np.float32)},
    {'x': np.array([3.0], dtype=np.float32)},
]

try:
    # Use a pipeline to test the transformation
    with beam.Pipeline() as pipeline:
      with tft_beam.Context(temp_dir='/tmp'):
        dataset_metadata = tft.DatasetMetadata.from_feature_spec({
            'x': tf.io.FixedLenFeature([], tf.float32)
        })
        transformed_dataset, transform_fn = (
            (pipeline | 'CreateData' >> beam.Create(input_data))
            .with_input_types(dataset_metadata.schema)
            | 'AnalyzeAndTransform' >> tft_beam.AnalyzeAndTransformDataset(
                preprocessing_fn
            )
        )
        (
            transformed_dataset
            | 'PrintOutput' >> beam.Map(print)
        )
except Exception as ex:
  print(f'error: {ex}')

