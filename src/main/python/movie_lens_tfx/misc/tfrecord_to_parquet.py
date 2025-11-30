import os
import random

import apache_beam as beam
from apache_beam.io import parquetio
import pyarrow as pa
import json

import tensorflow_transform as tft
from tfx.dsl.component.experimental import annotations
from tfx.dsl.component.experimental.decorators import component
from tfx.types import standard_artifacts
from tensorflow.io import parse_single_example
from tensorflow import float32 as tf_float32
from tensorflow import float64 as tf_float64
from tensorflow import int64 as tf_int64
from tensorflow import int32 as tf_int32
from tensorflow import int16 as tf_int16
from tensorflow import int8 as tf_int8
from tensorflow import float16 as tf_float16

from movie_lens_tfx.utils.movie_lens_utils import serialize_to_string

from tfx import v1 as tfx

@component(use_beam=True)
def FromTFRecordToParquet(
    transform_graph: tfx.dsl.components.InputArtifact[standard_artifacts.TransformGraph],
    transformed_examples: tfx.dsl.components.InputArtifact[standard_artifacts.Examples],
    output_file_path: tfx.dsl.components.Parameter[str],
    beam_pipeline: annotations.BeamComponentParameter[beam.Pipeline]=None
):
    """A custom TFX component that writes parquet files from the transformed examples
    to output_filepath."""
    
    tf_transform_output = tft.TFTransformOutput(transform_graph.uri)
    post_transform_feature_spec = tf_transform_output.transformed_feature_spec()
    
    pa_schema_list = []
    for k, v in post_transform_feature_spec.items():
      if k == 'genres':
        #NOTE: hard-wiring the serialization of array to string type here
        pa_schema_list.append((k, pa.string()))
      elif v.dtype == tf_float64:
        pa_schema_list.append((k, pa.float64()))
      elif v.dtype == tf_float32:
        pa_schema_list.append((k, pa.float32()))
      elif v.dtype == tf_float16:
        pa_schema_list.append((k, pa.float16()))
      elif v.dtype == tf_int64:
        pa_schema_list.append((k, pa.int64()))
      elif v.dtype == tf_int32:
        pa_schema_list.append((k, pa.int32()))
      elif v.dtype == tf_int16:
        pa_schema_list.append((k, pa.int16()))
      elif v.dtype == tf_int8:
        pa_schema_list.append((k, pa.int8()))
       
    pa_schema = pa.schema(pa_schema_list)
    #print(f"PA_SCHEMA={pa_schema}")
    #print(f'post_transform_feature_spec={post_transform_feature_spec}')
    
    os.makedirs(output_file_path, exist_ok=True)
    
    transformed_examples_uri = transformed_examples.uri
    
    split_names =transformed_examples.split_names
    if not split_names:
      split_names = []
    else:
      split_names = json.loads(split_names)
    print(f'FromTFRecordToParquet split_names = {split_names}')
    
    class ParseTFExampleToDict(beam.DoFn):
      """
      parse each tf.example into a dictionary.  Note that any arrays with size > 1 are string serialized,
      and so the pa_schema has to be adjusted accordingly
      """
      def process(self, example_proto):
        parsed_features = parse_single_example(example_proto, post_transform_feature_spec)
        output = {}
        for k, v in parsed_features.items():
          arr = v.numpy()
          if arr.size == 1:
            output[k] = arr[0]
          else:
            output[k] = serialize_to_string(arr[0].tolist())
        yield output
        
    with beam_pipeline as pipeline:
      for split_name in split_names:
        in_file_path = os.path.join(transformed_examples_uri, f"Split-{split_name}")
        out_file_path = os.path.join(output_file_path, f"Split-{split_name}")
        os.makedirs(in_file_path, exist_ok=True)
        # ReadFromTFRecords creates serialized tf.train.Example protocol buffers
        # ParseTFExampleToDict creates a dictionary of features of numpy arrays
        # WriteToParquet writes the data to parquet files
        in_file_pattern = os.path.join(in_file_path, "*")
        _ = ( pipeline
          | f"ReadTFRecords_{random.randint(0,1000000000)}"
          >> beam.io.tfrecordio.ReadFromTFRecord(file_pattern=in_file_pattern, coder=beam.coders.BytesCoder())
          #>> beam.io.tfrecordio.ReadFromTFRecord(file_pattern=in_file_pattern)
          | f"ParseStringIntoFeatures_{random.randint(0,1000000000)}"
          >> beam.ParDo(ParseTFExampleToDict())
          | f"WriteParquet_{random.randint(0,1000000000)}"
          >> parquetio.WriteToParquet(file_path_prefix=out_file_path, schema=pa_schema, file_name_suffix='.parquet')
          )
    

