import os.path
import unittest
from helper import *
import tensorflow as tf
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_metadata.proto.v0 import schema_pb2
import tfx.v1 as tfx

input_spec = tf.TensorSpec(shape=[None, 100], dtype=tf.float32)

# The internal model class
@tf.keras.utils.register_keras_serializable(package='MyPackage')
class InternalModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(10)
    def call(self, inputs):
      return self.dense(inputs)

# The top model class
@tf.keras.utils.register_keras_serializable(package='MyPackage')
class TopModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # InternalModel is a tracked attribute (composition)
        self.internal_model = InternalModel()
    
    @tf.function(input_signature=[input_spec])
    def call(self, inputs):
      return self.internal_model(inputs)
    
    @tf.function(input_signature=[input_spec])
    def serve_internal_model(self, internal_data):
      """A dedicated function to trace and serve the InternalModel."""
      return self.internal_model(internal_data)  # Use the internal model's call logic
      
class TmpTest2(unittest.TestCase):
  
    def test_1(self):
      s_dir = os.path.join(get_bin_dir(), "saved_tmp_2")
      os.makedirs(s_dir, exist_ok=True)
      # 1. Save the top model (example)
      top_model = TopModel()
      
      call_sig = top_model.call.get_concrete_function(
        tf.TensorSpec(shape=[None, 100], dtype=tf.float32)
      )
      internal_sig = top_model.serve_internal_model.get_concrete_function(
        tf.TensorSpec(shape=[None, 100], dtype=tf.float32)
      )
      signatures = {
        'serving_default': call_sig,
        'internal_serving': internal_sig
      }
      tf.saved_model.save(top_model, s_dir, signatures=signatures)
      
      # 2. Load the top model
      loaded_top_model = tf.saved_model.load(s_dir)
      
      print(f'loaded SavedModel signatures: {loaded_top_model.signatures}')
      infer = loaded_top_model.signatures["serving_default"]
      infer_inner = loaded_top_model.signatures["internal_serving"]
      

if __name__ == '__main__':
    unittest.main()
