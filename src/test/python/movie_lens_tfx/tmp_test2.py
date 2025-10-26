import os.path
import unittest

from IPython import embed

from helper import *
import tensorflow as tf
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_metadata.proto.v0 import schema_pb2
import tfx.v1 as tfx

input_spec = {'arg1':tf.TensorSpec(shape=[1,], dtype=tf.float32),
  'arg2':tf.TensorSpec(shape=[1,], dtype=tf.float32)}

# The internal model class
@tf.keras.utils.register_keras_serializable(package='MyPackage')
class InternalModel(tf.keras.Model):
    def __init__(self, embed_dim:int=3, **kwargs):
        super(InternalModel, self).__init__(**kwargs)
        self.emb = tf.keras.layers.Embedding(100, embed_dim)
        self.embed_dim = embed_dim
      
    def call(self, inputs):
      return self.emb(inputs['arg2'])
    
    def get_config(self):
      config = super(InternalModel, self).get_config()
      config.update({"embed_dim": self.embed_dim})
      return config

# The top model class
@tf.keras.utils.register_keras_serializable(package='MyPackage')
class TopModel(tf.keras.Model):
    def __init__(self, layer_sizes:list[int]=[32], embed_dim:int=3, **kwargs):
        super(TopModel, self).__init__(**kwargs)
        # InternalModel is a tracked attribute (composition)
        self.internal_model = InternalModel(embed_dim)
        self.top_emb = tf.keras.models.Sequential([
          tf.keras.layers.Dense(layer_sizes[-1]),
        ])
        self.embed_dim = embed_dim
        self.layer_sizes = layer_sizes
    
    @tf.function(input_signature=[input_spec])
    def call(self, inputs):
      l1 = self.internal_model(inputs)
      return self.top_emb(l1)
    
    @tf.function(input_signature=[input_spec])
    def serve_internal_model(self, internal_data):
      """A dedicated function to trace and serve the InternalModel."""
      return self.internal_model(internal_data)  # Use the internal model's call logic
    
    def get_config(self):
      config = super(TopModel, self).get_config()
      config.update({"embed_dim": self.embed_dim,
                     "layer_sizes": self.layer_sizes,
                     })
      return config


class TmpTest2(unittest.TestCase):
  
    def test_1(self):
      s_dir = os.path.join(get_bin_dir(), "saved_tmp_2")
      os.makedirs(s_dir, exist_ok=True)
      # 1. Save the top model (example)
      top_model = TopModel()
      
      call_sig = top_model.call.get_concrete_function(
        input_spec
      )
      internal_sig = top_model.serve_internal_model.get_concrete_function(
        input_spec
      )
     
      signatures = {
        'serving_default': call_sig,
        'internal_serving': internal_sig,
      }
      tf.saved_model.save(top_model, s_dir, signatures=signatures)
      
      # 2. Load the top model
      loaded_top_model = tf.saved_model.load(s_dir)
      
      print(f'loaded SavedModel signatures: {loaded_top_model.signatures}')
      infer = loaded_top_model.signatures["serving_default"]
      infer_inner = loaded_top_model.signatures["internal_serving"]
      
      data = {'arg1': tf.random.uniform(shape=(1,), minval=0.0, maxval=1.0),
        'arg2': tf.random.uniform(shape=(1,), minval=0.0, maxval=1.0)}
      
      print(f"infer={infer(inputs=data['arg1'], inputs_1=data['arg2'])}")
      print(f"infer={infer_inner(internal_data=data['arg1'], internal_data_1=data['arg2'])}")
      

if __name__ == '__main__':
    unittest.main()
