
import shutil

from tfx.dsl.io import fileio
from tfx.orchestration import metadata
from tfx.components import StatisticsGen, SchemaGen, ExampleValidator
from tfx.utils import io_utils
from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_transform.tf_metadata import schema_utils

from movie_lens_tfx.ingest_pyfunc_component.ingest_movie_lens_component import *
#import trainer_movie_lens

from ml_metadata.metadata_store import metadata_store
from movie_lens_tfx.tune_train_movie_lens import *

from helper import *

tf.get_logger().propagate = False
from absl import logging
logging.set_verbosity(logging.WARNING)
logging.set_stderrthreshold(logging.WARNING)

class TuneTrainTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.infiles_dict_ser, self.output_config_ser, self.split_names = \
      get_test_data()
    self.user_id_max = 6040
    self.movie_id_max = 3952
    self.n_genres = N_GENRES
    self.n_age_groups = N_AGE_GROUPS
    self.n_occupations = 21
    self.name = 'test run of ratings transform'

  def test_io(self):
    file_paths = ["/home/nichole/projects/github/recommender_systems/bin/tune_train_1/test_tune_and_train/TestPythonTransformPipeline/MovieLensExampleGen/output_examples/1/Split-train/tfrecord-00000-of-00004.gz"]
    test_raw_ds_ser = tf.data.TFRecordDataset(file_paths, compression_type="GZIP")
    
    raw_schema_file_path = "/home/nichole/projects/github/recommender_systems/bin/tune_train_1/test_tune_and_train/TestPythonTransformPipeline/Transform/pre_transform_schema/5/schema.pbtxt"
    raw_schema = tfx.utils.parse_pbtxt_file(raw_schema_file_path, schema_pb2.Schema())
    raw_feature_spec = schema_utils.schema_as_feature_spec(raw_schema).feature_spec
  
    def parse_tf_example(example_proto, feature_spec):
      return tf.io.parse_single_example(example_proto, feature_spec)
    raw_ds = test_raw_ds_ser.map(lambda x: parse_tf_example(x, raw_feature_spec))
   
    raw_ds_row = next(iter(raw_ds))
    raw_example = next(iter(test_raw_ds_ser))
    
    # raw_example:
    #    <tf.Tensor: shape=(), dtype=string, numpy=b'\n\x9f\x01\n\x1a\n\x06genres\x12\x10\n\x0e\n\x0cAction|Drama\n\x0f\n\x06gender\x12\x05\n\x03\n\x01M\n\x0f\n\x06rating\x12\x05\x1a\x03\n\x01\x04\n\x16\n\ttimestamp\x12\t\x1a\x07\n\x05\xf4\xab\xbe\xd2\x03\n\x0c\n\x03age\x12\x05\x1a\x03\n\x01-\n\x12\n\x08movie_id\x12\x06\x1a\x04\n\x02\xda\x1a\n\x13\n\noccupation\x12\x05\x1a\x03\n\x01\x07\n\x10\n\x07user_id\x12\x05\x1a\x03\n\x01\x04'>
    
    # raw_ds_row:
    #   {'age': <tf.Tensor: shape=(1,), dtype=int64, numpy=array([45])>, 'gender': <tf.Tensor: shape=(1,), dtype=string, numpy=array([b'M'], dtype=object)>, 'genres': <tf.Tensor: shape=(1,), dtype=string, numpy=array([b'Action|Drama'], dtype=object)>, 'movie_id': <tf.Tensor: shape=(1,), dtype=int64, numpy=array([3418])>, 'occupation': <tf.Tensor: shape=(1,), dtype=int64, numpy=array([7])>, 'rating': <tf.Tensor: shape=(1,), dtype=int64, numpy=array([4])>, 'timestamp': <tf.Tensor: shape=(1,), dtype=int64, numpy=array([978294260])>, 'user_id': <tf.Tensor: shape=(1,), dtype=int64, numpy=array([4])>}
    
    parsed_example = tf.train.Example.FromString(raw_example.numpy())
    
    q = convert_dict_inputs_to_tfexample_ser(raw_ds_row)
    for x in raw_ds.batch(2):
      tf_example = convert_dict_inputs_to_tfexample_ser(x)
      break
    
    tt = 4

    #  features {
    #   feature {
    #     key: "age"
    #     value {
    #       int64_list {
    #         value: 45
    #       }
    #     }
    #   }
    #   feature {
    #     key: "gender"
    #     value {
    #       bytes_list {
    #         value: "M"
    #       }
    #     }
    #   }
    #   feature {
    #     key: "genres"
    #     value {
    #       bytes_list {
    #         value: "Action|Drama"
    #       }
    #     }
    #   }
    #   feature {
    #     key: "movie_id"
    #     value {
    #       int64_list {
    #         value: 3418
    #       }
    #     }
    #   }
    #   feature {
    #     key: "occupation"
    #     value {
    #       int64_list {
    #         value: 7
    #       }
    #     }
    #   }
    #   feature {
    #     key: "rating"
    #     value {
    #       int64_list {
    #         value: 4
    #       }
    #     }
    #   }
    #   feature {
    #     key: "timestamp"
    #     value {
    #       int64_list {
    #         value: 978294260
    #       }
    #     }
    #   }
    #   feature {
    #     key: "user_id"
    #     value {
    #       int64_list {
    #         value: 4
    #       }
    #     }
    #   }
    # }
    
