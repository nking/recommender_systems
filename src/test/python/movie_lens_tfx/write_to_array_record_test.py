import glob
import time
import unittest

from helper import get_project_dir, get_test_data, get_bin_dir
from movie_lens_tfx.utils.WriteToArrayRecord import WriteToArrayRecord
from movie_lens_tfx.utils.ingest_movie_lens_beam import ReadFiles
#from ... main.python.infile_dict_util import *
#from ... main.python.infile_dict_util import _assert_dict_1
from movie_lens_tfx.utils.movie_lens_utils import *
import msgpack
import apache_beam as beam
from array_record.python import array_record_module

from absl import logging
logging.set_verbosity(logging.WARNING)
logging.set_stderrthreshold(logging.WARNING)

from apache_beam.options.pipeline_options import PipelineOptions


def serialize_example(element):
    user_id, movie_id, rating, timestamp = element
    feature = {
        'user_id': tf.train.Feature(
            int64_list=tf.train.Int32List(value=[user_id])),
        
        'movie_id': tf.train.Feature(
            int64_list=tf.train.Int32List(value=[movie_id])),
        
        'rating': tf.train.Feature(
            int64_list=tf.train.Int32List(value=[rating])),
        
        'timestamp': tf.train.Feature(
            int64_list=tf.train.Int32List(value=[timestamp])),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def cast_tuple_to_ints(element):
    """Unpacks the string element tuple and converts every value to an integer."""
    user_id, movie_id, rating, timestamp = element
    return int(user_id), int(movie_id), int(rating), int(timestamp)

class TestWriteToArrayRecord(unittest.TestCase):
    
    def setUp(self):
        self.infiles_dict_ser, self.output_config_ser, self.split_names = get_test_data()
        try:
            self.infiles_dict = deserialize(self.infiles_dict_ser)
        except Exception as ex:
            err = f"error with deserialize(infiles_dict_ser)"
            logging.error(f'{err} : {ex}')
            raise ValueError(f'{err} : {ex}')
        self.name = 'test write to array_record'
    
    def test_write(self):
        
        options = PipelineOptions(
            runner='DirectRunner',
            direct_num_workers=0,
            direct_running_mode='multi_processing',
            # direct_running_mode='multi_threading',
        )
        
        output = os.path.join(get_bin_dir(), 'tmp_ratings')
        
        with beam.Pipeline(options=options) as pipeline:
            # test read files
            pc = pipeline | f"read_{time.time_ns()}" >> ReadFiles(self.infiles_dict)
    
            ratings_pc = pc['ratings']
            
            #ratings_pc | "print movie_id_and_embeddings" >> beam.Map(lambda x: print(f"movie id, emb: {x}"))
            
            (ratings_pc
               | 'parse_to_ints' >> beam.Map(cast_tuple_to_ints)
               | 'msgpack' >> beam.Map(msgpack.packb)
               | 'write_rating_arrayrecord' >> WriteToArrayRecord(
                        file_path_prefix=output,
                        file_name_suffix='.array_record'
                    ))
            
        
        files = glob.glob(f"{output}*array_record")
        for file_path in files:
            reader = None
            try:
                reader = array_record_module.ArrayRecordReader(file_path)
                record = msgpack.unpackb(reader.read(), use_list=False)
                self.assertEqual(4, len(record))
                self.assertTrue(isinstance(record[0], int))
                self.assertTrue(isinstance(record[1], int))
                self.assertTrue(isinstance(record[2], int))
                self.assertTrue(isinstance(record[3], int))
            except Exception as e:
                self.fail(e)
            finally:
                if reader is not None:
                    reader.close()
        

if __name__ == '__main__':
    unittest.main()
