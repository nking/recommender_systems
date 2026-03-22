import os
import shutil
import numpy as np
from apache_beam.transforms.combiners import Top
from apache_beam.transforms.stats import ApproximateQuantiles
import io
import csv
from array_record_beam_sdk import arrayrecordio
import msgpack
from array_record.python import array_record_module

"""
This writes ratings file to array_record format for use by the Ranker project.
"""

from apache_beam.options.pipeline_options import PipelineOptions
from tensorflow_metadata.proto.v0 import schema_pb2

from movie_lens_tfx.ingest_pyfunc_component.ingest_movie_lens_component import *

from movie_lens_tfx.tune_train_movie_lens import *

from helper import *

tf.get_logger().propagate = False
from absl import logging

logging.set_verbosity(logging.DEBUG)
logging.set_stderrthreshold(logging.DEBUG)

MAX_MOVIE_ID = 3952
N_MOVIES = 3883


class WriteRankerInputArrayRecords(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        
        self.rewrite_all = False
        
        self.pipeline_options = PipelineOptions(
            runner='DirectRunner',
            direct_num_workers=0,
            direct_running_mode='multi_processing',
            # direct_running_mode='multi_threading',
        )
        
        self.input_path0 = os.path.join(get_project_dir(),
            'src/main/resources/ml-1m/ratings_timestamp_sorted_part_1.dat')
        self.input_path1 = os.path.join(get_project_dir(),
            'src/main/resources/ml-1m/ratings_timestamp_sorted_part_2.dat')
        
        self.movie_csv_uri = os.path.join(get_project_dir(),
            'src/main/resources/ml-1m/movies.dat')
        self.movie_array_record_uri = os.path.join(get_bin_dir(),
            'movie_ids.array_record')
        
        self.output_uri0 = os.path.join(get_bin_dir(), "ratings_part_1.array_record")
        self.output_uri1 = os.path.join(get_bin_dir(), "ratings_part_2.array_record")
    
    def test_write_array_records(self):
       
        pipeline0 = beam.Pipeline(options=self.pipeline_options)
        
        pc = (pipeline0 | f"read_ratings_raw_train" >>
              beam.io.ReadFromText(self.input_path0, skip_header_lines=0,
                  coder=CustomUTF8Coder())
              | f'parse_ratings_raw_train' >> beam.Map(lambda line: line.split("::")))
        
        #serialized_train = (pc | "FormatToDict_train" >> beam.Map(lambda x: {
        #    "user_id": int(x[0]), "movie_id": int(x[1]),
        #    "rating": float(x[2]),"timestamp": int(x[3])})
        #    | "SerializeWithMsgpack_train" >> beam.Map(msgpack.packb))
        
        serialized_train = (pc | "FormatToList_train" >> beam.Map(lambda x: [
            int(x[0]), int(x[1]), int(x[2]), int(x[3])])
            | "SerializeWithMsgpack_train" >> beam.Map(msgpack.packb))
        
        (serialized_train | f'write_array_record_train'
            >> arrayrecordio.WriteToArrayRecord(file_path_prefix=self.output_uri0,
            num_shards=1))
        
        pc2 = (pipeline0 | f"read_ratings_raw_test" >>
              beam.io.ReadFromText(self.input_path1,
                  skip_header_lines=0,
                  coder=CustomUTF8Coder())
              | f'parse_ratings_raw_test' >> beam.Map(
                    lambda line: line.split("::")))
        
        #serialized_train2 = (pc2 | "FormatToDict_test" >> beam.Map(lambda x: {
        #        "user_id": int(x[0]), "movie_id": int(x[1]),
        #        "rating": float(x[2]), "timestamp": int(x[3])})
        #            | "SerializeWithMsgpack_test" >> beam.Map(
        #        msgpack.packb))
        
        serialized_train2 = (pc2 | "FormatToList_test" >> beam.Map(lambda x: [
            int(x[0]), int(x[1]), int(x[2]), int(x[3])])
            | "SerializeWithMsgpack_test" >> beam.Map(msgpack.packb))
        
        (serialized_train2 | f'write_array_record_test'
         >> arrayrecordio.WriteToArrayRecord(file_path_prefix=self.output_uri1,
                    num_shards=1))
        
        pipeline0.run()
    
    def test_write_movies_array_record(self):
        
        writer = None
        try:
            writer = array_record_module.ArrayRecordWriter(self.movie_array_record_uri, "group_size:1")
            with open(self.movie_csv_uri, mode='r', encoding='iso-8859-1') as f:
                for line in f:
                    movie_id = int(line.strip().split("::")[0])
                    writer.write(msgpack.packb(movie_id, use_bin_type=True))
        except Exception as ex:
            raise ex
        finally:
            if writer is not None:
                writer.close()
        
        reader = None
        try:
            reader = array_record_module.ArrayRecordReader(self.movie_array_record_uri)
            record: list = msgpack.unpackb(reader.read())
            self.assertEqual(1, 1)
        except Exception as ex:
            raise ex
        finally:
            if reader is not None:
                reader.close()
        
    
    def test_read_array_records(self):
        import os
        from array_record.python import array_record_module

        for filename in ["ratings_part_1.array_record-00000-of-00001",
            "ratings_part_2.array_record-00000-of-00001"]:
            filepath = os.path.join(get_bin_dir(),filename)
            if os.path.exists(filepath):
                reader = None
                try:
                    reader = array_record_module.ArrayRecordReader(filepath)
                    # read with random access, just 1 record
                    record: list = msgpack.unpackb(reader.read())
                    self.assertEqual(4, len(record))
                finally:
                    if reader is not None:
                        reader.close()
                """record: dict = msgpack.unpackb(reader.read())
                self.assertTrue(record['user_id'] is not None)
                self.assertTrue(record['movie_id'] is not None)
                self.assertTrue(record['rating'] is not None)
                self.assertTrue(record['timestamp'] is not None)"""


