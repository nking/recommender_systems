import os
import shutil
import numpy as np
from apache_beam.transforms.combiners import Top
from apache_beam.transforms.stats import ApproximateQuantiles
import io
import csv
import glob
from array_record_beam_sdk import arrayrecordio
import msgpack
from array_record.python import array_record_module
import tempfile
"""
This writes various files needed for the retrieval project.
It really should be refactored into components and unit tests...

TODO: write a movie pivot table for rated movies, and include the not rated movies
with columns 'title', 'movie_id', '1', '2', '3', '4', '5'
where the values in the later are the number of ratings of star '1' for 'movie_id', etc.
e.g.
data = {
      'title': ['loved_many', 'loved_few', 'loved_and_hated', 'hated', 'new_unrated',  'hated_few'],
      'movie_id': [1, 2, 3, 4, 5, 6],
      '1': [500,   2,  4000, 800, 0, 40],
      '2': [100,   1,  1000, 100, 0, 5],
      '3': [200,   0,  500,   50, 0, 0],
      '4': [3000,  5,  1000,  20, 0, 0],
      '5': [8000, 40,  4000,  10, 0, 0]
    }
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


# TODO: write a component for making the user and movie tfrecords needed for inputs to
# the embeddings in the Retrieval project.  for now, hard-wiring the columns instead of using
# pre-transformed schema
class WriteRetrievalInputs(tf.test.TestCase):
    """
    Creates from user.dat, tfrecords having the columns needed for inputs to data models, that is, columns that are the
    same as the ratings_joined columns.  Columns not present in user.dat are filled in with fake numbers.
    
    Create a similar file from movies.data.
    """
    
    def setUp(self):
        super().setUp()
        
        self.rewrite_all = True
        
        self.pipeline_options = PipelineOptions(
            runner='DirectRunner',
            direct_num_workers=0,
            direct_running_mode='multi_processing',
            # direct_running_mode='multi_threading',
        )
        
        self.mean_train_val_timestamp = None
        
        self.train_full_uri = os.path.join(get_project_dir(),
            'src/main/resources/ml-1m/ratings_train.dat')
        
        self.input_path1 = os.path.join(get_project_dir(),
            'src/main/resources/ml-1m/movies.dat')
        self.output_uri1 = os.path.join(get_bin_dir(), "movie_emb_inp")
        self.input_path2 = os.path.join(get_project_dir(),
            'src/main/resources/ml-1m/users.dat')
        self.output_uri2 = os.path.join(get_bin_dir(), "user_emb_inp")

        self.saved_model_path = os.path.join(get_project_dir(),
            'src/test/resources/serving_model/BEST')
        
        self.joined_ratings_feature_spec = {
            "user_id": tf.io.FixedLenFeature([], tf.int64),
            "movie_id": tf.io.FixedLenFeature([], tf.int64),
            "rating": tf.io.FixedLenFeature([], tf.int64),
            "timestamp": tf.io.FixedLenFeature([], tf.int64),
            "gender": tf.io.FixedLenFeature([], tf.string),
            "age": tf.io.FixedLenFeature([], tf.int64),
            "occupation": tf.io.FixedLenFeature([], tf.int64),
            "genres": tf.io.FixedLenFeature([], tf.string)}
    
    def test_write_movie_emb_tfrecords(self):
        """
        1) reads movies.dat and formats it into the joined ratings format of columns, filling in the missing values
        with 0's etc.
        2) create serialized tfexamples from that
        3) creates movie embeddings from the candidate model
        4) writes embeddings to output_uri1
        
        inputs are input_path1, input_path2
        outputs are output_uri1
        
        Returns:
          pipeline handle for use in waiting until finish etc.
          else None if rewrite_all=False and files are already written
        """
        
        if not self.rewrite_all:
            movie_tf_exists = os.path.exists(self.output_uri1) and bool(
                os.listdir(self.output_uri1))
            print(f'files exist: {movie_tf_exists}')
            if movie_tf_exists:
                return None
        
        try:
            shutil.rmtree(self.output_uri1)
        except Exception as ex:
            pass
        os.makedirs(self.output_uri1, exist_ok=True)
        
        # TODO: guard against race conditions
        if self.mean_train_val_timestamp is None:
            self.mean_train_val_timestamp = self._find_mean_timestamp()
        
        def serialize_example(element):
            movie_id, embedding = element
            feature = {
                'movie_id': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[movie_id])),
                'embedding': tf.train.Feature(
                    float_list=tf.train.FloatList(value=embedding))
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            return example_proto.SerializeToString()
        
        pipeline = beam.Pipeline(options=self.pipeline_options)
        
        ##====== rad moves.dat into serialized examples for inputs to candidate model ========
        
        #read movies.dat
        pc = pipeline | f"r{random.randint(0, 10000000000)}" >> \
             beam.io.ReadFromText(self.input_path1, skip_header_lines=0, coder=CustomUTF8Coder()) \
             | f'parse_movie_{random.randint(0, 1000000000000)}' >> \
             beam.Map(lambda line: line.split("::"))
        
        #create model input, supplying fake data for necessary, but unused keys
        examples_ser = (pc | f'movies_dat_ToTFExample'
                >> beam.ParDo(_FakeJoinedRatingsExampleMaker("movie", self.mean_train_val_timestamp))
                | f"Serialize_fake_example_movie"
                >> beam.Map(lambda x: x.SerializeToString()))
        
        #examples_ser | "print ser ex" >> beam.Map(lambda x: print(f"ex ser: {x}"))
        
        # calculate tuples of movie_id, embeddings
        movie_id_and_embeddings = (examples_ser
            | f'create_movie_emb'
            >> beam.ParDo(_CandidateEmbeddingMaker(saved_model_path=self.saved_model_path)))
       
        #movie_id_and_embeddings | "print movie_id_and_embeddings" >> beam.Map(lambda x: print(f"movie id, emb: {x}"))
        
        # write (movie_id, embeddings) to tfrecord files
        (movie_id_and_embeddings
            | f'serialize_movie_emb_example' >> beam.Map(serialize_example)
            | f'write_movie_emb_tfrecord' >> beam.io.tfrecordio.WriteToTFRecord(
            file_path_prefix=f'{self.output_uri1}/movie_emb', file_name_suffix='.gz'))
        
        result = pipeline.run()
        result.wait_until_finish()
        
        #assert file contents
        feature_spec = {
            'movie_id': tf.io.FixedLenFeature([], tf.int64),
            'embedding': tf.io.VarLenFeature(tf.float32),
        }
        def _parse_function(example_proto):
            return tf.io.parse_single_example(example_proto, feature_spec)
            
        # 1. Get all files matching your prefix
        files = glob.glob(f"{self.output_uri1}/movie_emb*.gz")
        raw_dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
        for raw_record in raw_dataset.take(1):
            parsed_record = _parse_function(raw_record)
            m_id = parsed_record['movie_id'].numpy()
            emb = parsed_record['embedding'].values.numpy()
            self.assertTrue(isinstance(m_id, np.int64))
            self.assertTrue(emb.shape[0] > 0)
        
    def test_write_user_emb_tfrecords(self):
        """
        1) reads users.dat and formats it into the joined ratings format of columns, filling in the missing values
        with 0's etc.
           Needs a representative timestamp for the queries
        2) create serialized tfexamples from that
        3) creates user embeddings from the query model
        4) writes embeddings to output_uri1

        inputs are input_path1, input_path2
        outputs are output_uri1

        Returns:
          pipeline handle for use in waiting until finish etc.
          else None if rewrite_all=False and files are already written
        """
        
        if not self.rewrite_all:
            user_tf_exists = os.path.exists(self.output_uri2) and bool(
                os.listdir(self.output_uri2))
            print(f'files exist: {user_tf_exists}')
            if user_tf_exists:
                return None
        
        try:
            shutil.rmtree(self.output_uri2)
        except Exception as ex:
            pass
        os.makedirs(self.output_uri2, exist_ok=True)
        
        #TODO: guard against race conditions
        if self.mean_train_val_timestamp is None:
            self.mean_train_val_timestamp = self._find_mean_timestamp()
        
        def serialize_example(element):
            movie_id, embedding = element
            feature = {
                'user_id': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[movie_id])),
                'embedding': tf.train.Feature(
                    float_list=tf.train.FloatList(value=embedding))
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            return example_proto.SerializeToString()
        
        pipeline = beam.Pipeline(options=self.pipeline_options)
        
        ##====== rad moves.dat into serialized examples for inputs to candidate model ========
        
        # read users.dat
        pc = (pipeline | f"read_user_dat" >>
             beam.io.ReadFromText(self.input_path2, skip_header_lines=0,
                 coder=CustomUTF8Coder())
             | f'parse_user_dat' >> beam.Map(lambda line: line.split("::")))
             
        # create model input, supplying fake data for necessary, but unused keys
        examples = (pc | f'user_ToTFExample_{random.randint(0, 1000000000000)}'
            >> beam.ParDo(_FakeJoinedRatingsExampleMaker("user", self.mean_train_val_timestamp)))
        
        examples_ser = (examples
                | f"Serialize_{random.randint(0, 1000000000000)}"
                >> beam.Map(lambda x: x.SerializeToString()))
        
        # calculate tuples of user_id, embeddings
        user_id_and_embeddings = (examples_ser
            | f'create_user_emb_{random.randint(0, 1000000000)}'
            >> beam.ParDo(_QueryEmbeddingMaker(saved_model_path=self.saved_model_path)))
       
        # write (movie_id, embeddings) to tfrecord files
        (user_id_and_embeddings
            | f'serialize_user_embedding_example' >> beam.Map(serialize_example)
            | f'write_user_emb_tfrecord' >> beam.io.tfrecordio.WriteToTFRecord(
            file_path_prefix=f'{self.output_uri2}/user_emb', file_name_suffix='.gz'))
        
        result = pipeline.run()
        result.wait_until_finish()
        
        ## assert contents
        feature_spec = {
            'user_id': tf.io.FixedLenFeature([], tf.int64),
            'embedding': tf.io.VarLenFeature(tf.float32),
        }
        def _parse_function(example_proto):
            return tf.io.parse_single_example(example_proto, feature_spec)
            
        # 1. Get all files matching your prefix
        files = glob.glob(f"{self.output_uri2}/user_emb*.gz")
        raw_dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
        for raw_record in raw_dataset.take(1):
            parsed_record = _parse_function(raw_record)
            u_id = parsed_record['user_id'].numpy()
            emb = parsed_record['embedding'].values.numpy()
            self.assertTrue(isinstance(u_id, np.int64))
            self.assertTrue(emb.shape[0] > 0)
   
    def create_example_movie_id_prediction(row):
        # each row is a tuple like:
        # (parsed_features['movie_id'] is a tensor like: < tf.Tensor: shape=(1,), dtype = int64, numpy = array([7]),
        # emb is tensor like: tf.Tensor: shape = (1, 32), dtype = float32, numpy = array([[ 0.08...)
        m_id = row[0]
        pred = row[1]
        feature_map = {
            'movie_id': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[m_id])),
            'prediction_mm': tf.train.Feature(
                float_list=tf.train.FloatList(value=[pred]))}
        return tf.train.Example(
            features=tf.train.Features(feature=feature_map))
   
    def movie_ratings_mean(row: Dict[str, int], columns_to_sum: List[str]):
        sums = sum([row[key] * (float(key)) for key in columns_to_sum])
        row['movie_ratings_mean'] = sums / row['total_votes']
        return row
    
    def test_write_all_movies_to_tfrecords(self):
        """
        writes movies.dat to movies*tfrecord*gz
        """
        in_file_path = os.path.join(get_project_dir(),
            "src/main/resources/ml-1m/movies.dat")
        out_file_path = os.path.join(get_bin_dir(), "movies_tfrecords")
        os.makedirs(out_file_path, exist_ok=True)
        out_file_prefix = f'{out_file_path}/movies'
        
        pipeline1 = beam.Pipeline(options=self.pipeline_options)
        
        (pipeline1 | "read movies.dat" >>
         beam.io.ReadFromText(in_file_path, skip_header_lines=0,
             coder=CustomUTF8Coder())
         | "parse movies.dat" >> beam.Map(lambda line: line.split("::"))
         | "create serialized movie examples" >> beam.Map(
                    create_serialized_example_for_movies)
         | 'WriteToTFRecord for movies' >> beam.io.tfrecordio.WriteToTFRecord(
                    file_path_prefix=out_file_prefix,
                    file_name_suffix='.tfrecord')
         )
        result = pipeline1.run()
        result.wait_until_finish()
        
        ## assert recrods are readable
        feature_spec = {
            "movie_id": tf.io.FixedLenFeature([], tf.int64),
            "title": tf.io.FixedLenFeature([], tf.string),
            "genres": tf.io.FixedLenFeature([], tf.string),
        }
        
        def _parse_function(example_proto):
            return tf.io.parse_single_example(example_proto, feature_spec)
        
        t = f'{out_file_prefix}*.tfrecord'
        file_paths = glob.glob(t)
        dataset = tf.data.TFRecordDataset(file_paths)
        parsed_dataset = dataset.map(_parse_function)
        t = None
        for x in parsed_dataset.batch(1):
            t = x
            self.assertTrue('movie_id' in x)
            self.assertTrue('title' in x)
            self.assertTrue('genres' in x)
            break
        self.assertIsNotNone(t)
    
    def test_write_ratings_to_array_records(self):
        """
        writes all ratings*dat to ratings*array_record
        """
        
        pipeline = beam.Pipeline(options=self.pipeline_options)
        
        for file_name in ["ratings", "ratings_train", "ratings_val",
            "ratings_test", "ratings_train_liked", "ratings_val_liked", "ratings_test_liked"]:
            
            in_file_path = os.path.join(get_project_dir(),
                f"src/main/resources/ml-1m/{file_name}.dat")
            out_file_path = os.path.join(get_bin_dir(), f"{file_name}.array_record")
            
            (pipeline | f"read_{file_name}" >>
             beam.io.ReadFromText(in_file_path, skip_header_lines=0,
                 coder=CustomUTF8Coder())
             | f'parse_{file_name}' >> beam.Map(lambda line: line.split("::"))
             | f"FormatToList_{file_name}" >> beam.Map(
                        lambda x: [int(x[0]), int(x[1]), int(x[2]), int(x[3])])
             | f"SerializeWithMsgpack_{file_name}" >> beam.Map(msgpack.packb)
             | f'write_array_record_{file_name}'
             >> arrayrecordio.WriteToArrayRecord(
                        file_path_prefix=out_file_path, num_shards=1))
        
        os.makedirs(os.path.join(get_bin_dir(), "small"), exist_ok=True)
        for file_name in ["ratings", "ratings_train_liked",
            "ratings_val_liked", "ratings_test_liked"]:
            in_file_path = os.path.join(get_project_dir(),
                f"src/test/resources/ml-1m/small/{file_name}.dat")
            out_file_path = os.path.join(get_bin_dir(), "small",
                f"{file_name}.array_record")
            
            (pipeline | f"read_small_{file_name}" >>
             beam.io.ReadFromText(in_file_path, skip_header_lines=0,
                 coder=CustomUTF8Coder())
             | f'parse_small_{file_name}' >> beam.Map(lambda line: line.split("::"))
             | f"FormatToList_small_{file_name}" >> beam.Map(
                        lambda x: [int(x[0]), int(x[1]), int(x[2]), int(x[3])])
             | f"SerializeWithMsgpack_small_{file_name}" >> beam.Map(msgpack.packb)
             | f'write_array_record_small_{file_name}'
             >> arrayrecordio.WriteToArrayRecord(
                        file_path_prefix=out_file_path, num_shards=1))
        
        result = pipeline.run()
        result.wait_until_finish()
        
        # assert wrote correctly
        for file_name in ["ratings", "ratings_train", "ratings_val",
            "ratings_test",
            "ratings_train_liked", "ratings_val_liked", "ratings_test_liked"]:
            file_path = glob.glob(f'{get_bin_dir()}/{file_name}.array_record*')[0]
            out_file_path = os.path.join(get_bin_dir(), f"{file_name}.array_record")
            shutil.move(file_path, out_file_path)
            self._read_array_records(out_file_path)
            
        for file_name in ["ratings", "ratings_train_liked",
            "ratings_val_liked", "ratings_test_liked"]:
            file_path = glob.glob(f'{get_bin_dir()}/small/{file_name}.array_record*')[0]
            out_file_path = os.path.join(get_bin_dir(), "small", f"{file_name}.array_record")
            shutil.move(file_path, out_file_path)
            self._read_array_records(out_file_path)
    
    def _read_array_records(self, file_path: str):
        
        reader = None
        try:
            reader = array_record_module.ArrayRecordReader(file_path)
            record: list = msgpack.unpackb(reader.read())
            self.assertEqual(4, len(record))
        except Exception as e:
            self.fail(e)
        finally:
            if reader is not None:
                reader.close()
    
    def test_write_movies_array_record(self):
        infile_path = os.path.join(get_project_dir(),
            'src/main/resources/ml-1m/movies.dat')
        outfile_path = os.path.join(get_bin_dir(),
            'movie_ids.array_record')
        writer = None
        try:
            writer = array_record_module.ArrayRecordWriter(outfile_path,
                "group_size:1")
            with open(infile_path, mode='r', encoding='iso-8859-1') as f:
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
            reader = array_record_module.ArrayRecordReader(outfile_path)
            record: list = msgpack.unpackb(reader.read())
            self.assertEqual(1, 1)
        except Exception as ex:
            raise ex
        finally:
            if reader is not None:
                reader.close()
    
    def test_write_all_users_to_tfrecords(self):
        """
        writes users.dat to users*tfrecord*gz
        """
        in_file_path = os.path.join(get_project_dir(),
            "src/main/resources/ml-1m/users.dat")
        out_file_path = os.path.join(get_bin_dir(), "users_tfrecords")
        os.makedirs(out_file_path, exist_ok=True)
        out_file_prefix = f'{out_file_path}/users'
        
        pipeline1 = beam.Pipeline(options=self.pipeline_options)
        
        (pipeline1 | "read users.dat" >>
         beam.io.ReadFromText(in_file_path, skip_header_lines=0,
             coder=CustomUTF8Coder())
         | "parse users.dat" >> beam.Map(lambda line: line.split("::"))
         | "create serialized users examples" >> beam.Map(
                    create_serialized_example_for_users)
         | 'WriteToTFRecord for users' >> beam.io.tfrecordio.WriteToTFRecord(
                    file_path_prefix=out_file_prefix,
                    file_name_suffix='.tfrecord'))
        result = pipeline1.run()
        result.wait_until_finish()
        
        feature_spec = {
            "user_id": tf.io.FixedLenFeature([], tf.int64),
            "gender": tf.io.FixedLenFeature([], tf.string),
            "age": tf.io.FixedLenFeature([], tf.int64),
            "occupation": tf.io.FixedLenFeature([], tf.int64),
            "zipcode": tf.io.FixedLenFeature([], tf.string),
        }
        
        def _parse_function(example_proto):
            return tf.io.parse_single_example(example_proto, feature_spec)
        
        t = f'{out_file_prefix}*.tfrecord'
        file_paths = glob.glob(t)
        dataset = tf.data.TFRecordDataset(file_paths)
        parsed_dataset = dataset.map(_parse_function)
        t = None
        for x in parsed_dataset.batch(1):
            t = x
            self.assertTrue('user_id' in x)
            self.assertTrue('gender' in x)
            self.assertTrue('age' in x)
            self.assertTrue('occupation' in x)
            self.assertTrue('zipcode' in x)
            break
        self.assertIsNotNone(t)
    
    def test_write_movies_pivot_table(self):
        dir_path = os.path.join(get_bin_dir(), 'movies_pivot_table')
        file_name = "movie_ratings_pivot_table"
        if not self.rewrite_all:
            files1 = glob.glob(os.path.join(dir_path, f'{file_name}.array_record*'))
            files2 = glob.glob(os.path.join(dir_path, f'{file_name}.tfrecrod*'))
            if len(files1) > 0 and len(files2) > 0:
                return None
        
        try:
            shutil.rmtree(dir_path)
        except Exception as ex:
            pass
        os.makedirs(dir_path, exist_ok=True)
        
        #input_column_name_type_list = [('user_id', int), ('movie_id', int),
        #    ('rating', int), ('timestamp', int)]
        
        pipeline = beam.Pipeline(options=self.pipeline_options)
        
        # read in and merge the train tainted ratings files: user_id::movie_id::rating::timestamp
        pc1 = (pipeline | f"read_train_ratings" >>
             beam.io.ReadFromText(os.path.join(get_project_dir(), f"src/main/resources/ml-1m/ratings_train.dat"),
                 skip_header_lines=0, coder=CustomUTF8Coder())
             | f'parse_train_ratings' >> beam.Map(lambda line: line.split("::")))
             
        pc2 = (pipeline | f"read_val_ratings" >>
            beam.io.ReadFromText(os.path.join(get_project_dir(), f"src/main/resources/ml-1m/ratings_val.dat"),
            skip_header_lines=0, coder=CustomUTF8Coder())
            | f'parse_val_ratings' >> beam.Map(lambda line: line.split("::")))
        
        ratings_pc = (pc1, pc2) | "merge train and val ratings" >> beam.Flatten()
        
        #ratings_pc | 'print ratings_pc' >> beam.Map(lambda x: print(f'rpc={x}'))
        
        # create PC of "movie_id", "1", "2", "3", "4", "5"
        movie_key = 1
        rating_key = 2
        keyed_pc = ratings_pc | f'MapToKeyedRatings' >> beam.Map(
            lambda element: (
                int(element[movie_key]),
                (1, 0, 0, 0, 0) if element[rating_key] == "1" else
                (0, 1, 0, 0, 0) if element[rating_key] == "2" else
                (0, 0, 1, 0, 0) if element[rating_key] == "3" else
                (0, 0, 0, 1, 0) if element[rating_key] == "4" else
                (0, 0, 0, 0, 1) if element[rating_key] == "5" else
                (0, 0, 0, 0, 0)  # Handle other ratings safely
            )
        )
        #keyed_pc | 'print keyed_pc' >> beam.Map( lambda x: print(f'kpc={x}'))
        
        # each keyed element is like: keyed=(3430, (0, 0, 0, 1, 0))
        combined_pc = (
            keyed_pc | f'CombineByMovie' >> beam.CombinePerKey(_PivotCombineFn()))
        
        #combined_pc | 'print combined_pc' >> beam.Map(lambda x: print(f'cpc={x}'))
        
        # each pivoted row is like {'movie_id': 7122, '1': 30, '2': 125, '3': 431, '4': 838, '5': 699}
        pivoted = (
            combined_pc | 'MapToPivotSchema'
            >> beam.Map(
                lambda kv: {
                    'movie_id': kv[0], '1': kv[1][0], '2': kv[1][1],
                    '3': kv[1][2], '4': kv[1][3], '5': kv[1][4],
                }
            ))
        
        #pivoted | 'print pivoted' >> beam.Map( lambda x: print(f'pivoted={x}'))
        
        ## ==== create entries for movies not rated =====
        
        ## read in the movies as movie_id only, filter to keep only the movie_id, while making a key value tuple needed for set difference
        movies_pc = (pipeline | f"read_movies.dat" >>
            beam.io.ReadFromText(self.input_path1, skip_header_lines=0,
            coder=CustomUTF8Coder())
            | f'parse_movie_from_movies_.dat' >> beam.Map(lambda line: line.split("::"))
            | 'filter_movies_to_id_only' >> beam.Map(lambda row: (int(row[0]), None)))
        
        ratings_movies_pc = ratings_pc | 'filter_ratings_to_id_only' >> beam.Map(lambda row: (int(row[1]), None))
        grouped = {'p1': movies_pc, 'p2': ratings_movies_pc} | beam.CoGroupByKey()
        # entry is (movie_id, {'p1': [None], 'p2': []})
        unrated_movies_pc = (grouped | "Find Unique to p1" >> beam.Filter(
            lambda x: x[1]['p1'] and not x[1]['p2'])
            | "Get Movie ID" >> beam.Map(lambda x: x[0])
        )
        #debug:
        movie_count = unrated_movies_pc | "Count Movies" >> beam.combiners.Count.Globally()
        movie_count | "Print Count" >> beam.Map(lambda x: print(f'count of movies not rated = {x}'))

        #write {'movie_id': 7122, '1': 0, '2': 0, '3': 0, '4': 0, '5': o}
        unrated_pivoted = (unrated_movies_pc | 'write_unrated_movies' >>
            beam.Map(lambda x: {'movie_id' : x, '1':0, '2':0, '3':0, '4':0, '5':0}))
        
        final_pivot = (pivoted, unrated_pivoted) | "merge_rated_and_unrated_pivots" >> beam.Flatten()

        #final_pivot | "print_final_pivot" >> beam.Map(lambda x: print(f'final pivot = {x}'))
        
        # write to tfrecords file
        (final_pivot | f'pivot_ToTFExample_ser_{random.randint(0, 1000000000000)}'
            >> beam.ParDo(_PivotExampleMaker())
            | f"write_merged_ratings_pivot_to_tfrecord"
            >> beam.io.tfrecordio.WriteToTFRecord(
                    file_path_prefix=f"{dir_path}/{file_name}.tfrecord",
                    file_name_suffix='.gz'))
        
        # write to array_record
        (final_pivot | f"serialize_merged_ratings_pivot_with_msgpack_{file_name}"
            >> beam.Map(msgpack.packb)
            | f'write_array_record_{file_name}'>> arrayrecordio.WriteToArrayRecord(
            file_path_prefix=f"{dir_path}/{file_name}.array_record", num_shards=1))
        
        # <apache_beam.runners.portability.fn_api_runner.fn_runner.RunnerResult object
        result = pipeline.run()
        result.wait_until_finish()
        
        # ==== assert file contents ====
        feature_spec = {
            'movie_id': tf.io.FixedLenFeature([], tf.int64),
            '1': tf.io.FixedLenFeature([], tf.int64),
            '2': tf.io.FixedLenFeature([], tf.int64),
            '3': tf.io.FixedLenFeature([], tf.int64),
            '4': tf.io.FixedLenFeature([], tf.int64),
            '5': tf.io.FixedLenFeature([], tf.int64),
        }
        
        def _parse_function(example_proto):
            return tf.io.parse_single_example(example_proto, feature_spec)
        
        t = f"{dir_path}/{file_name}.tfrecord*"
        file_paths = glob.glob(t)
        dataset = tf.data.TFRecordDataset(file_paths, compression_type='GZIP')
        parsed_dataset = dataset.map(_parse_function)
        t = None
        for x in parsed_dataset.batch(1):
            t = x
            for key in ['movie_id', '1', '2', '3', '4', '5']:
                self.assertTrue(key in x)
                val = x[key].numpy()[0]
                #print(f'type val={type(val)}')
                self.assertTrue(isinstance(val, np.int64))
            break
        self.assertIsNotNone(t)
        
        ## assert array_record
        t = f"{dir_path}/{file_name}.array_record*"
        file_paths = glob.glob(t)
        outfile_path = f"{dir_path}/{file_name}.array_record"
        shutil.move(file_paths[0], outfile_path)
        reader = None
        try:
            reader = array_record_module.ArrayRecordReader(outfile_path)
            record: list = msgpack.unpackb(reader.read(), use_list=False)
            for key in ['movie_id', '1', '2', '3', '4', '5']:
                self.assertTrue(key in record)
                self.assertTrue(isinstance(record[key], int))
        except Exception as ex:
            raise ex
        finally:
            if reader is not None:
                reader.close()
    
    def _find_mean_timestamp(self):
        file_path = self.train_full_uri
        
        #create temporary file, will receive results of all workers
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
            
        pipeline = beam.Pipeline(options=self.pipeline_options)
        
        examples = (pipeline | "read_full_train" >>
            beam.io.ReadFromText(file_path, skip_header_lines=0,coder=CustomUTF8Coder())
            | 'parse_full_train' >> beam.Map(lambda line: line.split("::"))
        )
        mean_timestamp = (
            examples | "Extract Timestamps" >> beam.Map(lambda x: int(x[3]))
            | "Calculate Mean" >> beam.CombineGlobally(beam.combiners.MeanCombineFn())
        )
        mean_timestamp | 'print: ' >> beam.Map(lambda x : print(f'X={x}'))
        mean_timestamp | 'WriteResult' >> beam.io.WriteToText(temp_path, shard_name_template='')
        
        result = pipeline.run()
        result.wait_until_finish()
        if os.path.exists(temp_path):
            with open(temp_path, 'r') as f:
                val = f.read().strip()
                final_mean = float(val) if val else None
            os.remove(temp_path)
        return int(final_mean) if final_mean else None

class _FakeJoinedRatingsExampleMaker(beam.DoFn):
    def __init__(self, inp_type: str, timestamp:int):
        self.inp_type = inp_type
        self.timestamp = timestamp
        self.outp_column_name_type_list = None
        self.inp_column_name_type_list = None
    
    def setup(self):
        """
        Called once per DoFn instance (per worker process) after being unpickled.
        This is where you load heavy, non-serializable resources.
        """
        self.outp_column_name_type_list = [('genres', str), ('age', int),
            ('gender', str),
            ('movie_id', int), ('occupation', int), ('rating', int),
            ('timestamp', int), ('user_id', int)]
        if self.inp_type == "movie":
            self.inp_column_name_type_list = [('movie_id', int),
                ('movie_title', str),
                ('genres', str)]
        elif self.inp_type == "user":
            self.inp_column_name_type_list = [('user_id', int), ('gender', str),
                ('age', int), ('occupation', int), ('zipcode', str)]
        else:
            raise Exception(f"unknown inp_type {self.inp_type}")
    
    def process(self, row):
        #ROW=['6041', 'Toy Story (1995)', "Animation|Children's|Comedy"]
        final_keys = set([key for key, type in self.outp_column_name_type_list])
        feature_map = {}
        for i, value in enumerate(row):
            try:
                element_type = self.inp_column_name_type_list[i][1]
                name = self.inp_column_name_type_list[i][0]
                if name not in final_keys:
                    continue
                if element_type == float:
                    f = tf.train.Feature(
                        float_list=tf.train.FloatList(value=[float(value)]))
                elif element_type == int or element_type == bool:
                    f = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[int(value)]))
                elif element_type == str:
                    f = tf.train.Feature(bytes_list=tf.train.BytesList(
                        value=[value.encode('utf-8')]))
                else:
                    raise ValueError(
                        f"element_type={element_type}, but only float, int, and str classes are handled.")
                feature_map[name] = f
            except Exception as ex:
                logging.error(
                    f"ERROR: {ex}\nrow={row}, name={name}, element_type={element_type}"
                    f"\ni={i}\ncolumn_name_type_list={self.inp_column_name_type_list}")
                raise ex
        # add fake entries to make consistent with the joined ratings file columns
        for out_name, out_type in self.outp_column_name_type_list:
            try:
                if out_name in feature_map:
                    continue
                if out_type == float:
                    f = tf.train.Feature(float_list=tf.train.FloatList(value=[0.0]))
                elif out_type == int or element_type == bool:
                    if out_name == "timestamp":
                        value = self.timestamp
                    else:
                        value = 1
                    f = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
                elif out_type == str:
                    if out_name == "genres":
                        value = b"Drama"
                    elif out_name == "gender":
                        value = random.choice([b"M", b"F"])
                    else:
                        value = b""
                    f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
                else:
                    raise ValueError(f"out_type={out_type}, but only float, int, and str classes are handled.")
                feature_map[out_name] = f
            except Exception as ex:
                logging.error(
                    f"ERROR2: {ex}\nrow={row}, out_name={out_name}, out_type={out_type}")
                raise ex
        yield tf.train.Example(features=tf.train.Features(feature=feature_map))
        #ParDo is a 1 to many mapping so yield each example

class _CandidateEmbeddingMaker(beam.DoFn):
    def __init__(self, saved_model_path: str):
        self.saved_model_path = saved_model_path
        self.serving_default = None  # pickling error, so delay until setup
        self.feature_spec = {"movie_id": tf.io.FixedLenFeature([], tf.int64)}
        self.INPUT_KEY = None
    
    def setup(self):
        """
        Called once per DoFn instance (per worker process) after being unpickled.
        This is where you load heavy, non-serializable resources.
        """
        loaded_user_movie_model = tf.saved_model.load(self.saved_model_path)
        self.serving_default = loaded_user_movie_model.signatures["serving_candidate"]
        self.INPUT_KEY = list(self.serving_default.structured_input_signature[1].keys())[0]
        
    def process(self, example_ser):
        pred = self.serving_default(**{self.INPUT_KEY: [example_ser]})[ 'outputs']
        parsed_features = tf.io.parse_single_example(example_ser, self.feature_spec)
        #parsed_features={'movie_id': <tf.Tensor: shape=(), dtype=int64, numpy=6041>}
        m_id = int(parsed_features['movie_id'].numpy())
        pred = pred.numpy()[0].tolist()
        yield (m_id, pred)

class _QueryEmbeddingMaker(beam.DoFn):
    def __init__(self, saved_model_path: str):
        """
        param: saved_model_path: path to saved model
        """
        self.saved_model_path = saved_model_path
        self.serving_default = None  # pickling error, so delay until setup
        self.feature_spec = {"user_id": tf.io.FixedLenFeature([], tf.int64)}
        self.INPUT_KEY = None

    def setup(self):
        """
        Called once per DoFn instance (per worker process) after being unpickled.
        This is where you load heavy, non-serializable resources.
        """
        loaded_user_movie_model = tf.saved_model.load(self.saved_model_path)
        self.serving_default = loaded_user_movie_model.signatures["serving_query"]
        self.INPUT_KEY = list(self.serving_default.structured_input_signature[1].keys())[0]
        
    def process(self, example_ser):
        pred = self.serving_default(**{self.INPUT_KEY: [example_ser]})[ 'outputs']
        parsed_features = tf.io.parse_single_example(example_ser, self.feature_spec)
        # parsed_features['user_id_id'] is a tensor like: < tf.Tensor: shape=(1,), dtype = int64, numpy = array([7])
        # emb is tensor like: tf.Tensor: shape = (1, 32), dtype = float32, numpy = array([[ 0.08...
        u_id = int(parsed_features['user_id'].numpy())
        pred = pred.numpy()[0].tolist()
        yield (u_id, pred)

class _PivotExampleMaker(beam.DoFn):
    def __init__(self):
        self.inp_column_name_type_dict = None
    
    def setup(self):
        """
        Called once per DoFn instance (per worker process) after being unpickled.
        This is where you load heavy, non-serializable resources.
        """
        self.inp_column_name_type_dict = {'movie_id': int, '1': int, '2': int,
            '3': int, '4': int, '5': int}
    
    def process(self, row:Dict):
        feature_map = {}
        for key, value in row.items():
            try:
                element_type = self.inp_column_name_type_dict[key]
                if element_type == float:
                    feature_map[key] = tf.train.Feature(
                        float_list=tf.train.FloatList(value=[float(value)]))
                elif element_type == int or element_type == bool:
                    feature_map[key] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[int(value)]))
                elif element_type == str:
                    feature_map[key] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[value.encode('utf-8')]))
                else:
                    raise ValueError(
                        f"element_type={element_type}, but only float, int, and str classes are handled.")
            except Exception as ex:
                print(
                    f'ERROR: row={row},\nkey={key}, value={value}, type of value={type(value)}\n'
                    f' element_type={element_type}\nEXCEPTION={ex}')
                #alternatively, could instead of raise ex:
                #yield beam.pvalue.TaggedOutput('errors', (row, str(e)))
                raise ex
        example = tf.train.Example(
            features=tf.train.Features(feature=feature_map))
        yield example.SerializeToString()
        
class _ParseSerializedExamples(beam.DoFn):
    """
    converts string serialized examples into dictionaries of tensors, then converts
    the dictionary to dictionary of scalars.
    """
    
    def __init__(self, feature_spec_ser: str):
        self.feature_spec_ser = feature_spec_ser
        self.feature_spec = None
    
    def setup(self):
        self.feature_spec = deserialize(self.feature_spec_ser)
    
    def process(self, example_ser):
        tensor_dict = tf.io.parse_single_example(example_ser,
            self.feature_spec)
        # beam.CoGroupByKey cannot handle tensors (unless there is some way with a type hint)
        # so extract scalars from tensors
        for key in tensor_dict:
            val = tensor_dict[key].numpy()
            if isinstance(val, np.floating):
                tensor_dict[key] = float(val)
            else:
                tensor_dict[key] = int(val)
        yield tensor_dict

def create_serialized_example_for_movies(element):
    movie_id = int(element[0])
    title = element[1].encode('utf-8')
    genres = element[2].encode('utf-8')
    feature = {
        'movie_id': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[movie_id])),
        'title': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[title])),
        'genres': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[genres])),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def create_serialized_example_for_users(element):
    user_id = int(element[0])
    gender = element[1].encode('utf-8')
    age = int(element[2])
    occupation = int(element[3])
    zipcode = element[4].encode('utf-8')
    feature = {
        'user_id': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[user_id])),
        'gender': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[gender])),
        'age': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[age])),
        'occupation': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[occupation])),
        'zipcode': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[zipcode])),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def create_serialized_example_for_ratings_dat(element):
    user_id = int(element[0])
    movie_id = int(element[1])
    rating = int(element[2])
    timestamp = int(element[3])
    feature = {
        'user_id': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[user_id])),
        'movie_id': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[movie_id])),
        'rating': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[rating])),
        'timestamp': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[timestamp])),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


class _PivotCombineFn(beam.CombineFn):
    def create_accumulator(self):
        return (0, 0, 0, 0, 0)
    
    def add_input(self, accumulator, input_rating_count_tuple):
        c1, c2, c3, c4, c5 = accumulator
        i1, i2, i3, i4, i5 = input_rating_count_tuple
        return (c1 + i1, c2 + i2, c3 + i3, c4 + i4, c5 + i5)
    
    def merge_accumulators(self, accumulators):
        c1_sum, c2_sum, c3_sum, c4_sum, c5_sum = 0, 0, 0, 0, 0
        for c1, c2, c3, c4, c5 in accumulators:
            c1_sum += c1
            c2_sum += c2
            c3_sum += c3
            c4_sum += c4
            c5_sum += c5
        return (c1_sum, c2_sum, c3_sum, c4_sum, c5_sum)
    
    def extract_output(self, accumulator):
        return accumulator
    
def convert_tensor_to_scalar(row: Dict[str, tf.Tensor]) -> Dict[
    str, Any]:
    transformed_row = {}
    for key, tensor in row.items():
        try:
            transformed_row[key] = tensor.numpy().item()
        except Exception as e:
            if tensor.shape.rank > 0:
                transformed_row[key] = str(tensor.numpy().tolist())
            else:
                raise ValueError(
                    f"Failed to convert key '{key}' from Tensor to scalar: {e}")
    return transformed_row

def dict_to_csv_string(row_dict: Dict[str, Any],
        column_order: List[str]) -> str:
    values = [row_dict.get(col, '') for col in column_order]
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(values)
    return output.getvalue().strip()
