'''
writes all the ratings data in src/test/resources to other formats such as array_record and parquet.

writes the movie and user dat files in src/main/resources to other formats such as array_record and parquet.

the output is bin/other_formats

example USAGE from CLI:
export OUTPUT_BASE_DIR="../TMP_DATA"
python3 -m unittest src.test.python.movie_lens_tfx.write_ml1m_to_other_formats.WriteOtherFormats

'''
import unittest
from collections import OrderedDict

import msgpack
import polars as pl
import os
import io
import glob
from pathlib import Path
import shutil
import sys

is_cli = __name__ == '__main__' or any('unittest' in arg for arg in sys.argv)
if is_cli:
    sys.path.insert(0,
        os.path.join(os.getcwd(), "src/test/python/movie_lens_tfx"))
    sys.path.insert(0,
        os.path.join(os.getcwd(), "src/main/python/movie_lens_tfx"))

from helper import *
from array_record.python import array_record_module

pl.Config.set_fmt_str_lengths(900)


class WriteOtherFormats(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):

        output_base_dir = os.environ.get("OUTPUT_BASE_DIR", "")
        if output_base_dir.strip() == "":
            output_base_dir = get_bin_dir()
        
        out_dir = os.path.join(output_base_dir, "other_formats")
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)
        
        cls.out_dir = out_dir
        
    @classmethod
    def tearDownClass(cls):
        print(f'wrote to {cls.out_dir}')
        

    def test_write_ratings(self):
        
        ratings_schema = pl.Schema(OrderedDict({
            'user_id': pl.Int32,
            'movie_id': pl.Int32,
            'rating': pl.Int32,
            'timestamp': pl.Int64}))
        
        input_dir = os.path.join(get_project_dir(), "src/test/resources")
        subdirs = ["ml-1m", "ml-1m/small", "ml-1m/tiny", "ml-1m/tiny3"]
        
        for subdir in subdirs:
            
            out_dir_full = os.path.join(WriteOtherFormats.out_dir, subdir)
            os.makedirs(out_dir_full, exist_ok=True)
            
            file_paths = glob.glob(os.path.join(input_dir, subdir, "ratings*.dat"))
            for file_path in file_paths:
                
                file_name = Path(file_path).parts[-1]
                prefix = file_name[0:file_name.find(".dat")]
                
                processed_buffer = io.StringIO()
                
                with open(file_path, "r", encoding='iso-8859-1') as file:
                    for line in file:
                        line2 = line.replace('::', '\t')
                        processed_buffer.write(line2)
                    
                    processed_buffer.seek(0)
                    
                    df = pl.read_csv(processed_buffer,
                        encoding='iso-8859-1', has_header=False,
                        skip_rows=0, separator='\t', schema=ratings_schema,
                        try_parse_dates=True,
                        new_columns=ratings_schema.names(),
                        schema_overrides=ratings_schema,
                        use_pyarrow=True)
                    
                    write_to_array_record(df, os.path.join(out_dir_full, f'{prefix}.array_record'))
                    df.write_parquet(os.path.join(out_dir_full, f'{prefix}.parquet'))
                
    def test_write_movies(self):
        
        schema = pl.Schema(OrderedDict({'movie_id': pl.Int64,
            'title': pl.String, 'genres': pl.String}))
        
        input_dir = os.path.join(get_project_dir(), "src/main/resources")
        
        file_path = os.path.join(input_dir, "ml-1m", "movies.dat")
        
        file_name = Path(file_path).parts[-1]
        prefix = file_name[0:file_name.find(".dat")]
        out_dir_full = os.path.join(WriteOtherFormats.out_dir, "ml-1m")
        
        if not os.path.exists(out_dir_full):
            os.makedirs(out_dir_full)
        
        processed_buffer = io.StringIO()
    
        with open(file_path, "r", encoding='iso-8859-1') as file:
            for line in file:
                line2 = line.replace('::', '\t')
                processed_buffer.write(line2)
            
            processed_buffer.seek(0)
            
            df = pl.read_csv(processed_buffer,
                encoding='iso-8859-1', has_header=False,
                skip_rows=0, separator='\t', schema=schema,
                try_parse_dates=True,
                new_columns=schema.names(),
                schema_overrides=schema,
                use_pyarrow=True)
            
            write_movies_to_array_record(df,
                os.path.join(out_dir_full, f'{prefix}.array_record'))
            df.write_parquet(
                os.path.join(out_dir_full, f'{prefix}.parquet'))
    
        
    def test_write_users(self):
       
        schema = pl.Schema(OrderedDict({'user_id': pl.Int64,
            'gender': pl.String, 'age': pl.Int64,
            'occupation': pl.Int64,
            'zipcode': pl.String}))
        
        input_dir = os.path.join(get_project_dir(), "src/main/resources")
        
        file_path = os.path.join(input_dir, "ml-1m", "users.dat")
        
        file_name = Path(file_path).parts[-1]
        prefix = file_name[0:file_name.find(".dat")]
        out_dir_full = os.path.join(WriteOtherFormats.out_dir, "ml-1m")
        
        if not os.path.exists(out_dir_full):
            os.makedirs(out_dir_full)
            
        processed_buffer = io.StringIO()
    
        with open(file_path, "r", encoding='iso-8859-1') as file:
            for line in file:
                line2 = line.replace('::', '\t')
                processed_buffer.write(line2)
            
            processed_buffer.seek(0)
            
            df = pl.read_csv(processed_buffer,
                encoding='iso-8859-1', has_header=False,
                skip_rows=0, separator='\t', schema=schema,
                try_parse_dates=True,
                new_columns=schema.names(),
                schema_overrides=schema,
                use_pyarrow=True)
            
            write_users_to_array_record(df,
                os.path.join(out_dir_full, f'{prefix}.array_record'))
            df.write_parquet(
                os.path.join(out_dir_full, f'{prefix}.parquet'))


def write_to_array_record(df_write: pl.DataFrame, out_file_path: str):
    writer = array_record_module.ArrayRecordWriter(out_file_path, "group_size:1")
    try:
        for user_id, movie_id, rating, timestamp in df_write.select(
                ["user_id", "movie_id", "rating", "timestamp"]
        ).iter_rows():
            # Pack the 4 integers as a tuple into MessagePack bytes
            record_tuple = (int(user_id), int(movie_id), int(rating), int(timestamp))
            packed_bytes = msgpack.packb(record_tuple)
            writer.write(packed_bytes)
    finally:
        writer.close()
        
    ## assert file contents
    reader = array_record_module.ArrayRecordReader(out_file_path)
    try:
        assert reader.num_records() == len(df_write), (
            f"Expected {len(df_write)} records, got {reader.num_records()}"
        )
        
        # spot check with first 5 records
        expected_rows = (
            df_write.select(
                ["user_id", "movie_id", "rating", "timestamp"])
            .head(5)
            .iter_rows()
        )
        for i, (user_id, movie_id, rating, timestamp) in enumerate(
                expected_rows):
            expected = (int(user_id), int(movie_id), int(rating), int(timestamp))
            actual = msgpack.unpackb(reader.read(), use_list=False)
            assert actual == expected, f"Row {i} mismatch: {actual} != {expected}"
    finally:
        reader.close()
        
def write_users_to_array_record(df_write: pl.DataFrame, out_file_path: str):
    writer = array_record_module.ArrayRecordWriter(out_file_path, "group_size:1")
    try:
        for user_id, gender, age, occupation, zipcode in df_write.select(
                ['user_id', 'gender', 'age', 'occupation', 'zipcode']
        ).iter_rows():
            # Pack the 4 integers as a tuple into MessagePack bytes
            record_tuple = (int(user_id), str(gender), int(age),
                int(occupation), str(zipcode))
            packed_bytes = msgpack.packb(record_tuple)
            writer.write(packed_bytes)
    finally:
        writer.close()
        
    ## assert file contents
    reader = array_record_module.ArrayRecordReader(out_file_path)
    try:
        assert reader.num_records() == len(df_write), (
            f"Expected {len(df_write)} records, got {reader.num_records()}"
        )
        
        # spot check with first 5 records
        expected_rows = (
            df_write.select(['user_id', 'gender', 'age', 'occupation', 'zipcode'])
            .head(5)
            .iter_rows()
        )
        for i, (user_id, gender, age, occupation, zipcode) in enumerate(expected_rows):
            expected = (int(user_id), str(gender), int(age),
                int(occupation), str(zipcode))
            actual = msgpack.unpackb(reader.read(), use_list=False)
            assert actual == expected, f"Row {i} mismatch: {actual} != {expected}"
    finally:
        reader.close()


def write_movies_to_array_record(df_write: pl.DataFrame, out_file_path: str):
    writer = array_record_module.ArrayRecordWriter(out_file_path,
        "group_size:1")
    try:
        for movie_id, title, genres in df_write.select(
                ['movie_id', 'title', 'genres']
        ).iter_rows():
            # Pack the 4 integers as a tuple into MessagePack bytes
            record_tuple = (int(movie_id), str(title), str(genres))
            packed_bytes = msgpack.packb(record_tuple)
            writer.write(packed_bytes)
    finally:
        writer.close()
    
    ## assert file contents
    reader = array_record_module.ArrayRecordReader(out_file_path)
    try:
        assert reader.num_records() == len(df_write), (
            f"Expected {len(df_write)} records, got {reader.num_records()}"
        )
        
        # spot check with first 5 records
        expected_rows = (
            df_write.select(['movie_id', 'title', 'genres'])
            .head(5)
            .iter_rows()
        )
        for i, (movie_id, title, genres) in enumerate(expected_rows):
            expected = (int(movie_id), str(title), str(genres))
            actual = msgpack.unpackb(reader.read(), use_list=False)
            assert actual == expected, f"Row {i} mismatch: {actual} != {expected}"
    finally:
        reader.close()


if __name__ == '__main__':
    unittest.main()