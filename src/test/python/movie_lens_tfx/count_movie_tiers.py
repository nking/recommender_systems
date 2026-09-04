import collections
import io
import json
import os.path
import unittest
import glob
from collections import defaultdict
from typing import Any, Dict, Union, OrderedDict

import polars as pl
import numpy as np

import msgpack
from array_record.python import array_record_module

from helper import *

class ExploreMovieTiers(unittest.TestCase):
    def setUp(self):
        
        self.n_movies = 3883
        self.MOVIE_OFFSET = 6040 + 1
        
        movie_tiers_path = os.path.join(get_project_dir(),
            "src/test/resources/movie_tiers.json")
        self.movie_tiers_df = pl.read_ndjson(movie_tiers_path)
    
    def read_ratings_into_df(self, file_path: str) -> pl.DataFrame:
        schema = pl.Schema(OrderedDict({'user_id': pl.Int64,
            'movie_id': pl.Int64, 'rating': pl.Int64, 'timestamp': pl.Int64}))
        
        processed_buffer = io.StringIO()
        # print(f"key={key}, file_path={file_path}")
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
            use_pyarrow=True)
        return df
    
    def test_0(self):
        
        df_train_ratings = self.read_ratings_into_df(
            os.path.join(get_project_dir(), 'src/test/resources/ml-1m/ratings_train_liked.dat'))
        df_train_ratings = df_train_ratings.join(self.movie_tiers_df, on="movie_id", how="left")
        df_val_ratings = self.read_ratings_into_df(
            os.path.join(get_project_dir(), 'src/test/resources/ml-1m/ratings_val_liked.dat'))
        df_val_ratings = df_val_ratings.join(self.movie_tiers_df, on="movie_id", how="left")

        common_movie_ids = (
            df_train_ratings.select("movie_id").unique()
            .join(df_val_ratings.select("movie_id").unique(), on="movie_id", how="inner")
            .get_column("movie_id")
        )
        print(f"unique movies intersection of train and val: {len(common_movie_ids)}")
        
        df_train_ratings_inter = df_train_ratings.filter(
            pl.col("movie_id").is_in(common_movie_ids)
        )
        df_val_ratings_inter = df_val_ratings.filter(
            pl.col("movie_id").is_in(common_movie_ids)
        )
        print(f"ratings in train intersect by movies={df_train_ratings_inter['movie_id'].count()}")
        print(f"ratings in val intersect by movies={df_val_ratings_inter['movie_id'].count()}")
        
        df_train_ratings_inter_2 = df_train_ratings_inter.filter(
            pl.col("tier")==2
        )
        df_val_ratings_inter_2 = df_val_ratings_inter.filter(
            pl.col("tier")==2
        )
        print(
            f"ratings in train intersect by movies for tier=2={df_train_ratings_inter_2['movie_id'].count()}")
        print(
            f"ratings in val intersect by movies for tier=2={df_val_ratings_inter_2['movie_id'].count()}")
        
        ## filter by common user_ids
        common_user_ids = (
            df_train_ratings_inter.select("user_id").unique()
            .join(df_val_ratings_inter.select("user_id").unique(), on="user_id",
                how="inner")
            .get_column("user_id")
        )
        print(f"unique users intersection of train and val after unique movies intersection: {len(common_user_ids)}")
    
        df_train_ratings_inter_u = df_train_ratings_inter.filter(
            pl.col("user_id").is_in(common_user_ids)
        )
        df_val_ratings_inter_u = df_val_ratings_inter.filter(
            pl.col("user_id").is_in(common_user_ids)
        )
        print(
            f"ratings in train intersect by movies then users={df_train_ratings_inter_u['movie_id'].count()}")
        print(
            f"ratings in val intersect by movies then users={df_val_ratings_inter_u['movie_id'].count()}")
        
        df_train_ratings_inter_u_2 = df_train_ratings_inter_u.filter(
            pl.col("tier") == 2
        )
        df_val_ratings_inter_u_2 = df_val_ratings_inter_u.filter(
            pl.col("tier") == 2
        )
        print(
            f"ratings in train intersect by movies then users for tier=2={df_train_ratings_inter_u_2['movie_id'].count()}")
        print(
            f"ratings in val intersect by movies then users for tier=2={df_val_ratings_inter_u_2['movie_id'].count()}")
        
        #count the unique movies and unique users in the later 2
        common_movie_ids = (
            df_train_ratings_inter_u_2.select("movie_id").unique()
            .join(df_val_ratings_inter_u_2.select("movie_id").unique(), on="movie_id",
                how="inner")
            .get_column("movie_id")
        )
        print(
            f"unique movies in intersection of train and val by movies then users for tier=2: {len(common_movie_ids)}")
        
        common_user_ids = (
            df_train_ratings_inter_u_2.select("user_id").unique()
            .join(df_val_ratings_inter_u_2.select("user_id").unique(),
                on="user_id",
                how="inner")
            .get_column("user_id")
        )
        print(
            f"unique users in intersection of train and val by movies then users for tier=2: {len(common_user_ids)}")
        
    if __name__ == '__main__':
        unittest.main()
