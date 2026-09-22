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
    
    def read_all_ratings_into_df(self):
        ratings = []
        for t1 in ("train", "val", "test"):
            for t2 in ("liked", "3", "disliked"):
                tmp = self.read_ratings_into_df(
                    os.path.join(get_project_dir(),
                    f'src/test/resources/ml-1m/ratings_{t1}_{t2}.dat'))
                ratings.append(tmp)
        return pl.concat(ratings)
        
    def test_explore_movie_tier_stratification_for_splits(self):
        df = self.read_all_ratings_into_df()
        df = df.join(self.movie_tiers_df, on="movie_id", how="left")
        df = df.rename({"tier": "movie_tier"})
        
        #split into partitions
        df = df.sort(["user_id", "timestamp"])
        
        # Calculate per-user sequence index and total ratings count
        df = df.with_columns(
            user_seq=pl.cum_count("rating").over("user_id") - 1,
            user_total=pl.len().over("user_id")
        )
        
        # filter out users with less than 30 ratings and movies with less than 30 ratings
        # until convergence (since removing a user can drop a movie below 30 and vice versa)
        prev_len = 0
        while len(df) != prev_len:
            prev_len = len(df)
            df = df.filter(
                (pl.count("rating").over("user_id") >= 30) &
                (pl.count("rating").over("movie_id") >= 30)
            )
        
        # Compute fractional position to execute an 80:10:10 temporal split per user
        df = df.with_columns(
            fraction=pl.col("user_seq") / pl.col("user_total")
        )
        
        # trying psotive partition size: 539444, 10000, 10000  for ths positives
        total_pos_len = (df.filter(pl.col("rating") > 3))['user_id'].count()
        #NOTE: change this to total_pos_len - 2 * 50_000 to print current split stats
        train_pos_len = total_pos_len - 2 * 10_000
        
        #train partition
        last_p = None
        for p in np.arange(0.8, 1.0, 0.01):
            tmp = df.filter(pl.col("fraction") < p)
            if (tmp.filter(pl.col("rating")>3))['user_id'].count() > train_pos_len:
                break
            last_p = p
        print(f"p={last_p}")
        df_train = df.filter(pl.col("fraction") < last_p)
        
        #(1 - last_p)/2.
        next_p = last_p + ( (1.-last_p)/2.)
        df_val = df.filter((pl.col("fraction") >= last_p) & (pl.col("fraction") < next_p))
        df_test = df.filter(pl.col("fraction") > next_p)
        for tmp in (df_train, df_val, df_test):
            print(f'number of positive ratings = {(tmp.filter(pl.col("rating")>3))['user_id'].count()}')
        
        # ======================================================================================
        #count intersection of movies:  train and val
        common_movie_ids = (
            df_train.select("movie_id").unique()
            .join(df_val.select("movie_id").unique(), on="movie_id",
                how="inner")
            .get_column("movie_id")
        )
        print(
            f"unique movies intersection of train and val: {len(common_movie_ids)}")
        df_train_ratings_inter = df_train.filter(
            pl.col("movie_id").is_in(common_movie_ids)
        )
        df_val_ratings_inter = df_val.filter(
            pl.col("movie_id").is_in(common_movie_ids)
        )
        print(
            f"ratings in train intersect by movies={df_train_ratings_inter['movie_id'].count()}")
        print(
            f"ratings in val intersect by movies={df_val_ratings_inter['movie_id'].count()}")
        
        #intersection of movies: train and test
        common_movie_ids = (
            df_train.select("movie_id").unique()
            .join(df_test.select("movie_id").unique(), on="movie_id",
                how="inner")
            .get_column("movie_id")
        )
        print(
            f"unique movies intersection of train and test: {len(common_movie_ids)}")
        df_train_ratings_inter = df_train.filter(
            pl.col("movie_id").is_in(common_movie_ids)
        )
        df_test_ratings_inter = df_test.filter(
            pl.col("movie_id").is_in(common_movie_ids)
        )
        print(
            f"ratings in train intersect by movies={df_train_ratings_inter['movie_id'].count()}")
        print(
            f"ratings in test intersect by movies={df_test_ratings_inter['movie_id'].count()}")
        
        ## count the tiers
        for tier in range(0, 3):
            tmp = df_test_ratings_inter.filter(pl.col("movie_tier") == tier)
            count_ratings = tmp['movie_id'].count()
            count_unique_movies = tmp['movie_id'].unique().count()
            print(f"test intersect by movies, movie_tier={tier} #ratings={count_ratings}, #unique_movies={count_unique_movies                                                           }")
        
        # ======================================================================================
        # count intersection of users:  train and val
        user_tiers_df = self.get_user_tiers_df(df_train.filter(pl.col("rating") >= 3))
        df_train = df_train.join(user_tiers_df, on="user_id", how="left")
        df_val = df_val.join(user_tiers_df, on="user_id", how="left")
        df_test = df_test.join(user_tiers_df, on="user_id", how="left")
        
        common_user_ids = (
            df_train.select("user_id").unique()
            .join(df_val.select("user_id").unique(), on="user_id",
                how="inner")
            .get_column("user_id")
        )
        print(
            f"unique users intersection of train and val: {len(common_user_ids)}")
        df_train_ratings_inter = df_train.filter(
            pl.col("user_id").is_in(common_user_ids)
        )
        df_val_ratings_inter = df_val.filter(
            pl.col("user_id").is_in(common_user_ids)
        )
        print(
            f"ratings in train intersect by users={df_train_ratings_inter['user_id'].count()}")
        print(
            f"ratings in val intersect by users={df_val_ratings_inter['user_id'].count()}")
        
        # intersection of movies: train and test
        common_user_ids = (
            df_train.select("user_id").unique()
            .join(df_test.select("user_id").unique(), on="user_id",
                how="inner")
            .get_column("user_id")
        )
        print(
            f"unique users intersection of train and test: {len(common_user_ids)}")
        df_train_ratings_inter = df_train.filter(
            pl.col("user_id").is_in(common_user_ids)
        )
        df_test_ratings_inter = df_test.filter(
            pl.col("user_id").is_in(common_user_ids)
        )
        print(
            f"ratings in train intersect by users={df_train_ratings_inter['user_id'].count()}")
        print(
            f"ratings in test intersect by users={df_test_ratings_inter['user_id'].count()}")
        
        ## count the tiers
        for tier in range(0, 3):
            tmp = df_test_ratings_inter.filter(pl.col("user_tier") == tier)
            count_ratings = tmp['user_id'].count()
            count_unique = tmp['user_id'].unique().count()
            print(
                f"test intersect by users, user_tier={tier} #ratings={count_ratings}, #unique_users={count_unique}")
        
        print(f'ratings in train where user_tier=2={(df_train.filter(pl.col("user_tier")==2))['user_id'].count()}')
    
    def test_tail_users_across_all_datasets(self):
        df_train_ratings = self.read_ratings_into_df(
            os.path.join(get_project_dir(),
                'src/test/resources/ml-1m/ratings_train_liked.dat'))
        df_train_ratings = df_train_ratings.join(self.movie_tiers_df,
            on="movie_id", how="left")
        print(f'len(df_train_ratings) = {len(df_train_ratings)}')
        counts_train_unique_users = [
            df_train_ratings.filter(pl.col("tier") == t)
            .select("user_id")
            .n_unique()
            for t in [0, 1, 2]
        ]
        df_train_ratings = df_train_ratings.filter(pl.col("tier") == 2)
        
        df_val_ratings = self.read_ratings_into_df(
            os.path.join(get_project_dir(),
                'src/test/resources/ml-1m/ratings_val_liked.dat'))
        df_val_ratings = df_val_ratings.join(self.movie_tiers_df,
            on="movie_id", how="left")
        print(f'len(df_val_ratings) = {len(df_val_ratings)}')
        counts_val_unique_users= [
            df_val_ratings.filter(pl.col("tier") == t)
            .select("user_id")
            .n_unique()
            for t in [0, 1, 2]
        ]
        df_val_ratings = df_val_ratings.filter(pl.col("tier") == 2)
        
        df_test_ratings = self.read_ratings_into_df(
            os.path.join(get_project_dir(),
                'src/test/resources/ml-1m/ratings_test_liked.dat'))
        df_test_ratings = df_test_ratings.join(self.movie_tiers_df,
            on="movie_id", how="left")
        print(f'len(df_test_ratings) = {len(df_test_ratings)}')
        counts_test_unique_users = [
            df_test_ratings.filter(pl.col("tier") == t)
            .select("user_id")
            .n_unique()
            for t in [0, 1, 2]
        ]
        df_test_ratings = df_test_ratings.filter(pl.col("tier") == 2)
        
        # intersection of train and test
        common_train_test_user_ids = (
            df_train_ratings.select("user_id").unique()
            .join(df_test_ratings.select("user_id").unique(),
                on="user_id",
                how="inner")
            .get_column("user_id")
        )
        df_train_ratings_inter_test_u = df_train_ratings.filter(
            pl.col("user_id").is_in(common_train_test_user_ids)
        )
        df_test_ratings_inter_train_u = df_test_ratings.filter(
            pl.col("user_id").is_in(common_train_test_user_ids)
        )
        
        common_train_val_test_user_ids = (
            df_train_ratings.select("user_id").unique()
            .join(df_val_ratings.select("user_id").unique(),
                on="user_id",
                how="inner")
            .join(df_test_ratings.select("user_id").unique(),
                on="user_id",
                how="inner")
            .get_column("user_id")
        )
        df_train_ratings_inter_val_test_u = df_train_ratings.filter(
            pl.col("user_id").is_in(common_train_val_test_user_ids)
        )
        df_val_ratings_inter_val_train_u = df_val_ratings.filter(
            pl.col("user_id").is_in(common_train_val_test_user_ids)
        )
        df_test_ratings_inter_val_train_u = df_test_ratings.filter(
            pl.col("user_id").is_in(common_train_val_test_user_ids)
        )
        print(f'===============================')
        print(f'counts_train_unique_users by tier={counts_train_unique_users}')
        print(f'counts_val_unique_users by tier ={counts_val_unique_users}')
        print(f'counts_test_unique_users by tier ={counts_test_unique_users}')
        print(f'len(common_train_test_user_ids)={len(common_train_test_user_ids)}')
        print(f'len(df_train_ratings_inter_test_u)={len(df_train_ratings_inter_test_u)}')
        print(f'len(df_test_ratings_inter_train_u)={len(df_test_ratings_inter_train_u)}')
        
        print(f'len(common_train_val_test_user_ids={len(common_train_val_test_user_ids)}')
        print(f'len(df_train_ratings_inter_val_test_u)={len(df_train_ratings_inter_val_test_u)}')
        print(f'len(df_val_ratings_inter_val_train_u)={len(df_val_ratings_inter_val_train_u)}')
        print(f'len(df_test_ratings_inter_val_train_u)={len(df_test_ratings_inter_val_train_u)}')
    
    
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
    
    def get_user_tiers_df(self, ratings_df: pl.DataFrame) -> pl.DataFrame:
        """
        Given a Polars DataFrame with ['user_id', ...],
        returns DataFrame with columns 'user_id', 'user_tier' where tier is 0, 1, or 2 for
             head, torso, and tail of the distribution of the number of users ratings.
        """
        # Count history length per user
        user_counts = ratings_df.group_by("user_id").agg(
            pl.len().alias("history_length")
        )
        
        # Find the exact cutoff lengths based on quantiles
        tail_cutoff_val = user_counts["history_length"].quantile(0.20,
            interpolation="nearest")
        head_cutoff_val = user_counts["history_length"].quantile(0.80,
            interpolation="nearest")
        
        # Map to tiers based on the cutoffs
        user_tiers_df = user_counts.with_columns(
            pl.when(pl.col("history_length") <= tail_cutoff_val)
            .then(2)  # Tail
            .when(pl.col("history_length") >= head_cutoff_val)
            .then(0)  # Head
            .otherwise(1)  # Torso
            .alias("user_tier")
        ).select(["user_id", "user_tier"])
        
        return user_tiers_df
    
    if __name__ == '__main__':
        unittest.main()
