'''
reads the ratings_train,.dat, ratings_val.dat, and ratings_test.dat
and filters them for ratings > 3 and writes the files with _liked added to name
'''
from collections import OrderedDict

from helper import *
import polars as pl
import os
import io

COPY_TO_SRC_TREE = False

schema = pl.Schema(OrderedDict({'user_id': pl.Int64,
    'movie_id': pl.Int64, 'rating': pl.Int64,
    'timestamp' : pl.Int64}))

file_names = ["ratings_train", "ratings_val", "ratings_test"]

in_dir = os.path.join(get_project_dir(), "src/main/resources/ml-1m/")
out_dir = get_bin_dir()
if COPY_TO_SRC_TREE:
    out_dir = os.path.join(get_project_dir(), "src/test/resources/ml-1m/")
out_small_dir = os.path.join(out_dir, "small")
out_tiny_dir = os.path.join(out_dir, "tiny")
os.makedirs(out_small_dir, exist_ok=True)
os.makedirs(out_tiny_dir, exist_ok=True)

for file_name in file_names:
    in_file_path = os.path.join(in_dir, f"{file_name}.dat")
    
    processed_buffer = io.StringIO()
    df = None
    with open(in_file_path, "r", encoding='iso-8859-1') as file:
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

        # TwoTower bi-encoder needs to be trained only with likes, but the downstream models use the full train dataset
        
        df_liked = df.filter(pl.col("rating") > 3)
        print(f'# liked = {len(df_liked)} out of {len(df)}')
        for j in range(3):
            if j == 0:
                df_formatted = df_liked
                outfile = os.path.join(out_dir, f"{file_name}_liked.dat")
            elif j ==1:
                df_formatted = df_liked.head(1000)
                outfile = os.path.join(out_small_dir, f"{file_name}_liked.dat")
            else:
                df_formatted = df_liked.head(100)
                outfile = os.path.join(out_tiny_dir, f"{file_name}_liked.dat")
            df_formatted = df_formatted.select(
                pl.format("{}::{}::{}::{}",
                    pl.col("user_id"),
                    pl.col("movie_id"),
                    pl.col("rating"),
                    pl.col("timestamp")).alias("output")
            )
            df_formatted.write_csv(
                outfile,
                include_header=False,
                quote_style="never"
            )
        
        df_3 = df.filter(pl.col("rating") == 3)
        print(f'# 3 = {len(df_3)} out of {len(df)}')
        for j in range(3):
            if j == 0:
                df_formatted = df_3
                outfile = os.path.join(out_dir, f"{file_name}_3.dat")
            elif j == 1:
                df_formatted = df_3.head(1000)
                outfile = os.path.join(out_small_dir, f"{file_name}_3.dat")
            else:
                df_formatted = df_3.head(100)
                outfile = os.path.join(out_tiny_dir, f"{file_name}_3.dat")
            df_formatted = df_formatted.select(
                pl.format("{}::{}::{}::{}",
                    pl.col("user_id"),
                    pl.col("movie_id"),
                    pl.col("rating"),
                    pl.col("timestamp")).alias("output")
            )
            df_formatted.write_csv(
                outfile,
                include_header=False,
                quote_style="never"
            )
        
        df_disliked = df.filter(pl.col("rating") < 3)
        print(f'# disliked = {len(df_disliked)} out of {len(df)}')
        for j in range(3):
            if j == 0:
                df_formatted = df_disliked
                outfile = os.path.join(out_dir, f"{file_name}_disliked.dat")
            elif j == 1:
                df_formatted = df_disliked.head(1000)
                outfile = os.path.join(out_small_dir, f"{file_name}_disliked.dat")
            else:
                df_formatted = df_disliked.head(100)
                outfile = os.path.join(out_tiny_dir, f"{file_name}_disliked.dat")
            df_formatted = df_formatted.select(
                pl.format("{}::{}::{}::{}",
                    pl.col("user_id"),
                    pl.col("movie_id"),
                    pl.col("rating"),
                    pl.col("timestamp")).alias("output")
            )
            df_formatted.write_csv(
                outfile,
                include_header=False,
                quote_style="never"
            )
