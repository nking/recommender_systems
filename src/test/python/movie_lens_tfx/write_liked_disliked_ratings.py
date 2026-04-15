'''
reads the ratings_train,.dat, ratings_val.dat, and ratings_test.dat
and filters them for ratings > 3 and writes the files with _liked added to name
'''
from collections import OrderedDict

from helper import *
import polars as pl
import os
import io

schema = pl.Schema(OrderedDict({'user_id': pl.Int64,
    'movie_id': pl.Int64, 'rating': pl.Int64,
    'timestamp' : pl.Int64}))

file_names = ["ratings_train", "ratings_val", "ratings_test"]

for file_name in file_names:
    in_file_path = os.path.join(get_project_dir(), f"src/main/resources/ml-1m/{file_name}.dat")
    out_file_path = os.path.join(get_bin_dir(), f"{file_name}_liked.dat")
    out_file_path2 = os.path.join(get_bin_dir(), f"{file_name}_disliked.dat")
    
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
        
        df_formatted = df_liked.select(
            pl.format("{}::{}::{}::{}",
                pl.col("user_id"),
                pl.col("movie_id"),
                pl.col("rating"),
                pl.col("timestamp")).alias("output")
        )
        df_formatted.write_csv(
            out_file_path,
            include_header=False,
            quote_style="never"
        )
        
        df_disliked = df.filter(pl.col("rating") < 3)
        print(f'# disliked = {len(df_disliked)} out of {len(df)}')
        
        df_formatted = df_disliked.select(
            pl.format("{}::{}::{}::{}",
                pl.col("user_id"),
                pl.col("movie_id"),
                pl.col("rating"),
                pl.col("timestamp")).alias("output")
        )
        df_formatted.write_csv(
            out_file_path2,
            include_header=False,
            quote_style="never"
        )
        