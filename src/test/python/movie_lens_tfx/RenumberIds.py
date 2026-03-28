'''
User ids are alreay numbered 1 to nUsers without gaps.
This script renumbers the movie ids to start at nUsers and end at nUsers + nMovies.

Downstream models such as the Ranker need the numbers to be consecutive
and disjoint as indices for a cross encoder matrix.
'''
from typing import OrderedDict

from helper import *
import polars as pl
import os
import io

N_USERS = 6040

#read in ml-1m-orig/movies.dat and write out to ml-1m/movies.dat
#read in ml-1m-orig/ratings.dat and write out to ml-1m/ratings.dat
#write movieids dict out
pl.Config.set_fmt_str_lengths(900)

schemas = {
  'ratings' : pl.Schema(OrderedDict({'user_id': pl.Int64,
    'movie_id_orig': pl.Int64, 'rating': pl.Int64,
    'timestamp' : pl.Int64})),
  'users' : pl.Schema(OrderedDict({'user_id': pl.Int64,
    'gender': pl.String, 'age': pl.Int64,
    'occupation' : pl.Int64,
    'zipcode' : pl.String})),
  'movies' : pl.Schema(OrderedDict({'movie_id_orig': pl.Int64,
    'title': pl.String, 'genres': pl.String}))}

file_path = os.path.join(get_project_dir(), "src/main/resources/ml-1m-orig/movies.dat")
processed_buffer = io.StringIO()
df = None
with open(file_path, "r", encoding='iso-8859-1') as file:
    for line in file:
        line2 = line.replace('::', '\t')
        processed_buffer.write(line2)
    processed_buffer.seek(0)
    df = pl.read_csv(processed_buffer,
        encoding='iso-8859-1', has_header=False,
        skip_rows=0, separator='\t', schema=schemas['movies'],
        try_parse_dates=True,
        new_columns=schemas['movies'].names(),
        use_pyarrow=True)
    
df = df.with_row_index(name="movie_id", offset=N_USERS + 1)

#write movie_id_mapping.dat
df2 = df.select(["movie_id", "movie_id_orig"])
df2.write_csv(os.path.join(get_project_dir(), "src/main/resources/ml-1m/movie_id_mapping.dat"))

df_formatted = df.select(
    pl.format("{}::{}::{}",
        pl.col("movie_id"),
        pl.col("title"),
        pl.col("genres")).alias("output")
)
df_formatted.write_csv(
    os.path.join(get_project_dir(), "src/main/resources/ml-1m/movies.dat"),
    include_header=False,
    quote_style="never"
)

#add new number to

df = None
file_path = os.path.join(get_project_dir(), "src/main/resources/ml-1m-orig/ratings.dat")
processed_buffer = io.StringIO()
with open(file_path, "r", encoding='iso-8859-1') as file:
    for line in file:
        line2 = line.replace('::', '\t')
        processed_buffer.write(line2)
    processed_buffer.seek(0)
    df = pl.read_csv(processed_buffer,
        encoding='iso-8859-1', has_header=False,
        skip_rows=0, separator='\t', schema=schemas['ratings'],
        try_parse_dates=False,
        new_columns=schemas['ratings'].names(),
        use_pyarrow=True)
    
df = df.join(
    df2.select(["movie_id_orig", "movie_id"]),
    on="movie_id_orig",
    how="left"
)
df_formatted = df.select(
        pl.format("{}::{}::{}::{}",
            pl.col("user_id"),
            pl.col("movie_id"),
            pl.col("rating"),
            pl.col("timestamp")).alias("output")
    )
df_formatted.write_csv(
    os.path.join(get_project_dir(), "src/main/resources/ml-1m/ratings.dat"),
    include_header=False,
    quote_style="never"
)