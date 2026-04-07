"""
wanting to design an evaluation that can use the simplest principles of hypergeometric probability
distribution.  this needs a test set in which the candidates are somewhat obvious such as
users who have only rated movies that are a single genre and the same genre among those movies rated by the user.
"""
from collections import OrderedDict
import polars as pl
import io
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src/test/python/movie_lens_tfx"))
sys.path.append(os.path.join(os.getcwd(), "src/main/python/movie_lens_tfx"))
from helper import *

pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_width_chars(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(900)
pl.Config.set_fmt_table_cell_list_len(900)

_infiles_dict_ser, _, __ = get_test_data(use_small=False)
_infiles_dict = deserialize(_infiles_dict_ser)

file_paths = {
  #'ratings': _infiles_dict['ratings']['uri'],
  'ratings': os.path.join(get_project_dir(), "src/main/resources/ml-1m/",
  'ratings_train.dat'),
  'users':_infiles_dict['users']['uri'],
  'movies':_infiles_dict['movies']['uri'],
}

genres = ["Action", "Adventure", "Animation", "Children", "Comedy",
          "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
          "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
          "Thriller", "War", "Western"]

schemas = {
  'ratings' : pl.Schema(OrderedDict({'user_id': pl.Int64,
    'movie_id': pl.Int64, 'rating': pl.Int64,
    'timestamp' : pl.Int64})),
  'users' : pl.Schema(OrderedDict({'user_id': pl.Int64,
    'gender': pl.String, 'age': pl.Int64,
    'occupation' : pl.Int64,
    'zipcode' : pl.String})),
  'movies' : pl.Schema(OrderedDict({'movie_id': pl.Int64,
    'title': pl.String, 'genres': pl.String}))}

dfs = {}
for key in file_paths:
    processed_buffer = io.StringIO()
    file_path = file_paths[key]
    schema = schemas[key]
    #print(f"key={key}, file_path={file_path}")
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
    if key=="movies":
      df = df.with_columns(
        pl.col("genres").str.replace("Children's", "Children")
      )
    dfs[key] = df

# movies with only 1 genre
df_movies_single = dfs['movies'].filter(
    ~pl.col("genres").str.contains("|", literal=True)
)
df_movies_multiple = dfs['movies'].filter(
    pl.col("genres").str.contains("|", literal=True)
)

result = (
    dfs['ratings'].filter(pl.col("rating") > 4)
    .join(dfs['movies'].select(["movie_id", "genres"]), on="movie_id")
    .group_by("user_id")
    .agg([
        # Count how many movie_ids in this group exist in the multiple_genres table
        pl.col("movie_id").is_in(df_movies_multiple["movie_id"])
          .sum().alias("n_multiple"),
        # Count the number of unique genres seen for this user
        pl.col("genres").n_unique().alias("n_unique_genres")
    ])
)
#print(result)
result_users = result.filter(pl.col("n_multiple") == 0,
  pl.col("n_unique_genres") == 1)
print(f'{result_users.count()}\n{result_users}')

#19 users
df_users = dfs['users'].filter(
    pl.col("user_id").is_in(result_users['user_id'].implode())
)
print(df_users['user_id'].count())
print(df_users['user_id'].to_numpy().tolist())
#UserID::Gender::Age::Occupation::Zip-code
#1::F::1::10::48067
#write .dat file:
df_as_string = df_users.select(
    pl.concat_str(pl.all().cast(pl.String), separator="::").alias("combined")
)
df_as_string.write_csv(os.path.join(get_bin_dir(), "users_single_genre.dat"),
  include_header=False, quote_style="never")

#write a parquet file:
df_users.write_parquet(os.path.join(get_bin_dir(),"users_single_genre.parquet"))

print('done writing files bin/users_single_genre.*')

""" one might want to compare the hypergeomtric distr eval to a model trained only with
this data for something like best case statistics to compare and understand results.

For that reason, the partitioning of the 1 million ratings dataset is left here
commented out, but not used:

#train+val, test partitions
# for each user, make 80:10:10 splits randomly with at least 1 row to each parittion for each user
df_ratings_filtered = df_ratings_filtered.sample(fraction=1.0, is_train=True, seed=42)

# Add ranking and total count columns per user
df_split = df_ratings_filtered.with_columns([
    pl.int_range(pl.len()).over("user_id").alias("row_nr"),
    pl.len().over("user_id").alias("user_count")
]).with_columns(
    (pl.col("row_nr") / pl.col("user_count")).alias("rel_pos")
)
# First row -> Test, Second -> Val, Third -> Train (ensures "at least 1")
# All other rows follow the 80/10/10 distribution
df_final = df_split.with_columns(
    pl.when(pl.col("row_nr") == 0).then(pl.lit("test"))
    .when(pl.col("row_nr") == 1).then(pl.lit("val"))
    .when(pl.col("row_nr") == 2).then(pl.lit("train"))
    .when(pl.col("rel_pos") < 0.8).then(pl.lit("train"))
    .when(pl.col("rel_pos") < 0.9).then(pl.lit("val"))
    .otherwise(pl.lit("test"))
    .alias("partition")
)

train_df = df_final.filter(pl.col("partition") == "train").drop(["row_nr", "user_count", "rel_pos", "partition"])
val_df   = df_final.filter(pl.col("partition") == "val").drop(["row_nr", "user_count", "rel_pos", "partition"])
test_df  = df_final.filter(pl.col("partition") == "test").drop(["row_nr", "user_count", "rel_pos", "partition"])

# assert all users are in all 3 partitions:
expected_user_count = df_ratings_filtered['user_id'].n_unique()
assert train_df['user_id'].n_unique() == expected_user_count, "Train set missing users"
assert val_df['user_id'].n_unique() == expected_user_count,   "Val set missing users"
assert test_df['user_id'].n_unique() == expected_user_count,  "Test set missing users"

NOTE too that you would not want to make a Two-Tower bi-encoder with it because it would
be severely underdetermined problem:
  * user embedding of 94 times embedding dimension of (94)^(1/4) = 3
  * movie embedding of 651 movies times embedding dimension of 6
  * each model having at least 1 fully connected layer and the same output embed dim of 8 or so
  => total number params = (94*3) + (3*8)+8  + (651*6) + (6*8) +8 = 4276
  but only have 94 inputs.
Adding regularization to the model can help reduce fitting the noise to make a
simpler lower variance fit, but the result might still be overfit.
"""
