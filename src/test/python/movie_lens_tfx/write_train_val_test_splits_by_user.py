'''
splitting the data into 80:10:10 for train:val_test ratings dataset on a pers user
basis.
'''
from collections import OrderedDict

from helper import *
import polars as pl
import os
import io

pl.Config.set_fmt_str_lengths(900)

file_path = os.path.join(get_project_dir(), "src/main/resources/ml-1m/ratings.dat")

schema = pl.Schema(OrderedDict({'user_id': pl.Int64,
    'movie_id': pl.Int64, 'rating': pl.Int64,
    'timestamp' : pl.Int64}))

processed_buffer = io.StringIO()
df = None
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

df = df.sort(["user_id", "timestamp"])

# Calculate per-user sequence index and total ratings count
df = df.with_columns(
    user_seq=pl.cum_count("rating").over("user_id") - 1,
    user_total=pl.len().over("user_id")
)

if True:
    #filter out users with less than 30 ratings and movies with less than 30 ratings
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

df_train = df.filter(pl.col("fraction") < 0.80)
df_val = df.filter((pl.col("fraction") >= 0.80) & (pl.col("fraction") < 0.90))
df_test = df.filter(pl.col("fraction") >= 0.90)

# array records are written in WriteRamkerInputArrayRecords
# parquet records are written in WriteRetrievalInputParquet.py
# write to .dat here
for df_write, prefix in zip([df_train, df_val, df_test], ['train', 'val', 'test']):
    #write dat files
    file_path = os.path.join(get_bin_dir(), f'ratings_{prefix}.dat')
    
    df_formatted = df_write.select(
        pl.format("{}::{}::{}::{}",
            pl.col("user_id"),
            pl.col("movie_id"),
            pl.col("rating"),
            pl.col("timestamp")).alias("output")
    )
    df_formatted.write_csv(
        file_path,
        include_header=False,
        quote_style="never"
    )

