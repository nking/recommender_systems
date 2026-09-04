'''
splitting the data into 80:10:10 for train:val_test ratings dataset on a per user
basis.

writes files to bin/full, bin/tiny, and bin/small
'''
import shutil
from collections import OrderedDict

import msgpack

from helper import *
import polars as pl
import os
import io

from array_record.python import array_record_module

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

df_train_disliked = df_train.filter(pl.col("rating") < 3)
df_train_3 = df_train.filter(pl.col("rating") == 3)
df_train_liked = df_train.filter(pl.col("rating") > 3)

df_val_disliked = df_val.filter(pl.col("rating") < 3)
df_val_3 = df_val.filter(pl.col("rating") == 3)
df_val_liked = df_val.filter(pl.col("rating") > 3)

df_test_disliked = df_test.filter(pl.col("rating") < 3)
df_test_3 = df_test.filter(pl.col("rating") == 3)
df_test_liked = df_test.filter(pl.col("rating") > 3)

out_dir_full = os.path.join(get_bin_dir(), "full")
out_dir_tiny = os.path.join(get_bin_dir(), "tiny")
out_dir_small = os.path.join(get_bin_dir(), "small")
for path in [out_dir_full, out_dir_tiny, out_dir_small]:
    try :
        shutil.rmtree(path)
    except:
        pass
    os.makedirs(path, exist_ok=True)

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

# array records are written in WriteRamkerInputArrayRecords
# parquet records are written in WriteRetrievalInputParquet.py
# write to .dat here
for df_write, prefix in zip(
        [df_train, df_train_disliked, df_train_3, df_train_liked,
        df_val, df_val_disliked, df_val_3, df_val_liked,
        df_test, df_test_disliked, df_test_3, df_test_liked,],
        ['train', 'train_disliked', 'train_3', 'train_liked',
        'val', 'val_disliked', 'val_3', 'val_liked',
        'test', 'test_disliked', 'test_3', 'test_liked',]):
    #write dat files
    file_path = os.path.join(out_dir_full, f'ratings_{prefix}.dat')
    
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
    write_to_array_record(df_write, os.path.join(out_dir_full, f'ratings_{prefix}.array_record'))
    df_write.write_parquet(os.path.join(out_dir_full, f'ratings_{prefix}.parquet'))
    
    file_path = os.path.join(out_dir_small, f'ratings_{prefix}.dat')
    df_formatted = df_write.head(1000).select(
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
    write_to_array_record(df_write.head(1000),
        os.path.join(out_dir_small, f'ratings_{prefix}.array_record'))
    df_write.head(1000).write_parquet(
        os.path.join(out_dir_small, f'ratings_{prefix}.parquet'))
    
    file_path = os.path.join(out_dir_tiny, f'ratings_{prefix}.dat')
    df_formatted = df_write.head(100).select(
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
    write_to_array_record(df_write.head(100),
        os.path.join(out_dir_tiny, f'ratings_{prefix}.array_record'))
    df_write.head(100).write_parquet(
        os.path.join(out_dir_tiny, f'ratings_{prefix}.parquet'))
    
    ## array_record and parquet
    
print(f"wrote files to directories\n{out_dir_full}\n{out_dir_small}\n{out_dir_tiny}")

