'''
splitting the data into 80:10:10 for train:val_test ratings datasets

The first split is by time for (train + val) split from test.

The second split is by user between train and val.

The temporal split avoids leakage of user data between train and test.
The user split allows the validation dataset to truly check for generalization
of the trained model.
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

#order by time and take last 10% for test
df = df.sort("timestamp")
train_val_size = int(len(df) * 0.9)
df1 = df.head(train_val_size)
df_test = df.tail(len(df) - train_val_size)

# split df1 by user_id, into partitions that are 88.89% : 11.11%
# subsetsum is np-hard, so use an approx solution instead

def choose_randomly(df1):
    unique_users = df1.select("user_id").unique()
    unique_users = unique_users.sample(fraction=1.0, shuffle=True, seed=42)
    
    split_idx = int(len(unique_users) * 0.8889)
    train_user_ids = unique_users.head(split_idx)
    val_user_ids = unique_users.tail(len(unique_users) - split_idx)
    
    df_train = df1.filter(pl.col("user_id").is_in(train_user_ids["user_id"].implode()))
    df_val = df1.filter(pl.col("user_id").is_in(val_user_ids["user_id"].implode()))
    return df_train, df_val

df_train, df_val = choose_randomly(df1)

#print(f'{len(df_train)} train samples,  {len(df_val)}')
print(f'{len(df_train)} train2 samples,  {len(df_val)}, ratio = {len(df_val)/len(df_train)}')

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
    