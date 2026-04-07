'''
splitting the data into 80:10:10 for train:val_test ratings datasets

The first split is by time for (train + val) split from test.

The second split is random between train and val.

The temporal split avoids leakage of user data between train and test.
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

shuffled_df = df1.sample(fraction=1.0, shuffle=True, seed=42)

split_idx = int(len(df) * 0.1)
df_val = shuffled_df.slice(0, split_idx)   # First 10%
df_train = shuffled_df.slice(split_idx)      # Remaining

#print(f'{len(df_train)} train samples,  {len(df_val)}')
print(f'#train: {len(df_train)}, #val:{len(df_val)}, #test:{len(df_test)}')

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
    
    