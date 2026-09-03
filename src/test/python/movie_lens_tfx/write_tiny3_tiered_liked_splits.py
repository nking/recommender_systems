'''a smapligng of 100 ratings from the ratings*_liked.dat files to include
all 3 tiers in each and an overlap of users'''
import io
from typing import OrderedDict

from helper import *
import os
import glob
import polars as pl
import msgpack

from array_record.python import array_record_module

file_paths = glob.glob(os.path.join(get_project_dir(), "src/test/resources/ml-1m/ratings*_liked.dat"))

outdir = os.path.join(get_bin_dir(), "tiny3")
os.makedirs(outdir, exist_ok=True)

schema = pl.Schema(OrderedDict({'user_id': pl.Int64,
    'movie_id': pl.Int64, 'rating': pl.Int64,
    'timestamp' : pl.Int64}))

in_path_movie_tiers = os.path.join(get_project_dir(),
    "src/test/resources/movie_tiers.json")
movie_tiers_df = pl.read_ndjson(in_path_movie_tiers)

dfs = []

for file_path in file_paths:
    processed_buffer = io.StringIO()
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
    df = df.join(movie_tiers_df, on="movie_id", how="left")
    dfs.append(df)
    
common_users = (
    dfs[0].select("user_id")
    .unique()
    .join(dfs[1].select("user_id").unique(), on="user_id")
    .join(dfs[2].select("user_id").unique(), on="user_id")
)

test_filtered = dfs[0].join(common_users, on="user_id")
train_filtered = dfs[1].join(common_users, on="user_id")
val_filtered = dfs[2].join(common_users, on="user_id")

user_pool = common_users.sample(n=min(50, len(common_users)), seed=42)

train_subset = train_filtered.join(user_pool, on="user_id")
val_subset = val_filtered.join(user_pool, on="user_id")
test_subset = test_filtered.join(user_pool, on="user_id")

#same to have all 3 tiers in a file
def sample_100_with_tiers(df: pl.DataFrame, target_rows: int = 100) -> pl.DataFrame:
    tier_samples = []
    
    # Force inclusion of at least one row per tier if available
    for t in [0, 1, 2]:
        tier_subset = df.filter(pl.col("tier") == t)
        if len(tier_subset) > 0:
            tier_samples.append(tier_subset.sample(n=1, seed=42))
            
    mandatory_sample = pl.concat(tier_samples)
    remaining_needed = target_rows - len(mandatory_sample)
    
    if remaining_needed > 0:
        # Exclude already selected rows to avoid duplication
        remaining_pool = df.join(
            mandatory_sample,
            on=["user_id", "movie_id", "timestamp"],
            how="anti"
        )
        sample_size = min(remaining_needed, len(remaining_pool))
        
        if sample_size > 0:
            extra_sample = remaining_pool.sample(n=sample_size, seed=42)
            result = pl.concat([mandatory_sample, extra_sample])
        else:
            result = mandatory_sample
    else:
        result = mandatory_sample.head(target_rows)
        
    return result

# 4. Apply the sampling to each split
train_final = sample_100_with_tiers(train_subset, target_rows=100)
val_final = sample_100_with_tiers(val_subset, target_rows=100)
test_final = sample_100_with_tiers(test_subset, target_rows=100)

def write_to_outfile(df_write, out_file_path):
    df_formatted = df_write.select(
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
    assert(len(df_formatted) == 100)
    
def write_to_array_record(df_write: pl.DataFrame, out_file_path: str):
    assert len(df_write) == 100, f"Expected 100 rows, got {len(df_write)}"
    
    # Format rows into strings matching your previous delimiter structure
    df_formatted = df_write.select(
        pl.format(
            "{}::{}::{}::{}",
            pl.col("user_id"),
            pl.col("movie_id"),
            pl.col("rating"),
            pl.col("timestamp"),
        ).alias("output")
    )
    
    # Initialize the ArrayRecordWriter (group_size:1 is optimized for random access)
    writer = array_record_module.ArrayRecordWriter(out_file_path,
        "group_size:1")
    
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
    
    print(f"Successfully wrote {len(df_write)} records to {out_file_path}")


def verify_array_record(out_file_path: str):
    # Initialize the ArrayRecordReader
    reader = array_record_module.ArrayRecordReader(out_file_path)
    try:
        # Check total number of records
        num_records = reader.num_records()
        assert num_records == 100, f"Expected 100 records, but found {num_records}."
        
        # Read all records
        records = reader.read_all()
        assert len( records) == 100, f"Expected 100 records in read_all, but found {len(records)}."
        
        # Verify each row unpacks into exactly 4 integers
        for raw_bytes in records:
            row = msgpack.unpackb(raw_bytes)
            assert isinstance(row, (list, tuple)), f"Expected a list or tuple, got {type(row)}."
            assert len(row) == 4, f"Expected 4 items per row, got {len(row)}."
            assert all(isinstance(x, int) for x in
            row), f"Expected all items to be integers, got {row}."
        
        print(f"Verification successful: Read exactly {num_records} records, each containing 4 integers.")
    finally:
        reader.close()

write_to_outfile(train_final, os.path.join(outdir, "ratings_train_liked.dat"))
write_to_outfile(test_final, os.path.join(outdir, "ratings_test_liked.dat"))
write_to_outfile(val_final, os.path.join(outdir, "ratings_val_liked.dat"))

write_to_array_record(train_final, os.path.join(outdir, "ratings_train_liked.array_record"))
write_to_array_record(test_final, os.path.join(outdir, "ratings_test_liked.array_record"))
write_to_array_record(val_final, os.path.join(outdir, "ratings_val_liked.array_record"))

verify_array_record(os.path.join(outdir, "ratings_train_liked.array_record"))
verify_array_record(os.path.join(outdir, "ratings_test_liked.array_record"))
verify_array_record(os.path.join(outdir, "ratings_val_liked.array_record"))
