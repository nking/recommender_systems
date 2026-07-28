import io
import unittest
from typing import OrderedDict
from helper import *
import os
import polars as pl
import struct

def read_users_into_df(input_csv_path:str) -> pl.DataFrame:
    schema = pl.Schema(OrderedDict(
        {'user_id': pl.Int64,
        'gender': pl.String, 'age': pl.Int64,
        'occupation': pl.Int64,
        'zipcode': pl.String}))
    
    processed_buffer = io.StringIO()
    #print(f"key={key}, file_path={file_path}")
    with open(input_csv_path, "r", encoding='iso-8859-1') as file:
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

#NOTE: to improve scalability of this can use instead
# a datawarehouse or Polars Cloud and Polars On-Prem (Kubernetes),
# or native pyspark and sparksql
def build_user_binary_db(input_csv_path: str, output_bin: str):
    
    # Load and ensure it's strictly ordered by user_id
    df = read_users_into_df(input_csv_path).sort("user_id")
    
    #max_zip_len = df.select(
    #    pl.col("zipcode").fill_null("").str.len_bytes().max()
    #).item()
    #fmt = f'<cqq{max_zip_len}s'
    
    with open(output_bin, "wb") as f:
        # iter_rows(named=True) returns a dict for easy access
        for row in df.iter_rows(named=True):
            # Encode gender as a single byte
            gender_byte = str(row["gender"]).encode('ascii')[0:1]

            # Struct packing:
            # < : Little-endian standard size (no padding)
            # c : 1-byte char
            # I : 4-byte unsigned int (age)
            # I : 4-byte unsigned int (occupation)
            # Total = 9 bytes per row
            data = struct.pack('<cII',
                               gender_byte,
                               row["age"],
                               row["occupation"])
            f.write(data)

def verify_user_binary_db(output_bin: str, expected_df: pl.DataFrame):
    """
    Reads the binary file sequentially, decodes it, and asserts
    that every value matches the original Polars DataFrame.
    """
    fmt = "<cII"
    record_size = struct.calcsize(fmt)
    print(f"Validating binary file... Expected record size: {record_size} bytes")
    
    with open(output_bin, "rb") as f:
        for i, row in enumerate(expected_df.iter_rows(named=True)):
            chunk = f.read(record_size)
            
            # Assert we haven't hit EOF unexpectedly
            assert chunk, f"Unexpected EOF at row {i}"
            assert len(chunk) == record_size, f"Incomplete record at row {i}"
            
            # Unpack the binary chunk
            unpacked = struct.unpack(fmt, chunk)
            
            # Decode components
            gender_bin = unpacked[0].decode('ascii')
            age_bin = unpacked[1]
            occupation_bin = unpacked[2]
            
            # Format expected values
            expected_gender = str(row["gender"])[0:1]
            
            # Assertions
            assert gender_bin == expected_gender, f"Row {i}: Gender mismatch {gender_bin} != {expected_gender}"
            assert age_bin == row["age"], f"Row {i}: Age mismatch"
            assert occupation_bin == row[
                "occupation"], f"Row {i}: Occupation mismatch"
            
    # Ensure no trailing bytes exist at the end of the file
    with open(output_bin, "rb") as f:
        f.seek(0, 2)  # Go to end
        file_size = f.tell()
        expected_size = expected_df.height * record_size
        assert file_size == expected_size, f"File size contains trailing bytes! Size: {file_size}, Expected: {expected_size}"
    
    print("✅ Verification passed! Binary file perfectly matches the source DataFrame.")
    
class TestWriteUsersDB(unittest.TestCase):
    
    def test_write(self):
        
        in_path = os.path.join(get_project_dir(),
            "src/main/resources/ml-1m/users.dat")
        out_path = os.path.join(get_bin_dir(), "users.bin")
        build_user_binary_db(in_path, out_path)
        
        #assert can read
        df = read_users_into_df(in_path)
        verify_user_binary_db(out_path, df)
        print(f'wrote to {out_path}')
        

if __name__ == '__main__':
    unittest.main()

