import os
import shutil
import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
import tempfile

class WriteToArrayRecord(beam.PTransform):
    """A reusable Beam PTransform to write byte PCollections to ArrayRecord files with explicit sharding."""
    
    def __init__(self,
            file_path_prefix: str,
            file_name_suffix: str = ".array_record", num_files: int = 1):
        super().__init__()
        self.file_path_prefix = file_path_prefix
        self.file_name_suffix = file_name_suffix
        self.num_files = max(1, num_files)
    
    class _WriteShardFn(beam.DoFn):
        def __init__(self, prefix: str, suffix: str, num_files: int=1):
            self.prefix = prefix
            self.suffix = suffix
            self.num_files = num_files
        
        def process(self, element):
            # element is a tuple: (shard_index, iterable_of_bytes)
            shard_index, elements = element
            
            # Deferred import to ensure compatibility across distributed worker environments
            from array_record.python import array_record_module
            
            # Create a unique local temp file path for this shard
            temp_dir = tempfile.mkdtemp()
            local_filename = f"shard_{shard_index}{self.suffix}"
            local_path = os.path.join(temp_dir, local_filename)
            
            # Initialize ArrayRecordWriter (v0.8.3 compatible)
            writer = array_record_module.ArrayRecordWriter(local_path, options="")
            try:
                for el in elements:
                    writer.write(el)
            finally:
                if writer:
                    writer.close()
            
            destination_path = f"{self.prefix}-{shard_index:05d}-of-{self.num_files:05d}{self.suffix}"
            
            try:
                # Safely stream from local worker disk to GCS or Local Disk
                if os.path.exists(local_path):
                    
                    # Ensure the destination directory exists
                    dest_dir, _ = FileSystems.split(destination_path)
                    if dest_dir and not FileSystems.exists(dest_dir):
                        FileSystems.mkdirs(dest_dir)
                    
                    # Use standard open() for local, FileSystems.create() for destination.
                    # shutil.copyfileobj streams data in chunks, preventing RAM exhaustion.
                    with open(local_path, 'rb') as f_in:
                        with FileSystems.create(destination_path) as f_out:
                            shutil.copyfileobj(f_in, f_out,
                                length=1024 * 1024 * 16)  # 16MB chunks
            
            finally:
                # Guaranteed cleanup of the worker's temporary disk space
                if os.path.exists(local_path):
                    os.remove(local_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
    
    def expand(self, pcoll):
        return (
            pcoll
            | "AssignShard" >> beam.Map(lambda el: (hash(el) % self.num_files, el))
            | "GroupPerShard" >> beam.GroupByKey()
            | "WriteShards" >> beam.ParDo(
                self._WriteShardFn(
                    self.file_path_prefix,
                    self.file_name_suffix,
                    self.num_files
                )
            )
        )
