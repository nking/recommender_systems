import os
import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
import os
import apache_beam as beam
from apache_beam.io.filesystems import FileSystems

class WriteToArrayRecord(beam.PTransform):
    """A reusable Beam PTransform to write byte PCollections to ArrayRecord files with explicit sharding."""
    
    def __init__(self, file_path_prefix: str,
            file_name_suffix: str = ".arrayrecord", num_files: int = 1):
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
            local_path = f"/tmp/shard_{shard_index}{self.suffix}"
            
            # Initialize ArrayRecordWriter (v0.8.3 compatible)
            writer = array_record_module.ArrayRecordWriter(local_path, options="")
            try:
                for el in elements:
                    writer.write(el)
            finally:
                if writer:
                    writer.close()
            
            destination_path = f"{self.prefix}-{shard_index:05d}-of-{self.num_files:05d}{self.suffix}"
            
            # Safely stream from local worker disk to GCS or Local Disk using Beam's I/O layers
            if os.path.exists(local_path):
                with FileSystems.open(local_path) as f_in:
                    with FileSystems.create(destination_path) as f_out:
                        f_out.write(f_in.read())
                
                # Clean up the worker's temporary disk space
                os.remove(local_path)
    
    def expand(self, pcoll):
        return (
                pcoll
                | "AssignShard" >> beam.Map(
            lambda el: (hash(el) % self.num_files, el))
                | "GroupPerShard" >> beam.GroupByKey()
                | "WriteShards" >> beam.ParDo(
            self._WriteShardFn(
                self.file_path_prefix,
                self.file_name_suffix,
                self.num_files
            )
        )
        )
