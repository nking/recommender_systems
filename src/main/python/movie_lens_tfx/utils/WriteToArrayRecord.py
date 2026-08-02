import os
import apache_beam as beam
from apache_beam.io.filesystems import FileSystems

class WriteToArrayRecord(beam.PTransform):
    """A reusable Beam PTransform to write byte PCollections to ArrayRecord files."""
    
    def __init__(self, file_path_prefix: str,
            file_name_suffix: str = ".arrayrecord"):
        super().__init__()
        self.file_path_prefix = file_path_prefix
        self.file_name_suffix = file_name_suffix
    
    class _WriteArrayRecordFn(beam.DoFn):
        def __init__(self, prefix: str, suffix: str):
            self.prefix = prefix
            self.suffix = suffix
            self.writer = None
            self.local_path = None
        
        def setup(self):
            # Deferred import to ensure compatibility across distributed worker environments
            from array_record.python import array_record_module
            
            # Create a unique local temp file path for this distributed worker bundle
            bundle_id = id(self)
            self.local_path = f"/tmp/bundle_{bundle_id}{self.suffix}"
            
            # Initialize ArrayRecordWriter (v0.8.3 compatible)
            # Default options use standard ArrayRecord internal block compression
            self.writer = array_record_module.ArrayRecordWriter(
                self.local_path, options="")
        
        def process(self, element: bytes):
            # Accepts the serialized byte strings directly from your pipeline
            self.writer.write(element)
        
        def finish_bundle(self):
            if self.writer:
                self.writer.close()
                self.writer = None
            
            # Generate a distinct final destination shard name using the bundle ID
            bundle_id = id(self)
            destination_path = f"{self.prefix}-{bundle_id}{self.suffix}"
            
            # Safely stream from local worker disk to GCS or Local Disk using Beam's I/O layers
            if os.path.exists(self.local_path):
                with FileSystems.open(self.local_path) as f_in:
                    with FileSystems.create(destination_path) as f_out:
                        f_out.write(f_in.read())
                
                # Clean up the worker's temporary disk space
                os.remove(self.local_path)
    
    def expand(self, pcoll):
        return pcoll | beam.ParDo(
            self._WriteArrayRecordFn(self.file_path_prefix,
                self.file_name_suffix))
