from typing import TypedDict

from tfx.types import standard_artifacts, channel_utils
from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.annotations import InputArtifact, OutputArtifact
from tensorflow_metadata.proto.v0 import statistics_pb2
from tfx.types.system_executions import Process

import os
from absl import logging
logging.set_verbosity(logging.INFO)
logging.set_stderrthreshold(logging.INFO)

class DictOutput(TypedDict):
  num_train: int
  num_eval: int
  num_test: int

@component()
def NumExamplesExtractor(
  stats: InputArtifact[standard_artifacts.ExampleStatistics],
  num_train: OutputArtifact[standard_artifacts.Integer],
  num_test: OutputArtifact[standard_artifacts.Integer],
  num_eval: OutputArtifact[standard_artifacts.Integer],
  ) -> DictOutput:
  
  logging.info(f"NumExamplesExtractor")
  print(f"num_train.uri={num_train.uri}")
  
  stats_uri = stats.uri
  out = {}
  for split_name in ["train", "eval", "test"]:
    stats_path = os.path.join(stats_uri,f'Split-{split_name}', 'FeatureStats.pb')
    try:
      with open(stats_path, 'rb') as f:
        serialized_stats = f.read()
    except FileNotFoundError:
      print(f"Error: File not found at {stats_path}")
    stats = statistics_pb2.DatasetFeatureStatisticsList()
    stats.ParseFromString(serialized_stats)
    
    for dataset in stats.datasets:
      out[f'num_{split_name}'] = dataset.num_examples
      if split_name == "train":
        num_train.write(dataset.num_examples)
        with open(num_train.uri, 'w') as f:
          f.write(str(dataset.num_examples))
      elif split_name == "eval":
        num_eval.write(dataset.num_examples)
        with open(num_eval.uri, 'w') as f:
          f.write(str(dataset.num_examples))
      else:
        with open(num_test.uri, 'w') as f:
          f.write(str(dataset.num_examples))
        num_test.write(dataset.num_examples)
  return out