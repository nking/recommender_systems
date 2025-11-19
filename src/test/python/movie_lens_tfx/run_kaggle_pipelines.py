import shutil

from tfx.orchestration import metadata

from ml_metadata.metadata_store import metadata_store

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src/test/python/movie_lens_tfx"))
sys.path.append(os.path.join(os.getcwd(), "src/main/python/movie_lens_tfx"))

from helper import *
from movie_lens_tfx.PipelineComponentsFactory import *
from movie_lens_tfx.tune_train_movie_lens import *

from absl import logging
tf.get_logger().propagate = False
logging.set_verbosity(logging.WARNING)
logging.set_stderrthreshold(logging.WARNING)

infiles_dict_ser, output_config_ser, split_names = get_test_data(use_small=False)
user_id_max = 6040
movie_id_max = 3952
n_genres = N_GENRES
n_age_groups = N_AGE_GROUPS
n_occupations = 21
MIN_EVAL_SIZE = 50  # make this larger for production pipeline

BATCH_SIZE = 64
NUM_EPOCHS = 20

num_examples = 800187

PIPELINE_NAME = 'rs_pipeline'
PIPELINE_ROOT = os.path.join(get_bin_dir(), PIPELINE_NAME)

# remove results from previous test runs:
try:
  print(f"removing: {PIPELINE_ROOT}")
  shutil.rmtree(PIPELINE_ROOT)
except OSError as e:
  pass
METADATA_PATH = os.path.join(PIPELINE_ROOT, 'tfx_metadata',  'metadata.db')
os.makedirs(os.path.join(PIPELINE_ROOT, 'tfx_metadata'), exist_ok=True)

ENABLE_CACHE = True

metadata_connection_config = metadata.sqlite_metadata_connection_config(
  METADATA_PATH)

store = metadata_store.MetadataStore(metadata_connection_config)

tr_dir = os.path.join(get_project_dir(),
                      "src/main/python/movie_lens_tfx")

serving_model_dir = os.path.join(PIPELINE_ROOT, 'serving_model')
output_parquet_path = os.path.join(PIPELINE_ROOT, "transformed_parquet")

pre_dir = os.path.join(get_project_dir(), "src/main/resources", "pre_transform")
post_dir = os.path.join(get_project_dir(), "src/main/resources", "post_transform")

# for the custom ingestion component, the apache beam pipeline needs to be able to
# find the sibling scripts it imports.
# 2 solutions: (1) create a tar archive and use --extra_package in pipeline args
# or (2) use setup.py and --setup_file in pipeline args.

SETUP_FILE_PATH = os.path.abspath('setup.py')
beam_pipeline_args = [
  '--direct_running_mode=multi_processing',
  '--direct_num_workers=0',
  f'--setup_file={SETUP_FILE_PATH}',
  # f'--extra_package={ingest_tar_file}'
]

pipeline_factory = PipelineComponentsFactory(
  num_examples=num_examples, infiles_dict_ser=infiles_dict_ser,
  output_config_ser=output_config_ser, transform_dir=tr_dir,
  user_id_max=user_id_max, movie_id_max=movie_id_max,
  n_genres=n_genres, n_age_groups=n_age_groups,
  min_eval_size=MIN_EVAL_SIZE, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, device="CPU",
  serving_model_dir=serving_model_dir, output_parquet_path=output_parquet_path)

print(f"run baseline pipline to create a baseline model")
baseline_components = pipeline_factory.build_components(PIPELINE_TYPE.BASELINE,
  run_example_diff=False, pre_transform_schema_dir_path=pre_dir,
  post_transform_schema_dir_path=post_dir)

# create baseline model
my_pipeline = tfx.dsl.Pipeline(
  pipeline_name=PIPELINE_NAME,
  pipeline_root=PIPELINE_ROOT,
  components=baseline_components,
  enable_cache=ENABLE_CACHE,
  metadata_connection_config=metadata_connection_config,
  beam_pipeline_args=beam_pipeline_args,
)

tfx.orchestration.LocalDagRunner().run(my_pipeline)

print(f'run production pipeline to find best hyper-parameters, train the model and deploy')

artifact_types = store.get_artifact_types()
logging.debug(f"MLMD store artifact_types={artifact_types}")
artifacts = store.get_artifacts()
logging.debug(f"MLMD store artifacts={artifacts}")

components = pipeline_factory.build_components(PIPELINE_TYPE.PRODUCTION,
  run_example_diff=False, pre_transform_schema_dir_path=pre_dir,
  post_transform_schema_dir_path=post_dir)
  
# simulate experimentation of one model family
my_pipeline = tfx.dsl.Pipeline(
  pipeline_name=PIPELINE_NAME,
  pipeline_root=PIPELINE_ROOT,
  components=components,
  enable_cache=ENABLE_CACHE,
  metadata_connection_config=metadata_connection_config,
  beam_pipeline_args=beam_pipeline_args,
)

tfx.orchestration.LocalDagRunner().run(my_pipeline)
