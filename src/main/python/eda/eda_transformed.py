#!/usr/bin/env python
# coding: utf-8

# run this after local notebook has run the PRE-PROCESSING pipeline build

from tfx.orchestration import metadata

import tensorflow_transform as tft

from ml_metadata.proto import metadata_store_pb2
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


# ## EDA on the raw data

# ### w/ Polars, Seaborn, and Matplotlib

# In[2]:


import polars as pl
#import matplotlib.pyplot as plt
#seaborn version installed is 0.12.2.  need>= 0.13.0 for polars
#import seaborn as sns
#import seaborn_polars as snl
from scipy.stats.distributions import chi2
from collections import OrderedDict
import re
import io
from datetime import datetime
import pytz
import dcor
import numpy as np
#import altair as alt
import plotly.express as px
#needs pip install plotly jupyterlab anywidget
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StringType, StructField, \
  StructType, LongType, IntegerType
from pyspark.ml.fpm import FPGrowth, PrefixSpan
from pyspark.sql.functions import size as spark_size
from pyspark.sql import functions as F
import json
import multiprocessing
pl.Config.set_fmt_str_lengths(900)



# In[3]:


def can_reject_indep(x : np.ndarray, y:np.ndarray, alpha:float = 0.05, debug:bool=False):
  """
  Args:
    x: float array
    y: float array
  reject independence for 
    n*C >= inv(F{chi^2-1})(1-alpha)
    where n = len(x)
      C = fast distance covariance following 2019 Chaudhuri and Hu
      inv(F{chi^2-1}) is the inverse of the CDF.
  """
  with np.errstate(divide='ignore'):
    C = dcor.distance_covariance(x, y, method='mergesort')
  lhs = len(x)*C
  rhs = chi2.ppf(1 - alpha, df=x.shape[-1])
  if debug:
    print(f"nC={lhs}\nppf(1-{alpha}, dof={x.shape[-1]})={rhs}")
  return lhs >= rhs


# In[4]:


CTZ = pytz.timezone("America/Chicago")
genres = ["Action", "Adventure", "Animation", "Children", "Comedy",
          "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
          "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
          "Thriller", "War", "Western"]

schemas = {
  'ratings' : pl.Schema(OrderedDict({'user_id': pl.Int64, 
    'movie_id': pl.Int64, 'rating': pl.Int64,
    'timestamp' : pl.Int64})),
  'users' : pl.Schema(OrderedDict({'user_id': pl.Int64, 
    'gender': pl.String, 'age': pl.Int64,
    'occupation' : pl.Int64, 
    'zipcode' : pl.String})),
  'movies' : pl.Schema(OrderedDict({'movie_id': pl.Int64, 
    'title': pl.String, 'genres': pl.String}))}

_infiles_dict_ser, _, __ = get_test_data(use_small=False)
_infiles_dict = deserialize(_infiles_dict_ser)

file_paths = {
  'ratings': _infiles_dict['ratings']['uri'],
  'users':_infiles_dict['users']['uri'],
  'movies':_infiles_dict['movies']['uri'],
}

#polars.read_csv( source=
#  encoding='iso-8859-1', 
#  has_header=False, skip_rows=0, try_parse_dates=True, 
#  use_pyarrow=True

labels_dict = {}
labels_dict['age_group'] = {0:'1', 1:'18', 2:'25', 3:'35', 4:'45', 5:'50', 6:'56'} 
labels_dict['gender'] = {0:'F', 1:'M'}
labels_dict['occupation'] = {0:  "other", 1:  "academic/educator", 2:  "artist",
    3:  "clerical/admin", 4:  "college/grad student", 5:  "customer service",
    6:  "doctor/health care", 7:  "executive/managerial", 8:  "farmer", 9:  "homemaker",
    10:  "K-12 student", 11:  "lawyer", 12:  "programmer", 13:  "retired",
    14:  "sales/marketing", 15:  "scientist", 16:  "self-employed", 17:  "technician/engineer",
    18:  "tradesman/craftsman", 19:  "unemployed", 20:  "writer"}
labels_dict_arrays = {}
for k in labels_dict:
    labels_dict_arrays[k]=[labels_dict[k][k2] for k2 in labels_dict[k]]


# In[ ]:

img_dir = os.path.join(get_bin_dir(), "local_notebook", "images")
os.makedirs(img_dir, exist_ok=True)

# ### Run data pre-processing on full dataset

# In[5]:


infiles_dict_ser, output_config_ser, split_names = get_test_data(use_small=False)
user_id_max = 6040
movie_id_max = 3952
n_genres = N_GENRES
n_age_groups = N_AGE_GROUPS
n_occupations = 21
MIN_EVAL_SIZE = 50 #make this larger for production pipeline

test_num = "1"
    
PIPELINE_NAME = 'TestPipelines'
output_data_dir = os.path.join(get_bin_dir(), "local_notebook", test_num)
PIPELINE_ROOT = os.path.join(output_data_dir, PIPELINE_NAME)

#METADATA_PATH = os.path.join(PIPELINE_ROOT, 'tfx_metadata','metadata.db')
#metadata_connection_config = metadata.sqlite_metadata_connection_config(
#  METADATA_PATH)
#
#store = metadata_store.MetadataStore(metadata_connection_config)

if get_kaggle():
  tr_dir = "/kaggle/working/"
else:
  tr_dir = os.path.join(get_project_dir(), "src/main/python/movie_lens_tfx")

serving_model_dir = os.path.join(PIPELINE_ROOT, 'serving_model')
output_parquet_path = os.path.join(PIPELINE_ROOT, "transformed_parquet")

# ## EDA on the transformed data

# ### using Polars, Plotly.express and PySpark first 

# In[ ]:

from movie_lens_tfx.utils import movie_lens_utils
from plotly.subplots import make_subplots
import plotly.graph_objects as go

parquet_path = os.path.join(PIPELINE_ROOT, "transformed_parquet")

pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_width_chars(-1)
pl.Config.set_tbl_cols(-1)
colors = px.colors.qualitative.Plotly 

genres_set = set(genres)

#make co-occurence matrix and heatmap
def genres_co_occurence_heatmap(filtered, split_name:str, rating:int,
    item:str="genres"):
  
    print(f'genres_co_occurence_heatmap filtered={filtered}')
    print(f'{filtered.columns}')
    print(f'{filtered.dtypes}')
    
    exploded = filtered.explode(item)
    unique_items = exploded[item].unique().to_list()
    
    print(f'unique_items={unique_items.dtypes}')
    
    basket_item = (
        exploded.pivot(
            values=item,
            index="row",
            on=item,
            aggregate_function="len"
        )
        .fill_null(0)
        .with_columns([
            pl.col(col).cast(pl.Int32) for col in unique_items
        ])
    )
    matrix = basket_item.drop("row").to_numpy()
    co_occurrence = np.dot(matrix.T, matrix)
    fig = px.imshow(
        co_occurrence, x=genres, y=genres,
        color_continuous_scale="RdBu_r", # Red-Blue reversed for correlation
        title=f"{split_name}, rating {rating}: Co-occurrence Matrix Heatmap"
    )
    fig.write_image(os.path.join(img_dir, f"{split_name}_genre_cooccurence_rating_{rating}_heatmap.png"))
    
def write_dist_corr_heatmap(df, skip_columns:set[str], outfile_name:str, height:int=None, width:int=None):
  if os.path.exists(os.path.join(img_dir, outfile_name)):
    return
  labels = []
  dist_corr_matrix = []
  ii = 0
  for i, feature in enumerate(df.columns):
    if feature in skip_columns:
      continue
    labels.append(feature)
    d = [0 for _ in range(ii)]
    ii += 1
    feature_arr = df[feature].to_numpy().astype(np.float64)
    for j in range(i, len(df.columns)):
      feature2 = df.columns[j]
      if feature2 in skip_columns:
        continue
      feature_arr2 = df[feature2].to_numpy().astype(np.float64)
      # dcor wants np.float64 numpy array input
      d.append(dcor.distance_correlation(feature_arr, feature_arr2, method='mergesort'))
    dist_corr_matrix.append(d)
  logging.debug(f'{len(labels)}: labels in correlation matrix: {labels}')
  for i, line in enumerate(dist_corr_matrix):
    logging.debug(f"i={i}, {len(line)}")
  if width is not None and height is not None:
    fig = px.imshow(
      dist_corr_matrix, x=labels, y=labels, labels=dict(x="Column Names", y="Row Names", color="dist corr"),
      width=width, height=height,
      color_continuous_scale="RdBu_r", title="Dist Correlation Matrix Heatmap"
    )
  else:
    fig = px.imshow(
      dist_corr_matrix, x=labels, y=labels, labels=dict(x="Column Names", y="Row Names", color="dist corr"),
      color_continuous_scale="RdBu_r", title="Dist Correlation Matrix Heatmap"
    )
  fig.write_image(os.path.join(img_dir, outfile_name), format='png')

for split_name in ["train", "eval", "test"]:
    in_file_pattern = os.path.join(parquet_path, f"Split-{split_name}*")
    df = pl.read_parquet(in_file_pattern)
    #df = pl.scan_parquet(in_file_pattern)
    df = df.with_columns(
        pl.col("genres").map_elements(movie_lens_utils.deserialize, return_dtype=pl.List(pl.Float32))
    )
    # add an index for row number:
    df = df.with_row_index("row")
    df = df.with_columns(
      (pl.col("rating") * 5).round(0).cast(pl.Int64).alias("rating")
    )
    df = df.with_columns(
      pl.col("*").exclude(['genres']).cast(pl.Int32)
    )
    df = df.with_columns(
      pl.col("genres").cast(pl.List(pl.Int32))
    )
    print(f"\n{split_name} df: {df.sort('user_id').head(5)}")
    print(f'DESCRIBE:\n{df.describe()}')
    
    write_dist_corr_heatmap(df, skip_columns=set(["genres", "row"]), outfile_name=f"{split_name}_dist_corr_heatmap.png")
    
    df = df.with_columns(
        pl.col("occupation").map_elements(lambda x: labels_dict['occupation'].get(x,x), return_dtype=pl.String).alias("occ")
    )
    
    print(f"\ndf w/ occ: {df.sort('user_id').head(5)}")
    
    df_sparse_str = df.with_columns(
      pl.col("genres").list.eval(pl.arg_where(pl.element().cast(pl.Boolean)))
      .alias("genres")
    )
    df_sparse_str = df_sparse_str.with_columns(
      pl.col("genres").list.eval(pl.element().replace_strict(
        {k: v for k, v in enumerate(genres)}, default=pl.lit(None)))
      .alias("genres"))
    
    print(f'df_sparse_str={df_sparse_str.sort("user_id").head(5)}')
    
    #"""
    exploded = df.with_columns(
        pl.col("genres").list.to_array(width=n_genres).arr.to_struct(
                fields=[f"{genres[i]}" for i in range(n_genres)]
            )
        ).unnest("genres")
    print(f'columns:\n{exploded.columns}')
   
    n_occupations = len(labels_dict_arrays['occupation'])
    n_page = n_occupations//3
  
    for genre in genres_set:
        fig=px.box(exploded, x=genre, y='rating', color='gender')
        fig.write_image(os.path.join(img_dir, f"{split_name}_{genre}_gender_box.png"))
        #plotting occupation over 3 pages
        page_idx = 0
        for j in range(0, n_occupations, n_page):
            occs = labels_dict_arrays['occupation'][j : j+n_page]
            filtered = exploded.filter(pl.col("occ").is_in(occs))
            fig=px.box(filtered, x=genre, y='rating', color='occ')
            fig.write_image(os.path.join(img_dir, f"{split_name}_{genre}_occupation_page{page_idx}_box.png"))
            page_idx += 1
    del exploded
    #"""
    
    #pairplots of "rating", "gender", "age", "occ", "hr_wk", "month", "weekday"
    #"""
    _features = ["rating", "gender", "age", "occ", "hr_wk", "month", "weekday"]
    for i, feat1 in enumerate(_features):
        for j in range(i+1, len(_features)):
            feat2 = _features[j]
            fig=px.scatter(df, x=feat1, y=feat2)
            fig.write_image(os.path.join(img_dir, f"{split_name}_{feat1}_{feat2}_pair_plot.png"))
    #"""
    #for market basket analysis, will make a sparsely populated column
   
    exploded = df_sparse_str.explode('genres')
    
    print(f'exploded.columns={exploded.columns}')
    print(f'exploded head={exploded.sort("user_id").head(5)}')
    for rating in [5, 4, 3, 2, 1]:
        filtered = exploded.filter(pl.col("rating") == rating)
        print(f'filtered.columns={filtered.columns}')
        print(f'filtered head={filtered.sort("user_id").head(2)}')
        #pie chart counting each genre
        genres_counts = filtered['genres'].value_counts()#.reset_index()
        genres_counts.columns = ['Genre', 'Count']
        genres_counts = genres_counts.with_columns(pl.col('Count').cast(pl.Int64))
        print(f'dtypes={genres_counts.dtypes}')
        print(f'head={genres_counts.head(5)}')
        fig = px.pie(genres_counts, values='Count', names='Genre',
            title=f'{split_name} rating {rating}: Genres')
        fig.write_image(os.path.join(img_dir, f"{split_name}_genres_rating_{rating}_pie.png"))
  
        filtered = df_sparse_str.filter(pl.col("rating") == rating)
        genres_co_occurence_heatmap(filtered, split_name, rating, "genres")

        #NOTE: looked at frequent itemsets of genres as the basket items and found no assoc rules
        
        #further filter to only user_id and movie_id
        #group movie_id by user_id
        filtered = filtered.select(["user_id", "movie_id"])
        filtered = filtered.group_by("user_id").agg(pl.col("movie_id").unique().alias("movie_ids"))
        filtered = filtered.select(["movie_ids"])
        #order doesn't matter for FPGrowth, except that we want sets to be unique for their composition
        filtered = filtered.with_columns(
          pl.col("movie_ids").list.sort()
        )
        
print(f'wrote multivariate EDA images to {img_dir}')


# ### using TFDV

# #load the transformed examples
# 
# from tfx.dsl.io import fileio
# from tfx.orchestration import metadata
# from tfx.components import StatisticsGen, SchemaGen, ExampleValidator
# from tfx.utils import io_utils
# from tensorflow_metadata.proto.v0 import anomalies_pb2, schema_pb2
# from tensorflow_transform.tf_metadata import schema_utils
# 
# #from movie_lens_tfx.ingest_pyfunc_component.ingest_movie_lens_component import *
# #from movie_lens_tfx.tune_train_movie_lens import *
# #from tfx import v1 as tfx
# 
# schema_list = store.get_artifacts_by_type("Schema")
# schema_list = sorted(schema_list,
#   key=lambda x: x.create_time_since_epoch, reverse=True)
# for artifact in schema_list:
#     if "post_transform_schema" in artifact.uri:
#         schema_uri = artifact.uri
#         break
# assert(schema_uri is not None)
# schema_file_path = [os.path.join(schema_uri, name) for name in os.listdir(schema_uri)][0]
# schema = tfx.utils.parse_pbtxt_file(schema_file_path, schema_pb2.Schema())
# feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec
# 
# examples_list = store.get_artifacts_by_type("Examples")
# #print(f"examples_list={examples_list}")
# examples_list = sorted(examples_list,
#   key=lambda x: x.create_time_since_epoch, reverse=True)
# for artifact in examples_list:
#     if "transformed_examples" in artifact.uri:
#         transformed_examples_uri = artifact.uri
#         break
# assert(transformed_examples_uri is not None)
# logging.debug(f"transfomed_examples_uri={transformed_examples_uri}")
# transform_uri = transformed_examples_uri[0:transformed_examples_uri.index("transformed_examples")]
# 
# """
# transformed_examples
# post_transform_anomalies 
# post_transform_schema
# pre_transform_stats
# post_transform_stats
# transform_graph
# updated_analyzer_cache
# pre_transform_schema
# """
# 
# def parse_tf_example(example_proto, feature_spec):
#     return tf.io.parse_single_example(example_proto, feature_spec)
# for split_name in ["train", "eval", "test"]:
#     tfrecord_uri = os.path.join(transform_uri, f"Split-{split_name}")
#     file_paths = [os.path.join(tfrecord_uri, name) for name in os.listdir(tfrecord_uri)]
#     ds_ser = tf.data.TFRecordDataset(file_paths, compression_type="GZIP")
#     ds = ds_ser.map(lambda x: parse_tf_example(x, feature_spec))
# 
#     """
#     
#     """
# 
# 

# ## Run baseline model pipeline with full dataset

# pipeline_factory = PipelineComponentsFactory(
#   infiles_dict_ser=infiles_dict_ser, output_config_ser=output_config_ser,
#   transform_dir=tr_dir, user_id_max=user_id_max, movie_id_max=movie_id_max,
#   n_genres=n_genres, n_age_groups=n_age_groups, min_eval_size=MIN_EVAL_SIZE,
#   serving_model_dir=serving_model_dir,
# )
# 
# beam_pipeline_args = [
#   '--direct_running_mode=multi_processing',
#   '--direct_num_workers=0'
#     ]

# baseline_components = pipeline_factory.build_components(MODEL_TYPE.BASELINE)
#     
# # create baseline model
# my_pipeline = tfx.dsl.Pipeline(
#   pipeline_name=PIPELINE_NAME,
#   pipeline_root=PIPELINE_ROOT,
#   components=baseline_components,
#   enable_cache=ENABLE_CACHE,
#   metadata_connection_config=metadata_connection_config,
#   beam_pipeline_args=beam_pipeline_args,
# )
# 
# tfx.orchestration.LocalDagRunner().run(my_pipeline)

# artifact_types = store.get_artifact_types()
# logging.debug(f"MLMD store artifact_types={artifact_types}")
# artifacts = store.get_artifacts()
# logging.debug(f"MLMD store artifacts={artifacts}")
# 
# components = pipeline_factory.build_components(MODEL_TYPE.PRODUCTION)
# # simulate experimentation of one model family
# my_pipeline = tfx.dsl.Pipeline(
#   pipeline_name=PIPELINE_NAME,
#   pipeline_root=PIPELINE_ROOT,
#   components=components,
#   enable_cache=ENABLE_CACHE,
#   metadata_connection_config=metadata_connection_config,
#   beam_pipeline_args=beam_pipeline_args,
# )
# 
# tfx.orchestration.LocalDagRunner().run(my_pipeline)
# 

# artifact_types = store.get_artifact_types()
# print(f"MLMD store artifact_types={artifact_types}")
# artifacts = store.get_artifacts()
# print(f"MLMD store artifacts={artifacts}")
# 
# executions = store.get_executions()
# logging.debug(f"MLMD store executions={executions}")
# 
# # executions has custom_properties.key: "infiles_dict_ser"
# #    and custom_properties.key: "output_config_ser"
# artifact_count = len(artifacts)
# execution_count = len(executions)
# 
