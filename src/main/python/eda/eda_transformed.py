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
pl.Config.set_fmt_str_lengths(900)
pl.Config.set_fmt_table_cell_list_len(900)
colors = px.colors.qualitative.Plotly 

genres_set = set(genres)

#make co-occurence matrix and heatmap
def genres_co_occurence_heatmap(exploded, split_name:str, rating:int,
    item:str="genres"):
  
    unique_items = exploded[item].unique().to_list()
    
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
    
    #genres are normalized multi-hot
    df = df.with_columns(
        pl.col("genres").map_elements(movie_lens_utils.deserialize, return_dtype=pl.List(pl.Float32))
    )
    
    # add an index for row number:
    df = df.with_row_index("row")
    df = df.with_columns(
      (pl.col("rating") * 5).round(0).cast(pl.Int32).alias("rating")
    )
    df = df.with_columns(
      pl.col("*").exclude(['genres']).round(0).cast(pl.Int32)
    )
    #because 'genres' is normalized multi-hot, use ceil before cast
    df = df.with_columns(
      pl.col("genres").list.eval(pl.element().ceil().cast(pl.Int32)).alias("genres")
    )
    
    #write the genres to sparse indexes
    df = df.with_columns(
      pl.col("genres")
      .list.eval((pl.element() == 1).arg_true())
      .alias("sparse_genres_str")
    )
    name_series = pl.Series(genres)
    df = df.with_columns(
      pl.col("sparse_genres_str")
      .list.eval(
        pl.element().map_elements(lambda x: name_series[x], return_dtype=pl.String)
      )
      .alias("sparse_genres_str")
    )
    """
    ┌───────┬─────┬────────┬────────────────────────────────────────────────────────┬─────┬───────┬───────┬──────────┬────────────┬────────┬─────────────┬─────────┬─────────┬──────┬──────────────────────┐
    │ row   ┆ age ┆ gender ┆ genres                                                 ┆ hr  ┆ hr_wk ┆ month ┆ movie_id ┆ occupation ┆ rating ┆ sec_into_yr ┆ user_id ┆ weekday ┆ yr   ┆ sparse_genres_str    │
    │ ---   ┆ --- ┆ ---    ┆ ---                                                    ┆ --- ┆ ---   ┆ ---   ┆ ---      ┆ ---        ┆ ---    ┆ ---         ┆ ---     ┆ ---     ┆ ---  ┆ ---                  │
    │ i32   ┆ i32 ┆ i32    ┆ list[i32]                                              ┆ i32 ┆ i32   ┆ i32   ┆ i32      ┆ i32        ┆ i32    ┆ i32         ┆ i32     ┆ i32     ┆ i32  ┆ list[str]            │
    ╞═══════╪═════╪════════╪════════════════════════════════════════════════════════╪═════╪═══════╪═══════╪══════════╪════════════╪════════╪═════════════╪═════════╪═════════╪══════╪══════════════════════╡
    │ 39363 ┆ 0   ┆ 0      ┆ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] ┆ 17  ┆ 126   ┆ 377   ┆ 938      ┆ 10         ┆ 4      ┆ 31512552    ┆ 1       ┆ 7       ┆ 2000 ┆ ["Musical"]          │
    │ 66350 ┆ 0   ┆ 0      ┆ [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] ┆ 17  ┆ 126   ┆ 377   ┆ 1270     ┆ 10         ┆ 5      ┆ 31510856    ┆ 1       ┆ 7       ┆ 2000 ┆ ["Comedy", "Sci-Fi"] │

    """
    
    # add label columns
    df = df.with_columns(
      pl.col("occupation").map_elements(
        lambda x: labels_dict['occupation'].get(x, x),
        return_dtype=pl.String).alias("occ")
    )
    
    #"""
    write_dist_corr_heatmap(df, skip_columns=set(["genres", "row", "occ", "sparse_genres_str"]), outfile_name=f"{split_name}_dist_corr_heatmap.png")
    
    exploded = df.with_columns(
        pl.col("genres").list.to_array(width=n_genres).arr.to_struct(
                fields=[f"{genres[i]}" for i in range(n_genres)]
            )
        ).unnest("genres")
    #print(f'columns:\n{exploded.columns}')
   
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
    
    extraction_expressions = [
      pl.col("genres").list.get(i).alias(genres[i]) for i in range(len(genres))
    ]
    exploded = df.with_columns(extraction_expressions).drop("genres")
    
    #print(f'exploded.head={exploded.head(5)}')
    #print(f'exploded.head sorted by user_id={exploded.sort("user_id").head(5)}')
    
    """
     ───────┬─────┬────────┬─────┬───────┬───────┬──────────┬────────────┬────────┬─────────────┬─────────┬─────────┬──────┬──────────────────────┬──────────────┬────────┬───────────┬───────────┬──────────┬────────┬───────┬─────────────┬───────┬─────────┬───────────┬────────┬─────────┬─────────┬─────────┬────────┬──────────┬─────┬─────────┐
    │ row   ┆ age ┆ gender ┆ hr  ┆ hr_wk ┆ month ┆ movie_id ┆ occupation ┆ rating ┆ sec_into_yr ┆ user_id ┆ weekday ┆ yr   ┆ sparse_genres_str    ┆ occ          ┆ Action ┆ Adventure ┆ Animation ┆ Children ┆ Comedy ┆ Crime ┆ Documentary ┆ Drama ┆ Fantasy ┆ Film-Noir ┆ Horror ┆ Musical ┆ Mystery ┆ Romance ┆ Sci-Fi ┆ Thriller ┆ War ┆ Western │
    │ ---   ┆ --- ┆ ---    ┆ --- ┆ ---   ┆ ---   ┆ ---      ┆ ---        ┆ ---    ┆ ---         ┆ ---     ┆ ---     ┆ ---  ┆ ---                  ┆ ---          ┆ ---    ┆ ---       ┆ ---       ┆ ---      ┆ ---    ┆ ---   ┆ ---         ┆ ---   ┆ ---     ┆ ---       ┆ ---    ┆ ---     ┆ ---     ┆ ---     ┆ ---    ┆ ---      ┆ --- ┆ ---     │
    │ i32   ┆ i32 ┆ i32    ┆ i32 ┆ i32   ┆ i32   ┆ i32      ┆ i32        ┆ i32    ┆ i32         ┆ i32     ┆ i32     ┆ i32  ┆ list[str]            ┆ str          ┆ i32    ┆ i32       ┆ i32       ┆ i32      ┆ i32    ┆ i32   ┆ i32         ┆ i32   ┆ i32     ┆ i32       ┆ i32    ┆ i32     ┆ i32     ┆ i32     ┆ i32    ┆ i32      ┆ i32 ┆ i32     │
    ╞═══════╪═════╪════════╪═════╪═══════╪═══════╪══════════╪════════════╪════════╪═════════════╪═════════╪═════════╪══════╪══════════════════════╪══════════════╪════════╪═══════════╪═══════════╪══════════╪════════╪═══════╪═════════════╪═══════╪═════════╪═══════════╪════════╪═════════╪═════════╪═════════╪════════╪══════════╪═════╪═════════╡
    │ 39363 ┆ 0   ┆ 0      ┆ 17  ┆ 126   ┆ 377   ┆ 938      ┆ 10         ┆ 4      ┆ 31512552    ┆ 1       ┆ 7       ┆ 2000 ┆ ["Musical"]          ┆ K-12 student ┆ 0      ┆ 0         ┆ 0         ┆ 0        ┆ 0      ┆ 0     ┆ 0           ┆ 0     ┆ 0       ┆ 0         ┆ 0      ┆ 1       ┆ 0       ┆ 0       ┆ 0      ┆ 0        ┆ 0   ┆ 0       │
    │ 43661 ┆ 0   ┆ 0      ┆ 17  ┆ 126   ┆ 377   ┆ 1035     ┆ 10         ┆ 5      ┆ 31512552    ┆ 1       ┆ 7       ┆ 2000 ┆ ["Musical"]          ┆ K-12 student ┆ 0      ┆ 0         ┆ 0         ┆ 0        ┆ 0      ┆ 0     ┆ 0           ┆ 0     ┆ 0       ┆ 0         ┆ 0      ┆ 1       ┆ 0       ┆ 0       ┆ 0      ┆ 0        ┆ 0   ┆ 0       │
    │ 61139 ┆ 0   ┆ 0      ┆ 17  ┆ 126   ┆ 377   ┆ 1246     ┆ 10         ┆ 4      ┆ 31512892    ┆ 1       ┆ 7       ┆ 2000 ┆ ["Drama"]            ┆ K-12 student ┆ 0      ┆ 0         ┆ 0         ┆ 0        ┆ 0      ┆ 0     ┆ 0           ┆ 1     ┆ 0       ┆ 0         ┆ 0      ┆ 0       ┆ 0       ┆ 0       ┆ 0      ┆ 0        ┆ 0   ┆ 0       │
    │ 66350 ┆ 0   ┆ 0      ┆ 17  ┆ 126   ┆ 377   ┆ 1270     ┆ 10         ┆ 5      ┆ 31510856    ┆ 1       ┆ 7       ┆ 2000 ┆ ["Comedy", "Sci-Fi"] ┆ K-12 student ┆ 0      ┆ 0         ┆ 0         ┆ 0        ┆ 1      ┆ 0     ┆ 0           ┆ 0     ┆ 0       ┆ 0         ┆ 0      ┆ 0       ┆ 0       ┆ 0       ┆ 1      ┆ 0        ┆ 0   ┆ 0       │
    │ 82572 ┆ 0   ┆ 0      ┆ 18  ┆ 132   ┆ 377   ┆ 1545     ┆ 10         ┆ 4      ┆ 498939      ┆ 1       ┆ 6       ┆ 2001 ┆ ["Drama"]            ┆ K-12 student ┆ 0      ┆ 0         ┆ 0         ┆ 0        ┆ 0      ┆ 0     ┆ 0           ┆ 1     ┆ 0       ┆ 0         ┆ 0      ┆ 0       ┆ 0       ┆ 0       ┆ 0      ┆ 0        ┆ 0   ┆ 0       │
    └
    """
    
    for rating in [5, 4, 3, 2, 1]:
        filtered = df.filter(pl.col("rating") == rating)
        #pie chart counting each genre.  explode the sparse strings into their own rows
        filtered = filtered.select(['row', 'sparse_genres_str'])
        filtered = filtered.explode('sparse_genres_str')
        genres_counts = filtered['sparse_genres_str'].value_counts()#.reset_index()
        genres_counts.columns = ['Genre', 'Count']
        genres_counts = genres_counts.with_columns(pl.col('Count').cast(pl.Int64))
        #print(f'dtypes={genres_counts.dtypes}')
        #print(f'head={genres_counts.head(5)}')
        fig = px.pie(genres_counts, values='Count', names='Genre',
            title=f'{split_name} rating {rating}: Genres')
        fig.write_image(os.path.join(img_dir, f"{split_name}_genres_rating_{rating}_pie.png"))
  
        genres_co_occurence_heatmap(filtered, split_name, rating, "sparse_genres_str")

        
print(f'wrote multivariate EDA images to {img_dir}')
