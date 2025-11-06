#!/usr/bin/env python
# coding: utf-8

# run this after local notebook has run the PRE-PROCESSING pipeline build

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

import polars as pl

from scipy.stats.distributions import chi2
from collections import OrderedDict
import re
import io
from datetime import datetime
import pytz
import dcor
import numpy as np
import plotly.express as px

pl.Config.set_fmt_str_lengths(900)

def can_reject_indep(x : np.array, y:np.array, alpha:float = 0.05, debug:bool=False):
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

img_dir = os.path.join(get_bin_dir(), "local_notebook", "images")
os.makedirs(img_dir, exist_ok=True)

dfs = {}

for key in file_paths:
    processed_buffer = io.StringIO()
    file_path = file_paths[key]
    schema = schemas[key]
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

    dfs[key] = df

    if key=="movies":
        df = df.with_columns(
          pl.col("genres").str.replace("Children's", "Children")
        )
        df = df.with_columns(
          pl.col("genres").str.split("|")
        )
        movie_genres = df.explode('genres')
        ordered_genres = movie_genres['genres'].value_counts().sort('count', descending=True)
        fig = px.bar(ordered_genres, x="genres", y="count", title="genres histogram",)
        fig.write_image(os.path.join(img_dir, "genres_hist.png"))
    if key=="ratings":
        #user_id, movie_id, rating, timestamp
        fig = px.histogram(df, x='rating', title='rating')
        fig.write_image(os.path.join(img_dir, "rating_hist.png"))
        fig = px.histogram(df, x='timestamp', title='timestamp')
        fig.write_image(os.path.join(img_dir, "timestamp_hist.png"))
        fig = px.histogram(df, x='movie_id', title='movie_id')
        fig.write_image(os.path.join(img_dir, "movieid_hist.png"))
        fig = px.histogram(df,  x='user_id', title='user_id')
        fig.write_image(os.path.join(img_dir, "userid_hist.png"))
        #run the ndep tests on transformed data instead of raw data
        #x = df.select(pl.col("rating")).to_numpy()
        #y = df.select(pl.col("timestamp")).to_numpy()
        #print(f"rating, timestamp are indep: {can_reject_indep(x, y, 0.05, True)}")
        fig = px.density_heatmap(df, x='movie_id', y='rating')
        fig.write_image(os.path.join(img_dir, "movieid_rating_heatmap.png"))
        fig = px.density_heatmap(df, x='timestamp', y='rating')
        fig.write_image(os.path.join(img_dir, "timestamp_rating_heatmap.png"))
        fig = px.density_heatmap(df, x='user_id', y='rating')
        fig.write_image(os.path.join(img_dir, "userid_rating_heatmap.png"))
        fig = px.density_heatmap(df, x='timestamp', y='movie_id')
        fig.write_image(os.path.join(img_dir, "timestamp_movieid_heatmap.png"))
        #fig = px.scatter_ternary(df, a="rating", b="timestamp", c="movie_id",
        #    #size="total", size_max=15,
        #    color_discrete_map = {"rating": "blue", "timestamp": "green", "movie_id":"red"} )
        #fig.show(renderer='notebook')
        #fig = px.scatter_ternary(df, a="rating", b="user_id", c="movie_id",
        #    #size="total", size_max=15,
        #    color_discrete_map = {"rating": "blue", "user_id": "green", "movie_id":"red"} )
        #fig.show(renderer='notebook')
    if key=="users":
        #user_id, gender, age, occupation, zipcode
        fig = px.histogram(df, x='gender', title='gender')
        fig.write_image(os.path.join(img_dir, "gender_hist.png"))
        fig = px.histogram(df, x='age',  title='age')
        fig.write_image(os.path.join(img_dir, "age_hist.png"))
        df = df.with_columns(
            pl.col("occupation").map_elements(lambda x: labels_dict['occupation'].get(x,x)).alias("occ")
        )
        df = df.with_columns(
            pl.col("occ").cast(pl.Categorical)
        )
        ordered_occupation = df['occ'].value_counts().sort('count', descending=True)
        fig = px.bar(ordered_occupation, x="occ", y="count", title="occupation histogram",)
        fig.update_xaxes(tickangle=45)
        fig.write_image(os.path.join(img_dir, "occupation_hist.png"))
        fig = px.histogram(df, x='zipcode',  title='zipcode')
        fig.write_image(os.path.join(img_dir, "zipcode_hist.png"))
        #run the ndep tests on transformed data instead of rrrawawwa data
        _features=['gender', 'age', 'occupation', 'zipcode']
        for ii, feature in enumerate(_features):
            for jj in range(ii+1, len(_features)):
                feature2 = _features[jj]
                if feature2 == 'occupation':
                    occ_counts = df.group_by("occ").len().rename({"len": "occ_count"})
                    df_sorted = (
                        df.join(occ_counts, on="occ", how="left")
                        .sort(by="occ_count", descending=True)
                        .drop("occ_count")
                    )
                    fig = px.density_heatmap(df_sorted, x=feature, y=feature2)
                else:
                    fig = px.density_heatmap(df, x=feature, y=feature2)
                fig.write_image(os.path.join(img_dir, f"{feature}_{feature2}_heatmap.png"))
                #for kk in range(jj+1, len(_features)):
                #    feature3 = _features[kk]
                #    fig = px.scatter_ternary(df, a=feature, b=feature2, c=feature3,
                #        #size="total", size_max=15,
                #        color_discrete_map = {feature: "blue", feature2: "green", feature3:"red"} )
                #    fig.show(renderer='notebook')

df_no_match = dfs['movies'].join(dfs['ratings'], on="movie_id", how="anti")
print(f'{len(df_no_match)} movies were not rated')
df_no_match = dfs['users'].join(dfs['ratings'], on="user_id", how="anti")
print(f'{len(df_no_match)} users did not rate')

for key in dfs:
  print(f'\n{key}: HEAD')
  print(dfs[key].head())
  print(f'\n{key}: DESCRIBE')
  print(dfs[key].describe)

del df
del dfs

print(f'\nimages were written to {img_dir}')