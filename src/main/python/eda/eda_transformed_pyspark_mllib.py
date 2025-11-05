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


#set variable if not defined
if 'PLOT_PREFIXSPAN' not in locals() and 'PLOT_PREFIXSPAN' not in globals():
  PLOT_PREFIXSPAN = False

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

# to look at common behaviors: make MIN_SUPPORT large and MIN_ITEMSET_SIZE small
# to look at the rare, but actionaable: make MISUPPORT small and MIN_ITEMSIZE large

MIN_SUPPORT = 0.005  # items must appear in MIN_SUPPORT*100% of transactions
MIN_ITEMSET_SIZE = 4 # min size of itemset
MIN_CONFIDENCE = 0.6  # The rule X => Y is valid if MIN_CONFIDENCE*100% of transactions containing X also contain Y.3

MIN_SUPPORT_PREFIXSPAN = 0.005
MAX_PATTERN_LENGTH = 20 #max seqeuntail pattern length for PrefixSpan

def _plot_fpgrowth_assoc_rules(model, split_name, rating, is_inferrence:bool=False):
  # Generate the association rules
  # This returns a PySpark DataFrame with columns: antecedent, consequent, confidence, lift, support
  association_rules_spark = model.associationRules.filter(
    model.associationRules.confidence >= MIN_CONFIDENCE
  ).filter(
    # Filter out rules where the antecedent (X) is a single item (size > 1)
    # This often focuses the analysis on complex rules, but you can remove this filter if needed.
    spark_size(model.associationRules.antecedent) > 1
  )
  
  # Convert PySpark results to Pandas
  rules_pd = association_rules_spark.toPandas()
  
  # Format the Antecedent (X) and Consequent (Y) for visualization
  def format_itemset(itemset):
    """Converts a list of items into a readable string format {item1, item2}"""
    return "{" + ", ".join(itemset) + "}"
  
  # rules_pd['Antecedent_X'] = rules_pd['antecedent'].apply(format_itemset)
  # rules_pd['Consequent_Y'] = rules_pd['consequent'].apply(format_itemset)
  rules_pd['Antecedent_X'] = rules_pd['antecedent'].apply(json.dumps)
  rules_pd['Consequent_Y'] = rules_pd['consequent'].apply(json.dumps)
  rules_pd['Rule_Label'] = rules_pd.apply(
    lambda row: f"{row['Antecedent_X']} => {row['Consequent_Y']}",
    axis=1
  )
  
  print(f'{split_name}, rating {rating}, is_inferrence={is_inferrence}')
  # Sort by Lift for the most interesting rules
  rules_pd = rules_pd.sort_values(by='lift', ascending=False)
  print(f"Generated Rules (Top 5 by Lift):")
  print(
    rules_pd[['Rule_Label', 'confidence', 'lift', 'support']].head())
  if len(rules_pd) == 0:
    return
  
  # Create the Scatter Plot
  if is_inferrence:
    title = f'{split_name}, rating {rating} inferrence: Association Rules Analysis (Min Confidence: {MIN_CONFIDENCE})'
  else:
    title = f'{split_name}, rating {rating}: Association Rules Analysis (Min Confidence: {MIN_CONFIDENCE})'
    
  fig_rules = px.scatter(
    rules_pd,
    x='confidence',
    y='lift',
    size='support',  # Use support to determine the size of the bubble
    color='lift',  # Color the points based on the lift value
    hover_name='Rule_Label',  # Show the rule text on hover
    title=title,
    labels={'confidence': 'Confidence (P(Y|X))',
            'lift': 'Lift (Interestingness)'},
    template='plotly_white'
  )
  
  # Add a horizontal line at Lift = 1.0 for visual reference
  fig_rules.add_hline(
    y=1.0,
    line_dash="dash",
    line_color="red",
    annotation_text="Independence (Lift=1.0)"
  )
  
  fig_rules.update_traces(
    mode='markers',
    marker=dict(sizemode='area',
                sizeref=2. * max(rules_pd['support']) / (15 ** 2),
                sizemin=4)
  )
  if is_inferrence:
    path = f"{split_name}_movies_assoc_rules_rating_{rating}_inferred.png"
  else:
    path = f"{split_name}_movies_assoc_rules_rating_{rating}.png"
  fig_rules.write_image(os.path.join(img_dir, path))

def stop_session(spark):
  try:
    if spark.getActiveSession():
      spark.stop()
    return
  except Exception:
    spark.stop()

def movies_prefixspan(df, split_name, rating):
  """use pyspark's PrefixSpan MLLib model"""
  num_logical_cores = multiprocessing.cpu_count()
  print(f"start spark session for {num_logical_cores} local cores")
  num_logical_cores="*"
  
  spark = SparkSession.builder.appName(f"PrefixSpan_{split_name}_{rating}").master(f"local[{num_logical_cores}]").getOrCreate()
  
  df = df.with_columns(pl.col("movie_ids").cast(pl.List(pl.Int32)))
  print(f"{split_name} prefixscan input df dtype={df.dtypes}")
  print(f'movies_prefixspan len={len(df)}, head=\n{df.head(5)}')
  
  pandas_df = df.to_pandas()
  
  def cast_list_to_python_ints(int_list):
    """Casts all elements in a list from NumPy int32 to Python int."""
    if int_list is None:
      return None
    # This explicit list comprehension forces the conversion to Python int
    return [int(x) for x in int_list]
  
  pandas_df['movie_ids'] = pandas_df['movie_ids'].apply(cast_list_to_python_ints)
  
  def sequence_to_itemsets(sequence):
    """Converts a sequence of items [1, 2, 3] to a sequence of itemsets [[1], [2], [3]]"""
    return [[item] for item in sequence]
  
  pandas_df['sequence_col'] = pandas_df['movie_ids'].apply(sequence_to_itemsets)
  print(f'{split_name}_{rating:} prefixscan input pandas_df.head={pandas_df.head(5)}')
  print(f'dtypes={pandas_df.dtypes}')

  schema = StructType([
    StructField("sequence_col",
                ArrayType(ArrayType(IntegerType())), False)
  ])
  
  #schema = StructType([
  #  StructField("sequence_col",
  #              ArrayType(IntegerType(), containsNull=False), True)
  #])
  
  spark_df = spark.createDataFrame(
    pandas_df[['sequence_col']],schema=schema
  )
  
  prefix_span = PrefixSpan(
    minSupport=MIN_SUPPORT_PREFIXSPAN,
    maxPatternLength=MAX_PATTERN_LENGTH,
    sequenceCol="sequence_col")
    
  #prefix_span.getMaxLocalProjDBSize()
  #prefix_span.getSequenceCol()
  
  patterns_df = (prefix_span.findFrequentSequentialPatterns(spark_df)
      .orderBy("freq", ascending=False).show(truncate=False))
  
  #has cols "sequence" which is an array of arrays of the pattern
  # and "freq" is a 64 bit integer of the freq/support count
  
  if patterns_df is None or patterns_df.count() == 0:
    stop_session(spark)
    return
  
  try:
    patterns_pd = patterns_df.toPandas()
    if len(patterns_pd) == 0:
      stop_session(spark)
      return
  except Exception as e:
    stop_session(spark)
    return
  
  print(f'patterns_pd head\n{patterns_pd.head()}')
  
  # Convert the sequence (Array<Array<Item>>) to a readable string
  patterns_pd['pattern_str'] = patterns_pd['sequence'].apply(
    lambda s: ' -> '.join(['({})'.format(', '.join(map(str, step))) for step in s]))
  # Sort and select top N
  TOP_N = 20
  top_n = patterns_pd.nlargest(TOP_N, 'freq')
  
  # 2. Create the Plot
  fig = px.bar(
    top_n,
    x='pattern_str',
    y='freq',
    title=f'{split_name}, rating {rating}: Top {TOP_N} Frequent Sequential Patterns',
    labels={'pattern_str': 'Sequential Pattern',
            'freq': 'Frequency (Support Count)'},
    hover_data={'sequence': False, 'freq': True}
    # Optionally hide the raw sequence in hover
  )
  
  fig.update_layout(xaxis={'categoryorder': 'total descending'})
  fig.write_image(os.path.join(img_dir,
    f"{split_name}_movies_prefixspan_rating_{rating}_bar.png"))
  
  #plot a treemap or sunburst
  # ... continuing from the previous setup
  # Add a length column
  patterns_pd['length'] = patterns_pd['sequence'].apply(len)
  
  #TODO: may need to reduce size of this?
  # Create a Treemap/Sunburst
  # Use `length` and a string representation of the sequence as the hierarchy
  fig = px.sunburst(
    patterns_pd,
    path=['length', 'pattern_str'],  # Defines the hierarchy
    values='freq',
    # Size of the sector/area is proportional to frequency
    title=f'{split_name}, rating {rating}: Sequential Pattern Frequency by Length and Pattern'
  )
  
  fig.write_image(os.path.join(img_dir,
    f"{split_name}_movies_prefixspan_rating_{rating}_sunburst.png"))
  
  #scatter
  fig = px.scatter(
    patterns_pd,
    x='length',
    y='freq',
    size='freq',  # Use frequency to determine marker size
    color='length',  # Color markers by length
    hover_name='pattern_str',
    title=f'{split_name}, rating {rating}: Pattern Length vs. Frequency',
    labels={'length': 'Pattern Length',
            'freq': 'Frequency (Support Count)'}
  )
  fig.write_image(os.path.join(img_dir,
    f"{split_name}_movies_prefixspan_rating_{rating}_scatter.png"))
  stop_session(spark)

def movies_frequent_itemsets(df, split_name, rating):
    """use pyspark's FPGrowth MLLib model.
    Note that ideally, these are model families in the pipeline Tuner stage, but here, using it
    for EDA"""
    # Support measures how frequently an itemset appears in the data.
    #  reducing min support increases the length of the itemsets in the results, genreally.
    # Confidence measures the reliability of an "if-then" rule, indicating the probability of an item 
    #  appearing given another has appeared. 
    #  reliability of P(Y|X)
    # Lift compares the observed confidence to what would be expected if the items were independent, 
    #  showing the strength of the association. 
    #  Lift > 1: items are purchased together more often than expected by chance
    #  Lift < 1: items are purchased together less often than expected.
    #  Lift = 1: The items are independent.
    #gemini was used in this method, especially helpful in resolving version related data formatting.
    
    num_logical_cores = multiprocessing.cpu_count()
    print(f"start spark session for {num_logical_cores} local cores")
    num_logical_cores="*"
    
    spark = (SparkSession.builder.appName(f"FPGrowth_{split_name}_{rating}")
      .master(f"local[{num_logical_cores}]").getOrCreate())
    
    df = df.select(['movie_ids'])
    #df = df.with_columns(pl.col("movie_ids").cast(pl.List(pl.Int32)))
    print(f"dtype={df.dtypes}")
    print(f'head={df.head(5)}')
    
    pandas_df = df.to_pandas()
    print(f'pandas_df.head={pandas_df.head(5)}')
    
    def cast_list_to_python_ints(int_list):
      """Casts all elements in a list from NumPy int32 to Python int."""
      if int_list is None:
        return None
      # This explicit list comprehension forces the conversion to Python int
      return [int(x) for x in int_list]
    
    pandas_df['movie_ids'] = pandas_df['movie_ids'].apply(cast_list_to_python_ints)
    print(f'pandas_df.dtypes={pandas_df.dtypes}')
    
    schema = StructType([
      StructField("movie_ids", ArrayType(IntegerType(), containsNull=False),True)
    ])
    spark_df = spark.createDataFrame(pandas_df, schema=schema)
    print(f'spark_df schema=')
    spark_df.printSchema()
    print(f'head: {spark_df.take(2)}')
    
    """
    #using pyarrow failed for this older pyspark library
    arrow_table = _df.to_arrow()
    single_chunk_table = arrow_table.combine_chunks()
    schema = StructType([
      StructField("genres", ArrayType(StringType(), containsNull=True), True)
    ])
    spark_df = spark.createDataFrame(single_chunk_table, schema=schema)
    spark_df.printSchema()
    spark_df.show()
    """
    print(f'construct FPGrowth model')
    fp_growth = FPGrowth(
        itemsCol="movie_ids",
        minSupport=MIN_SUPPORT, minConfidence=MIN_CONFIDENCE,
    )
    model = fp_growth.fit(spark_df)
    frequent_itemsets_spark = model.freqItemsets
    print(f'TYPE={type(frequent_itemsets_spark)}')
    try:
      if frequent_itemsets_spark.isEmpty():
        stop_session(spark)
        return
    except Exception:
      stop_session(spark)
      return
    
    print(f"\nfrequent_itemsets_spark:")
    frequent_itemsets_spark.show(truncate=200)
    
    filtered_itemsets = frequent_itemsets_spark.filter(
      F.size(F.col("items")) >= MIN_ITEMSET_SIZE
    )
    print(f"\nfiltered_itemsets:")
    filtered_itemsets.show(truncate=200)
    
    # Collect results and convert to Pandas for Plotly
    # Collecting to the driver is only feasible for results that fit in memory.
    results_pd = filtered_itemsets.toPandas()
    
    # Prepare data for visualization (e.g., convert list of items to a comma-separated string)
    #results_pd['itemset_str'] = results_pd['items'].apply(lambda x: ", ".join(x))
    results_pd['itemset_str'] = results_pd['items'].apply(json.dumps)

    # Calculate Support for visualization (freq / total_transactions)
    total_transactions = spark_df.count()
    results_pd['support'] = results_pd['freq'] / total_transactions
    
    results_pd = results_pd.sort_values(by='support', ascending=False)
    
    print(f"results_pd len={len(results_pd)}")
    print(f"sorted by support={results_pd.head(5)}")
    
    if len(results_pd) == 0:
      stop_session(spark)
      return
    
    # Visualize with Plotly Express (Bar Chart is common)
    fig = px.bar(
        results_pd.head(100),
        x='itemset_str',
        y='support',
        title=f'{split_name}, rating {rating}, Frequent movie Itemsets (Min Support: {MIN_SUPPORT*100}%, Min Size: {MIN_ITEMSET_SIZE})',
        labels={'itemset_str': 'Itemset', 'support': 'Support (%)'},
        color='itemset_str',
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_yaxes(tickformat=".2%") # Format y-axis as percentage
    fig.write_image(os.path.join(img_dir, 
        f"{split_name}_movies_itemsets_rating_{rating}.png"))

    #sort results_pd by itemset length, support
    results_pd['itemset_len'] = results_pd['items'].apply(len)
    results_pd = results_pd.sort_values(by=['itemset_len', 'support'], ascending=[False,False])
    print(f"sorted by itemset_length then support={results_pd.head(5)}")
    fig = px.bar(
      results_pd.head(100),
      x='itemset_str',
      y='support',
      title=f'{split_name}, rating {rating},  movie Itemsets (Min Support: {MIN_SUPPORT * 100}%, Min Size: {MIN_ITEMSET_SIZE})',
      labels={'itemset_str': 'Itemset', 'support': 'Support (%)'},
      color='itemset_str',
      color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_yaxes(tickformat=".2%")  # Format y-axis as percentage
    fig.write_image(os.path.join(img_dir,
                                 f"{split_name}_movies_itemsets_rating_{rating}_2.png"))
    
    #with the fpgrowth model, we can extract association rules, lift and confidence
    
    # inference with trained model:
    # ideally, would use the eval to tune MIN_SUPPORT, MIN_ITEMSET_SIZE, and MIN_CONFIDENCE for fpgrowth and prefix scan
    saved_path = os.path.join(PIPELINE_ROOT, "spark_fpgrowth")
    os.makedirs(saved_path, exist_ok=True)
    if split_name == "train":
      model.write().overwrite().save(saved_path)
    elif split_name == "test" and os.path.exists(saved_path):
      model_infer = FPGrowth.load(saved_path)
      predictions = model_infer.transform(spark_df)
      if not predictions.isEmpty():
        _plot_fpgrowth_assoc_rules(model_infer, split_name, rating, True)
      del model_infer
    
    _plot_fpgrowth_assoc_rules(model, split_name, rating, False)
    
    stop_session(spark)
    
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
      pl.col("movie_id").cast(pl.Int32)
    )
    df = df.with_columns(pl.col('yr').cast(pl.Int32))
    df = df.with_columns(pl.col('sec_into_yr').cast(pl.Int32))
    
    df = df.with_columns(
        pl.col("occupation").map_elements(lambda x: labels_dict['occupation'].get(x,x)).alias("occ")
    )
    #df = df.with_columns(
    #    pl.col("occ").cast(pl.Categorical)
    #)
    
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
        fig.write_image(os.path.join(img_dir, f"{split_name}_genres_rating_{rating}.png"))
  
        filtered = df_sparse_str.filter(pl.col("rating") == rating)

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
        
        movies_frequent_itemsets(filtered, split_name, rating)
        del filtered
    
        if PLOT_PREFIXSPAN:
          #for prefixspan and sequence modeling, need timestamp in order to order the movies:
          filtered = df.filter(pl.col("rating") == rating)
          #print(f'dtypes={filtered.dtypes}')
          filtered = filtered.with_columns(
            (pl.col("yr").cast(pl.Utf8).str.strptime(pl.Datetime, format="%Y")
              + pl.duration(seconds=pl.col("sec_into_yr"))
            ).alias("timestamp")
          )
          print(f'columns={filtered.columns}')
          filtered = filtered.sort(["user_id", "timestamp"])
          filtered = filtered.select(["user_id", "movie_id"])
          filtered = filtered.group_by("user_id").agg(pl.col("movie_id").unique().alias("movie_ids"))
          filtered = filtered.select(["movie_ids"])
          movies_prefixspan(filtered, split_name, rating)
        
print(f'wrote multivariate EDA images to {img_dir}')
