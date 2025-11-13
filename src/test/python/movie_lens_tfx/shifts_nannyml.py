import unittest
import os
import sys
import numpy as np
import pandas as pd
import nannyml as nml
from IPython.display import display
from tfx.orchestration import metadata
from ml_metadata.metadata_store import metadata_store

from helper import *

genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
          'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
          'Thriller', 'War', 'Western']

class NannyMLTests(unittest.TestCase):
  
  def setUp(self):
    PIPELINE_ROOT = os.path.join(get_bin_dir(), "local_notebook/1/TestPipelines")
    if not os.path.exists(PIPELINE_ROOT):
      self.fail('needs  pipeline to have been run for local_notebook')
    METADATA_PATH = os.path.join(PIPELINE_ROOT, 'tfx_metadata', 'metadata.db')
    metadata_connection_config = metadata.sqlite_metadata_connection_config(
      METADATA_PATH)
    store = metadata_store.MetadataStore(metadata_connection_config)
    
    #find latest model
    artifact_list = store.get_artifacts_by_type("Model")
    latest_artifact = sorted(artifact_list, key=lambda
      x: x.create_time_since_epoch, reverse=True)[0]
    model_uri = os.path.join(latest_artifact.uri, "Format-Serving")
    logging.debug(f"model_uri={model_uri}")
    loaded_saved_model = tf.saved_model.load(model_uri)
    self.infer_twotower = loaded_saved_model.signatures["serving_default"]
    self.transform_raw = loaded_saved_model.signatures["transform_features"]
    
    #find latest raw examples for test split
    examples_list = store.get_artifacts_by_type("Examples")
    for artifact in examples_list:
      if "MovieLensExampleGen" in artifact.uri:
        test_examples_uri = os.path.join(artifact.uri, "Split-test")
        file_paths = [os.path.join(test_examples_uri, name) for name
          in os.listdir(test_examples_uri)]
        self.ratings_test_ds_ser = tf.data.TFRecordDataset(file_paths,
          compression_type="GZIP")
        
    self.assertIsNotNone(self.ratings_test_ds_ser)
    self.assertIsNotNone(self.transform_raw)
  
  # preparing datasets for use with NannyML
  def infer_rating(self, ds, drop_rating: bool = False, incl_genres:bool=False, batch_size: int = 32):
    INPUT_KEY = list(self.infer_twotower.structured_input_signature[1].keys())[0]
    TR_INPUT_KEY = list(self.transform_raw.structured_input_signature[1].keys())[0]
    batched_ds = ds.batch(batch_size)
    data_list = []
    true_ratings = []
    predicted_ratings = []
    for serialized_batch in batched_ds:
      transformed_batch = self.transform_raw(**{TR_INPUT_KEY: serialized_batch})
      processed_batch = {k: v.numpy().squeeze(axis=1) for k, v in transformed_batch.items()}
      ##for genres, each entry is a multi-hot array so expand into their own columns
      ## or make a string of the array at expense of losing information.
      if incl_genres:
        for k in genres:
          processed_batch[k] = []
        for sample_index, mh in enumerate(processed_batch['genres']):
          for mh_index, mh_value in enumerate(mh):
            processed_batch[genres[mh_index]].append(mh_value)
      del processed_batch['genres']
      start_of_years = np.array([f'{int(year)}-01-01'
        for year in processed_batch['yr'].astype(int)], dtype='datetime64[D]')
      time_deltas = processed_batch['sec_into_yr'].astype('timedelta64[s]')
      processed_batch['timestamp'] = start_of_years + time_deltas
      prediction = self.infer_twotower(**{INPUT_KEY: serialized_batch})['outputs'].numpy().reshape(-1)
      processed_batch['y_pred'] = prediction
      predicted_ratings.extend(prediction.tolist())
      true_ratings.extend(processed_batch['rating'].tolist())
      if drop_rating:
        del processed_batch['rating']
      data_list.append(processed_batch)
    data = {key:[] for key in data_list[0]}
    for _dict in data_list:
      for key, value in _dict.items():
        if isinstance(value, list):
          data[key].extend(value)
        else:
          data[key].extend(value.tolist())
    for key in data:
      data[key] = np.array(data[key])
    df = pd.DataFrame(data)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    df['timestamp']=df['timestamp'].astype(object)
    return df, np.array(true_ratings), np.array(predicted_ratings)
  
  def test_model_perf_monitoring(self):
    """
    estimate performance before the ground truth is collected.
    
    
    
    """
    df_test, true_analysis, predicted_analysis = self.infer_rating(
      self.ratings_test_ds_ser, incl_genres=False, batch_size=1000)
    n = len(df_test)
    df_reference = df_test.head(n // 2)
    df_analysis = df_test.tail(n - (n // 2))
    
    cols = df_analysis.columns.tolist().copy()
    cols.remove('y_pred')
    
    estimator = nml.DLE(
      feature_column_names=cols,
      y_pred='y_pred',
      y_true='rating',
      timestamp_column_name='timestamp',
      metrics=['rmse', 'mae'],
      chunk_size=6000,
      tune_hyperparameters=False
    )
    estimator.fit(df_reference)
    results = estimator.estimate(df_analysis)
    display(results.filter(period='analysis').to_df())
    display(results.filter(period='reference').to_df())
    metric_fig = results.plot()
    metric_fig.show()
    
    #one at a time
    for col in cols:
      if col == 'timestamp':
        continue
      estimator = nml.DLE(
        feature_column_names=[col],
        y_pred='y_pred',
        y_true='rating',
        timestamp_column_name='timestamp',
        metrics=['rmse', 'mae'],
        chunk_size=6000,
        tune_hyperparameters=False
      )
      estimator.fit(df_reference)
      results = estimator.estimate(df_analysis)
      display(results.filter(period='analysis').to_df())
      display(results.filter(period='reference').to_df())
      metric_fig = results.plot()
      metric_fig.show()

if __name__ == '__main__':
  unittest.main()