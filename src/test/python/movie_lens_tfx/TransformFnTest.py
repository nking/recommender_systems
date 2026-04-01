import unittest

import tensorflow as tf

from movie_lens_tfx.transform_movie_lens import _transform_timestamp
from datetime import datetime, timedelta, timezone

class TransformFnTest(unittest.TestCase):
  def test0(self):
    data = [[978301752], [978824139]]
    concrete_tensor = tf.constant(data, dtype=tf.int64)
    inputs = {'timestamp': concrete_tensor}
    outputs = {}
    _transform_timestamp(inputs['timestamp'], outputs)
    
    #print(f'outputs={outputs}')
    chicago_tz_offset = 18000
    #recompose ts from "yr" and "sec_into_yr"
    for timestamp, yr, sec_into_yr in zip(inputs['timestamp'], outputs['yr'], outputs['sec_into_yr']):
      
      yr = int(yr.numpy().item())
      sec_into_yr = int(sec_into_yr.numpy().item())
      timestamp = int(timestamp.numpy().item())
      
      start_of_year = datetime(yr, 1, 1, tzinfo=timezone.utc)
      elapsed_time = timedelta(seconds=sec_into_yr)
      target_datetime = start_of_year + elapsed_time
      #this includes negative chicago offset
      epoch_timestamp = int(target_datetime.timestamp())
      #seems the leap-year 2000 estimate is 1 day less than expected
      #print(f'expected={timestamp}, calc={est}')
    
    #$$Sunday, December 31, 2000 22:29:12 UTC
    #Saturday, January 6, 2001 23:35:39 UTC
    expected = {'yr': [2000, 2001], 'month': [12, 1], 'hr': [22, 23], 'weekday':[6, 5]}
    for x, y in zip(expected['yr'], outputs['yr']):
      y = int(y.numpy().item())
      self.assertEqual(x, y)
      
    for x, y in zip(expected['hr'], outputs['hr']):
      y = int(y.numpy().item())
      self.assertEqual(x, y)
      
    for x, y in zip(expected['weekday'], outputs['weekday']):
      y = int(y.numpy().item())
      self.assertEqual(x, y)
    
  def test_month_floordiv(self):
      import numpy as np
      
      days_since_1970 = tf.constant([0, 30, 360, 390], dtype=tf.int64)
      
      days_since_1970_2 = tf.constant([0 + 1*365, 30 + 2*365, 360+ 3*365, 390+ 4*365], dtype=tf.int64)
      
      expected_months = [0, 1, 0, 1]
      
      total_months = tf.math.floordiv(days_since_1970, tf.constant(30, dtype=tf.int64))
      month_of_year = tf.math.mod(total_months, 12)
      self.assertTrue(np.array_equal(month_of_year, expected_months))
      
      total_months = tf.math.floordiv(days_since_1970_2,
          tf.constant(30, dtype=tf.int64))
      month_of_year = tf.math.mod(total_months, 12)
      self.assertTrue(np.array_equal(month_of_year, expected_months))

if __name__ == '__main__':
  unittest.main()