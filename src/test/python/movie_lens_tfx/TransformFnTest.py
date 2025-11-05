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
      #remove chicago offset
      est = epoch_timestamp + chicago_tz_offset
      #seems the leap-year 2000 estimate is 1 day less than expected
      #print(f'expected={timestamp}, calc={est}')
    
    expected = {'yr': [2000, 2001], 'month': [12, 1], 'hr': [16, 17]}
    for x, y in zip(expected['yr'], outputs['yr']):
      y = int(y.numpy().item())
      self.assertEqual(x, y)
      
    for x, y in zip(expected['hr'], outputs['hr']):
      y = int(y.numpy().item())
      self.assertEqual(x, y)
    

if __name__ == '__main__':
  unittest.main()