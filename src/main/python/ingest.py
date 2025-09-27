import apache_beam as beam

import os
import tensorflow as tf
from tfx.components.example_gen import base_example_gen_executor
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx_bsl.public import tfxio
import time

class ReadMergeAndSplitExecutor(base_example_gen_executor.BaseExampleGenExecutor):
    """Custom executor for ExampleGen that handles multiple files."""
  def Do (
    self,
    input_dict: tf.TfList[standard_artifacts.ExternalSource],
    output_dict: tf.TfList[standard_artifacts.Examples],
    exec_properties: tf.TfDict[str, tf.TfAny]
  ) -> None:
    # Get the input file paths from the input_dict.
    # The keys will depend on your component definition.
    ratings_uri = input_dict['ratings.dat'].uri
    movies_uri = input_dict['movies.dat'].uri
    users_uri = input_dict['users.dat'].u

    pipeline = self._MakeBeamPipeline()

    # user_id,movie_id,rating
    ratings_pc = pipeline | 'ReadRatings' >> beam.io.ReadFromText(ratings_uri) \
      | 'ParseRatings' >> beam.Map(lambda line: line.split(','))

    # movie_id,genre
    movies_pc = pipeline | 'ReadMovies' >> beam.io.ReadFromText(movies_uri) \
      | 'ParseMovies' >> beam.Map(lambda line: line.split(','))

    #UserID::Gender::Age::Occupation::Zip-code
    users_pc = pipeline | 'ReadUsers' >> beam.io.ReadFromText(users_uri) \
      | 'ParseUsers' >> beam.Map(lambda line: line.split(','))

    def merge_by_key(l_pc, r_pc, l_key_col, r_key_col):
      #need unique names for each beam process, so adding a timestamp
      ts = time.time_ns()
      l_keyed = l_pc | f'kv_l_{ts}' >> beam.Map(lambda x: (x[l_key_col], x))
      r_keyed = r_pc | f'kv_r_{ts}' >> beam.Map(lambda x: (x[r_key_col], x))

      # l_keyed | beam.Map(print)
      # r_keyed | 'beam.Map(print)

      grouped_data = (\
          {'left': l_keyed, 'right': r_keyed} | \
          'CoGroupByKey' >> beam.CoGroupByKey())
      # there are multiple lefts on one line now, and one in right's list

      class left_join_fn(beam.DoFn):
        def process(self, kv):
          key, grouped_elements = kv
          # grouped_elements is a dictionary with keys 'left' and 'right'
          # both are lists of lists.
          assert (len(grouped_elements['right']) == 1)
          for left in grouped_elements['left']:
            # join grouped_elements['left'] and grouped_elements['right'][0]
            # merge, reorder etc as wanted
            row = left.copy()
            for i, right in enumerate(grouped_elements['right'][0]):
              if i != r_key_col:
                row.append(right)
            yield row

      joined_data = grouped_data | f'left_join_values_{ts}' >> beam.ParDo(left_join_fn())
      #joined_data |  >> beam.Map(print)
      return joined_data

    # user_id,movie_id,rating,timestamp,Gender::Age::Occupation::Zip-code
    ratings_1 = merge_by_key(ratings_pc, users_pc, 0, 1)
    # user_id,movie_id,rating,Gender::Age::Occupation::Zip-code,genres
    ratings = merge_by_key(ratings, movies_pc, 1, 0)

    #not yet finished

    #TODO: order by timestamp and then split into train and test

    #write to output files if wanted
    # The output path is determined by the output_dict.
    output_path = os.path.join(output_dict['examples'].uri, self._MakeSplitName())
    _ = (
    merged_data
    | 'WriteExamples' >> tfxio.WriteTFXIOToTFRecords(output_path)
    )

    # Run the Beam pipeline?
    pipeline.run()
