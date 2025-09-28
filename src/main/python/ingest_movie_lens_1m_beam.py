import apache_beam as beam
import time
import random

def merge_and_split(pipeline : apache_beam.pipeline.Pipeline, \
  ratings_uri : str, movies_uri : str, users_uri : str, \
  ratings_key_dict : dict[str, int], movies_key_dict : dict[str, int], users_key_dict : dict[str, int],
  partitions : list[int]):
  '''
  :param pipeline:
  :param ratings_uri:
  :param movies_uri:
  :param users_uri:
  :param ratings_key_dict: for ratings file, a dictionary with key:values
    being header_column_name:column number
  :param movies_key_dict: for movies file, a dictionary with key:values being header_column_name:column number
  :param users_key_dict: for users file, a dictionary with key:values being header_column_name:column number
  :param partition: list of partitions in percent
  :return: a tuple of PCollection of ratings with joined information from users and movies where each tuple is for a
     partition specified in partition list
  '''

  skip = 1

  # user_id,movie_id,rating
  ratings_pc = pipeline \
    | 'ReadRatings' >> beam.io.ReadFromText(ratings_uri, skip_header_lines=skip) \
    | 'ParseRatings' >> beam.Map(lambda line: line.split(','))

  # movie_id,genre
  movies_pc = pipeline \
    | 'ReadMovies' >> beam.io.ReadFromText(movies_uri, skip_header_lines=skip) \
    | 'ParseMovies' >> beam.Map(lambda line: line.split(','))

  # UserID::Gender::Age::Occupation::Zip-code
  users_pc = pipeline \
    | 'ReadUsers' >> beam.io.ReadFromText(users_uri, skip_header_lines=skip) \
    | 'ParseUsers' >> beam.Map(lambda line: line.split(','))

  def merge_by_key(l_pc, r_pc, l_key_col, r_key_col, filter_cols):
    # need unique names for each beam process, so adding a timestamp
    ts = time.time_ns()
    l_keyed = l_pc | f'kv_l_{ts}' >> beam.Map(lambda x: (x[l_key_col], x))
    r_keyed = r_pc | f'kv_r_{ts}' >> beam.Map(lambda x: (x[r_key_col], x))

    # l_keyed | beam.Map(print)
    # r_keyed | 'beam.Map(print)

    grouped_data = ({'left': l_keyed, 'right': r_keyed} \
      | f'group_by_key_{ts}' >> beam.CoGroupByKey())
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
            if i != r_key_col and i not in filter_cols:
              row.append(right)
          yield row

    joined_data = grouped_data \
      | f'left_join_values_{ts}' >> beam.ParDo(left_join_fn())
    return joined_data

  # user_id,movie_id,rating,timestamp,Gender::Age::Occupation::Zip-code
  ratings_1 = merge_by_key(ratings_pc, users_pc, \
    ratings_key_dict['user_id'], users_key_dict['user_id'], \
    filter_cols=[users_key_dict['zipcode']])
  # user_id,movie_id,rating,Gender::Age::Occupation::Zip-code,genres
  ratings = merge_by_key(ratings_1, movies_pc, \
    ratings_key_dict['movie_id'], movies_key_dict['movie_id'],\
    filter_cols=[])

  #print(f'RATINGS type{type(ratings)}')
  #ratings | f'ratings_{time.time_ns()}' >> beam.Map(print)
  #['user_id', 'movie_id', 'rating', 'gender', 'age', 'occupation', 'genres']

  def split_fn(row, num_partitions, ppartitions):
    # Using a deterministic hash function ensures the split is consistent
    total = sum(ppartitions)
    buckets = [p / total for p in ppartitions]
    rand_num = random.random()
    s = 0
    for i, p in enumerate(buckets):
      s += p
      if rand_num < s:
        return i
    return len(ppartitions) - 1

  # The `split_fn` returns `True` for the first 80% and `False` for the last 20%.
  ratings_parts = ratings \
    | f'split_{time.time_ns()}' >> beam.Partition(\
    split_fn, len(partitions), partitions)

  #for i, part in enumerate(ratings_parts):
  #  part | f'PARTITIONED_{i}_{time.time_ns()}' >> beam.io.WriteToText(\
  #    file_path_prefix=f'a_{i}_', file_name_suffix='.txt')

  return ratings_parts
