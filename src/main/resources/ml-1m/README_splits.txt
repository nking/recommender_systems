The ratings train, val and test splits were made by
the file in the test branch called write_train_val_splits_by_user.py
   src/test/resources/movie_lens_tfx/write_train_val_splits.py
   they write to the bin directory, so copy those to src/main/resources/ml-1m/

The splits are proportions percents 80:10:10 for
train:10:10 where the partitions are formed for each user's
data ordered by increasing timestamp.

