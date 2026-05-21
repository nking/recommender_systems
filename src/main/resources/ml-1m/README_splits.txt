The ratings train, val and test splits were made by
the file in the test branch called write_train_val_splits.py
   src/test/resources/movie_lens_tfx/write_train_val_splits.py
   they write to the bin directory, so to write to
   src/main/resources instead, edit the file to use 
   COPY_TO_SRC_TREE=True temporarily

The splits are proportions percents 80:10:10 for
train:10:10 where the partitions are formed for
data ordered by increasing timestamp.

The time split enables forecasting and easier shift
of window upon model iterations as data is acquired
over time.

