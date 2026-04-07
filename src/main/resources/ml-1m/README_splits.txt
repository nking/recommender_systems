The ratings train, val and test splits were made by
the file in the test branch called write_train_val_splits.py

The splits are proportions percents 80:10:10 for
train:10:10 where the partitions are formed for
data ordered iby increasing timestamp.

The time split enables forecasting and easier shift
of window upon model iterations as data is acquired
over time.

