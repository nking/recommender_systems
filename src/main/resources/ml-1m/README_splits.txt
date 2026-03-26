The ratings train, val and test splits were made by
the file train_val_test_splitter.py in the test branch.

The first split is by time such that (train and val) are < test in timestamps.

The second split is by user_id such that roughly 88.89% of the
(train and val) users are in train and the remaining are in
the val partition where train and val have no users in common.

The time split enables forecasting.
The user split enables a check that the model generalizes well
to unseen data.

