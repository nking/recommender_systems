serving model was created from run_kaggle_pipelines.py
which is in the test python source code.

use_bias_cor was fixed to True and the Tuner used keras_tuner.Hyberband
to find the best hyper-parameters.

The validation metrics:
   val_hit_rate = 0.0.0017
   normalized for batch_size=1024 gives NHR = 1.71
   which is better than an NHR of 1 for random.

The best fitting model is in saved_model and has these hyper-parameters:
earning_rate: 0.0019626556417708576
weight_decay: 0.0010527248318872142
regl2: 3.4782253230110776e-05
drop_rate: 0.3675077768870417
embed_out_dim: 32
layer_sizes: [32]
feature_acronym: ahos
incl_genres: True
BATCH_SIZE: 1024
NUM_EPOCHS: 20
use_bias_corr: True
bias_corr_alpha: 0.05
temperature: 0.15000000000000002
n_users: 6040
n_movies: 3952
n_age_groups: 7
n_genres: 18
run_eagerly: False
device: CPU
MAX_TUNE_TRIALS: 10
EXECUTIONS_PER_TRIAL: 1
input_dataset_element_spec_ser: ...
num_train: 370838
num_eval: 46354
version: 1.0.0
model_name: user_movie
Score: 0.001699846819974482

=========================================================
The metadata model (not saved in this project, but you
can create it with run_kaggle_metadata_pipelines.py)
is a regression model and calculates RMSE as one of its
metrics, so can be compared with the Netflix competition
which won with a model with RMSE of 0.8567, improving 
upon the Netflix standard by 10%.
The metadata model with batch_size 32 has RMSE 0.25
on this project's validation dataset.

