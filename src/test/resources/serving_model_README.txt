serving model was created from run_kaggle_pipelines.py
which is in the test python source code.

use_bias_cor was fixed to True and the Tuner used RandomSearch
among the hyper-parameters.

The best fitting model is in saved_model and has these hyper-parameters:

"learning_rate": 0.004537750100817641, 
"regl2": 3.3037707935951735e-06, 
"drop_rate ": 0.1110236162515454, 
"embed_out_dim": 32, 
"layer_sizes": "[32]", 
"feature_acronym": "a", 
"incl_genres": true
"BATCH_SIZE": 1024, 
"NUM_EPOCHS": 20, 
"use_bias_corr": true, 
"bias_corr_alpha": 0.05, 
"temperature": 0.15000000000000002, 
"n_users": 6040, 
"n_movies": 3952, 
"n_age_groups": 7, 
"n_genres": 18, 
"run_eagerly": false, 
"device": "CPU", 
"MAX_TUNE_TRIALS": 10, 
"EXECUTIONS_PER_TRIAL": 1,

The validation metrics:
   val_hit_rate = 0.0023
   normalized for batch_size=1024 gives NHR = 2.36
   which is healthy learning above an NHR of 1 for random.
NOTE: can improve upon the NHR by a factor >=3 
by using Hyperband in the tune_fn.
The model saved in test/resources found hyperparameters
using the fast track keras tuner RandomSearch, but
Hyperband is more thorough.

fit history.history={'hit_rate': [0.004506396595388651, 0.008250406943261623, 0.010327128693461418, 0.01246777
8287827969], 'loss': [5.65212345123291, 5.374941825866699, 5.348050117492676, 5.2616496086120605], 'val_hit_ra
te': [0.002761336974799633, 0.002535830484703183, 0.00240512122400105, 0.0025421911850571632], 'val_loss': [9.
630130767822266, 9.74338150024414, 9.744372367858887, 9.888795852661133]}

=========================================================
The metadata model (not saved in this project, but you
can create it with run_kaggle_metadata_pipelines.py)
is a regression model and calculates RMSE as one of its
metrics, so can be compared with the Netflix competition
which won with a model with RMSE of 0.8567, improving 
upon the Netflix standard by 10%.
The metadata model with batch_size 32 has RMSE 0.25
on this project's validation dataset.

