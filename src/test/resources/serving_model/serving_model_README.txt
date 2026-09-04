serving model was created from run_kaggle_pipelines.py
which is in the test python source code.

use_bias_cor was fixed to True and the Tuner used keras_tuner.Hyberband
to find the best hyper-parameters.

The validation metrics:
   composite ndcg@20 = 0.0234
   head ndcg@20 = 0.0238
   torso ndcg@20 = 0.0236
   tail ndcg@20 = 0.0222
   ndcg@20 = 0.0237
   mrr@20 = 0.0196
   recall@20 = 0.0395
   hit rate = 0.026

The best fitting model is in saved_model and has these hyper-parameters
and parameters (see assets.extra in serving_model_dir):
       "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "drop_rate": 0.65,
        "log_q_correction_factor": 1.0,
        "regl2": 0.0,
        "layer_sizes": "[64, 32]",
        "feature_acronym": "ahosy",
        "incl_genres": true,
        "BATCH_SIZE": 2048,
        "NUM_EPOCHS": 20,
        "use_bias_corr": true,
        "bias_corr_alpha": 0.01,
        "temperature": 0.24000000000000002,
        "movie_tiers_uri": "/kaggle/working/recommender_systems/src/test/resources/movie_tiers.json",
        "n_users": 6040,
        "n_movies": 3883,
        "n_genres": 18,
        "run_eagerly": false,
        "num_train": 370838,
        "num_eval": 46354,
        "version": "1.0.0",
        "model_name": "user_movie",
        "git_hash": "cd1cd9ceabebd9b00e739fb8d12257f8363d5424",



=========================================================
The metadata model (not saved in this project, but you
can create it with run_kaggle_metadata_pipelines.py)
is a regression model and calculates RMSE as one of its
metrics, so can be compared with the Netflix competition
which won with a model with RMSE of 0.8567, improving 
upon the Netflix standard by 10%.
The metadata model with batch_size 32 has RMSE 0.25
on this project's test dataset (which a train, val, test 
split of the train dataset)
