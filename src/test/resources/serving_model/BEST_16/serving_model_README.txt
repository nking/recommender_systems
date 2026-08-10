serving model was created from run_kaggle_pipelines.py
which is in the test python source code.

use_bias_cor was fixed to True and the Tuner used keras_tuner.Hyberband
to find the best hyper-parameters.

The validation metrics:
   ndcg@20 = 0.049
   mrr@20 = 0.048
   recall@20 = 0.055
   hit rate = 0.047

The best fitting model is in saved_model and has these hyper-parameters:
        "learning_rate": 0.00010260217616970745,
        "weight_decay": 0.00016785171923416138,
        "regl2": 0.0,
        "drop_rate": 0.1175417396617746,
        "embed_out_dim": 32,
        "layer_sizes": "[16]",
        "feature_acronym": "ahosy",
        "incl_genres": true,
        "BATCH_SIZE": 2048,
        "NUM_EPOCHS": 40,
        "use_bias_corr": true,
        "bias_corr_alpha": 0.01,
        "temperature": 0.1,
        "n_users": 6040,
        "n_movies": 3883,
        "n_genres": 18,
        "run_eagerly": false,
        "device": "CPU",
        "num_train": 370838,
        "num_eval": 46354,
        "version": "1.0.0",
        "model_name": "user_movie",

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
