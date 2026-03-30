serving model was created from run_kaggle_pipelines.py
which is in the test python source code.

use_bias_cor was fixed to True and the Tuner used keras_tuner.Hyberband
to find the best hyper-parameters.

The best fitting model is in saved_model and has these hyper-parameters:
<add here>

The validation metrics:
   val_hit_rate = <add here>
   normalized for batch_size=1024 gives NHR = <add here>
   which is healthy learning above an NHR of 1 for random.

=========================================================
The metadata model (not saved in this project, but you
can create it with run_kaggle_metadata_pipelines.py)
is a regression model and calculates RMSE as one of its
metrics, so can be compared with the Netflix competition
which won with a model with RMSE of 0.8567, improving 
upon the Netflix standard by 10%.
The metadata model with batch_size 32 has RMSE 0.25
on this project's validation dataset.

