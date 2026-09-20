#!/bin/bash

#USAGE:
#    ./post_training.sh
#    or
#    bash < post_training.sh

#this is a convencience script that assumes that the github projects 
#  recommender_systems
#  retrieval
#  ranker
#  are sibling directories.
#  this is a placeholder script for CI/CD tasks after TwoTowerDNN training.
#
# (1) to write the training, val curves for visualization use
#    recommender_systems/src/test/python/movie_lens_tfx/write_tensorboard_pngs_test.py
#    it writes to recommender_systems/bin/pngs
#    which are copied to model_analysis directory below
# (2) to analyze a recently trained TwoTowerDNN model, use retrieval project's script
#    retrieval/src/test/python/movie_lens_retrieval/post_training_eda.py  
#    by changing the saved model directory to the latest
#    saved model directory and run the script.
#    the output directory will be copied where needed below
#    it will write to retrieval/bin/post_training_analysis
#    which are copied to model_analysis directory below
# (3) if the results of (1) and (2) meet requirements,
#    proceed to the steps below to copy files to the other projects.
#    (mainly to ranker which needs the embeddings for training)

#copy the latest approved trained model to subdirectories in
#  recommender_systems/src/test/resources/serving_model/

# **** EDIT THIS PATH AS NEEDED ******
export trained_model_dir=../TMP3/bin/rs_pipeline/Trainer/model/19/

# output dirs in recommender_systems
export dest_saved_model_dir=src/test/resources/serving_model/
export docs_dir=docs/mlops/best_fitting_model_stats/twotowderdnn_softmax

#copy the models to retrieval project
rm -rf $dest_saved_model_dir/BEST/*
rm -rf $dest_saved_model_dir/serving_candidate_model/*
rm -rf $dest_saved_model_dir/serving_query_model/*
cp -rf $trained_model_dir/Format-Serving/* $dest_saved_model_dir/BEST/
cp -rf $trained_model_dir/serving_candidate_model/* $dest_saved_model_dir/serving_candidate_model/
cp -rf $trained_model_dir/serving_query_model/* $dest_saved_model_dir/serving_query_model/

#copy the query model to the ranker project
export ranker_q_dir=../ranker/src/test/resources/model_repositories/saved_model_formats/bi-encoder/query/1
rm -rf $ranker_q_dir/*
cp -rf $trained_model_dir/serving_candidate_model/* $ranker_q_dir/
saved_model_cli show --dir $ranker_q_dir --all >& $ranker_q_dir/../../query_model_signatures.txt

#copy files to model_analysis
cp $trained_model_dir/Format-Serving/assets.extra/hyperparameters.json $docs_dir/
cp bin/pngs/*png $docs_dir/snapshots_tensorboard/
cp bin/pngs/*json $docs_dir/train_val_curve_metrics/

#copy to model_analysis retrieval_analysis
cp -rf ../retrieval/bin/post_training_analysis/* $docs_dir/retrieval_analysis/

# these write embeddings files (used by ranker)
python src/test/python/movie_lens_tfx/WriteRetrievalInputs.py WriteRetrievalInputs.test_write_movie_embeddings
python src/test/python/movie_lens_tfx/WriteRetrievalInputs.py WriteRetrievalInputs.test_write_user_embeddings

#copy embeddings tfrecods and json files to retrieval
cp -rf bin/movie_emb_inp/*gz ../retrieval/src/test/resources/data/movie_emb_inp/
cp -rf bin/user_emb_inp/*gz ../retrieval/src/test/resources/data/user_emb_inp/
cp -rf bin/movie_emb_inp/*.json ../retrieval/src/test/resources/data/movie_emb_inp/
cp -rf bin/user_emb_inp/*.json ../retrieval/src/test/resources/data/user_emb_inp/

#copy embeddings files to ranker except the gz files:
find bin/movie_emb_inp/ -maxdepth 1 -type f ! -name "*.gz" -exec cp {} ../ranker/src/test/resources/data/ \;
find bin/user_emb_inp/ -maxdepth 1 -type f ! -name "*.gz" -exec cp {} ../ranker/src/test/resources/data/ \;
if [ -d "../ranker/tmp_mounts/fake-gcs-server/data" ]; then
    find bin/movie_emb_inp/ -maxdepth 1 -type f ! -name "*.gz" -exec cp {} ../ranker/tmp_mounts/fake-gcs-server/data/ \;
    find bin/user_emb_inp/ -maxdepth 1 -type f ! -name "*.gz" -exec cp {} ../ranker/tmp_mounts/fake-gcs-server/data/ \;
fi

