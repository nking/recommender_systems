#!/bin/bash

####################################################################################
#  After have approved of the model, this script can be used to write twoer products
#     and copy files to directories in ranker project
#
####################################################################################

export TMP=`pwd`/../TMP11
#the sub_directory number in ../ranker/src/test/resources/model_repositories/saved_model_formats/bi-encoder/ to store model files.
#   this version number is also used to store the tower_products in the data directory
export RANKER_QUERY_SUDBIR_VERSION="2"

#activate the retriever project's venv.
conda activate py_311

cd ../retrieval

export PIPELINE_DATA_DIR="$TMP/bin/rs_pipeline"
export OUTPUT_BASE_DIR=$TMP
echo "writing user and movie embeddings and recommendations"
python3 -m unittest src.test.python.movie_lens_retrieval.write_tower_products.WriteTowerProducts

editing this for new data locations and including metadata files for better data lineage tracking


export trained_model_dir=../TMP11/bin/rs_pipeline/Trainer/model/19/

# output dirs in recommender_systems
export dest_saved_model_dir=src/test/resources/serving_model/
export docs_dir=docs/mlops/best_fitting_model_stats/twotowderdnn_softmax

rm -rf $dest_saved_model_dir/BEST/*
rm -rf $dest_saved_model_dir/serving_candidate_model/*
rm -rf $dest_saved_model_dir/serving_query_model/*
cp -rf $trained_model_dir/Format-Serving/* $dest_saved_model_dir/BEST/
cp -rf $trained_model_dir/serving_candidate_model/* $dest_saved_model_dir/serving_candidate_model/
cp -rf $trained_model_dir/serving_query_model/* $dest_saved_model_dir/serving_query_model/
rm -rf ../retrieval/src/main/resources/serving_models/user_movie_model/*
cp -rf $trained_model_dir/Format-Serving/* ../retrieval/src/main/resources/serving_models/user_movie_model/

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



