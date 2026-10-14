#!/bin/bash

####################################################################################
#  After have approved of the model, this script can be used to write tower products
#     and copy files to directories in this, retrieval, and ranker project
#
# USAGE:
#    ./post_training2.sh
#    or
#    bash < post_training2.sh
####################################################################################

#the base dir containing the tower_products directory
export TMP=`pwd`/../TMP10
export PIPELINE_DATA_DIR="$TMP/bin/rs_pipeline"
export OUTPUT_BASE_DIR=$TMP
#the sub_directory number in ../ranker/src/test/resources/model_repositories/saved_model_formats/bi-encoder/ to store model files.
#   this version number is also used to store the tower_products in the data directory
RANKER_QUERY_SUDBIR_VERSION="1"

#activate the retriever project's venv.
eval "$(conda shell.bash hook)"
conda activate py_311
if [ "$CONDA_DEFAULT_ENV" != "py_311" ]; then
    echo "Error: Failed to activate Conda environment."
    exit 1
fi

cd ../retrieval

echo "writing user and movie embeddings and recommendations"
python3 -m unittest src.test.python.movie_lens_retrieval.write_tower_products.WriteTowerProducts

read -r pusher_uri trained_model_dir <<< `python3 src/test/python/movie_lens_retrieval/find_pushed_trainer_uri.py`
trained_model_dir=`realpath $trained_model_dir`
echo "trained_model_dir=$trained_model_dir"
#export PUSHER_URI=$pusher_uri
#export TRAINER_URI=$trained_model_dir
if [ ! -d "$trained_model_dir" ]; then
    echo "Error: Directory does not exist: $trained_model_dir"
    exit 1
fi

cd ../recommender_systems

# output dirs in recommender_systems
dest_saved_model_dir=`realpath ./src/test/resources/serving_model/`
docs_dir=`realpath ./docs/mlops/best_fitting_model_stats/twotowderdnn_softmax`
tower_prod_dir=$OUTPUT_BASE_DIR/tower_products
#echo "dest_saved_model_dir=$dest_saved_model_dir"
#echo "docs_dir=$docs_dir"
#echo "tower_prod_dir=$tower_prod_dir"

## ============================================================================
##   dopy to recommender_systems    test resources
##=============================================================================
echo "copy saved_model files to recommender_systems to src/test/resources..."
rm -rf $dest_saved_model_dir/BEST/*
rm -rf $dest_saved_model_dir/serving_candidate_model/*
rm -rf $dest_saved_model_dir/serving_query_model/*
cp -rf $trained_model_dir/Format-Serving/* $dest_saved_model_dir/BEST/
cp -rf $trained_model_dir/serving_candidate_model/* $dest_saved_model_dir/serving_candidate_model/
cp -rf $trained_model_dir/serving_query_model/* $dest_saved_model_dir/serving_query_model/

echo "copy analysis files to recommender_systems docs/..."
cp $trained_model_dir/Format-Serving/assets.extra/hyperparameters.json $docs_dir/
cp $OUTPUT_BASE_DIR/pngs/*png $docs_dir/snapshots_tensorboard/
cp $OUTPUT_BASE_DIR/pngs/*json $docs_dir/train_val_curve_metrics/
rm -rf $docs_dir/post_training_analysis
cp -rf $OUTPUT_BASE_DIR/post_training_analysis $docs_dir/post_training_analysis

echo "copy saved_model files to  retrieval"
rm -rf ../retrieval/src/main/resources/serving_models/user_movie_model/*
cp -rf $trained_model_dir/Format-Serving/* ../retrieval/src/main/resources/serving_models/user_movie_model/

echo "copy embeddings and recommendations to retrieval"
retr_dir=../retrieval/src/test/resources/data
rm -rf $retr_dir/movie_emb_inp/*
rm -rf $retr_dir/user_emb_inp/*
cp -rf $tower_prod_dir/user_emb* $retr_dir/user_emb_inp/
cp -rf $tower_prod_dir/movie_emb* $retr_dir/movie_emb_inp/
cp -rf $tower_prod_dir/recommended_movies* $retr_dir/

## ============================================================================
##   dopy to ranker   project
##=============================================================================
echo "copy saved_models to ranker"
ranker_q_dir=../ranker/src/test/resources/model_repositories/saved_model_formats/bi-encoder/query/$RANKER_QUERY_SUDBIR_VERSION
ranker_c_dir=../ranker/src/test/resources/model_repositories/saved_model_formats/bi-encoder/candidate/$RANKER_QUERY_SUDBIR_VERSION
mkdir -p $ranker_q_dir
mkdir -p $ranker_c_dir
rm -rf $ranker_q_dir/* $ranker_c_dir/*
cp -rf $trained_model_dir/serving_query_model/* $ranker_q_dir/
cp -rf $trained_model_dir/serving_candidate_model/* $ranker_c_dir/
saved_model_cli show --dir $ranker_q_dir --all >& $ranker_q_dir/../../query_model_signatures.txt
saved_model_cli show --dir $ranker_c_dir --all >& $ranker_c_dir/../../candidate_model_signatures.txt

echo "copy tower_products to ranker"
ranker_data_dir=../ranker/src/test/resources/data/bi-encoder/$RANKER_QUERY_SUDBIR_VERSION
mkdir -p $ranker_data_dir
cp -rf $tower_prod_dir/* $ranker_data_dir/
rm $ranker_data_dir/*tfrecord*
