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

# Exit immediately if a command exits with a non-zero status, treat unset variables as an error,
# and ensure errors in pipelines are not masked.
set -euo pipefail

#the base dir containing the tower_products directory
export TMP="$(realpath -m ../TMP62)"
export PIPELINE_DATA_DIR="$TMP/bin/rs_pipeline"
export OUTPUT_BASE_DIR=$TMP
#the sub_directory number in ../ranker/src/test/resources/model_repositories/saved_model_formats/bi-encoder/ to store model files.
#   this version number is also used to store the tower_products in the data directory
RANKER_QUERY_SUDBIR_VERSION="1"

#activate the retriever project's venv.
set +u
eval "$(conda shell.bash hook)"
conda activate py_311
set -u
if [ "${CONDA_DEFAULT_ENV:-}" != "py_311" ]; then
    echo "Error: Failed to activate Conda environment."
    exit 1
fi

echo "=== [2/5] Generating Tower Products in Retrieval Project ==="
if [ ! -d "../retrieval" ]; then
    echo "Error: Directory '../retrieval' does not exist relative to $(pwd)." >&2
    exit 1
fi

cd ../retrieval

echo "writing user and movie embeddings and recommendations"
python3 -m unittest src.test.python.movie_lens_retrieval.write_tower_products.WriteTowerProducts

if ! URI_OUTPUT="$(python3 src/test/python/movie_lens_retrieval/find_pushed_trainer_uri.py)"; then
    echo "Error: Failed to execute find_pushed_trainer_uri.py" >&2
    exit 1
fi

read -r pusher_uri trained_model_dir <<< "$URI_OUTPUT"

trained_model_dir="$(realpath "$trained_model_dir")"

if [ -z "${trained_model_dir:-}" ]; then
    echo "Error: Returned trained_model_dir is empty." >&2
    exit 1
fi

#echo "trained_model_dir=$trained_model_dir"
#export PUSHER_URI=$pusher_uri
#export TRAINER_URI=$trained_model_dir
if [ ! -d "$trained_model_dir" ]; then
    echo "Error: Directory does not exist: $trained_model_dir"
    exit 1
fi

if [ ! -d "../recommender_systems" ]; then
    echo "Error: Directory '../recommender_systems' does not exist relative to $(pwd)." >&2
    exit 1
fi

cd ../recommender_systems

# output dirs in recommender_systems
dest_saved_model_dir="$(realpath -m ./src/test/resources/serving_model/)"
docs_dir="$(realpath -m ./docs/mlops/best_fitting_model_stats/twotowerdnn_softmax)"
tower_prod_dir="$OUTPUT_BASE_DIR/tower_products"

if [ ! -d "$tower_prod_dir" ]; then
    echo "Error: tower_prod_dir does not exist at $tower_prod_dir" >&2
    exit 1
fi
#echo "dest_saved_model_dir=$dest_saved_model_dir"
#echo "docs_dir=$docs_dir"
#echo "tower_prod_dir=$tower_prod_dir"

## ============================================================================
##   copy to recommender_systems    test resources
##=============================================================================
echo "=== Copying Files to recommender_systems ==="
mkdir -p "$dest_saved_model_dir/BEST" \
         "$dest_saved_model_dir/serving_candidate_model" \
         "$dest_saved_model_dir/serving_query_model" \
         "$docs_dir/snapshots_tensorboard" \
         "$docs_dir/train_val_curve_metrics"

#:? checks that the variable is not empty and not null, and exits with stderr if not
rm -rf "${dest_saved_model_dir:?}/BEST/"*
rm -rf "${dest_saved_model_dir:?}/serving_candidate_model/"*
rm -rf "${dest_saved_model_dir:?}/serving_query_model/"*

cp -rf "$trained_model_dir/Format-Serving/"* "$dest_saved_model_dir/BEST/"
cp -rf "$trained_model_dir/serving_candidate_model/"* "$dest_saved_model_dir/serving_candidate_model/"
cp -rf "$trained_model_dir/serving_query_model/"* "$dest_saved_model_dir/serving_query_model/"

if [ -f "$trained_model_dir/Format-Serving/assets.extra/hyperparameters.json" ]; then
    cp "$trained_model_dir/Format-Serving/assets.extra/hyperparameters.json" "$docs_dir/"
fi
[ -d "$OUTPUT_BASE_DIR/pngs" ] && cp "$OUTPUT_BASE_DIR/pngs/"*png "$docs_dir/snapshots_tensorboard/" 2>/dev/null || true
[ -d "$OUTPUT_BASE_DIR/pngs" ] && cp "$OUTPUT_BASE_DIR/pngs/"*json "$docs_dir/train_val_curve_metrics/" 2>/dev/null || true

rm -rf "$docs_dir/post_training_analysis"
if [ -d "$OUTPUT_BASE_DIR/post_training_analysis" ]; then
    cp -rf "$OUTPUT_BASE_DIR/post_training_analysis" "$docs_dir/post_training_analysis"
fi

echo "=== Copying Files to Retrieval Project ==="
mkdir -p ../retrieval/src/main/resources/serving_models/user_movie_model \
         ../retrieval/src/test/resources/data/movie_emb_inp \
         ../retrieval/src/test/resources/data/user_emb_inp

rm -rf ../retrieval/src/main/resources/serving_models/user_movie_model/*
cp -rf "$trained_model_dir/Format-Serving/"* ../retrieval/src/main/resources/serving_models/user_movie_model/

retr_dir="../retrieval/src/test/resources/data"
rm -rf "${retr_dir:?}/movie_emb_inp/"*
rm -rf "${retr_dir:?}/user_emb_inp/"*
cp -rf "$tower_prod_dir/user_emb"* "$retr_dir/user_emb_inp/"
cp -rf "$tower_prod_dir/movie_emb"* "$retr_dir/movie_emb_inp/"
cp -rf "$tower_prod_dir/recommended_movies"* "$retr_dir/"

echo "=== Copying Files to Ranker Project ==="
if [ ! -d "../ranker" ]; then
    echo "Error: Directory '../ranker' does not exist." >&2
    exit 1
fi

ranker_q_dir="../ranker/src/test/resources/model_repositories/saved_model_formats/bi-encoder/query/$RANKER_QUERY_SUDBIR_VERSION"
ranker_c_dir="../ranker/src/test/resources/model_repositories/saved_model_formats/bi-encoder/candidate/$RANKER_QUERY_SUDBIR_VERSION"
mkdir -p "$ranker_q_dir" "$ranker_c_dir"

rm -rf "${ranker_q_dir:?}"/* "${ranker_c_dir:?}"/*
cp -rf "$trained_model_dir/serving_query_model/"* "$ranker_q_dir/"
cp -rf "$trained_model_dir/serving_candidate_model/"* "$ranker_c_dir/"

if command -v saved_model_cli &> /dev/null; then
    saved_model_cli show --dir "$ranker_q_dir" --all > "$ranker_q_dir/../../query_model_signatures.txt" 2>&1 || true
    saved_model_cli show --dir "$ranker_c_dir" --all > "$ranker_c_dir/../../candidate_model_signatures.txt" 2>&1 || true
else
    echo "Warning: 'saved_model_cli' not found in PATH. Skipping signature inspection files." >&2
fi

ranker_data_dir="../ranker/src/test/resources/data/bi-encoder/$RANKER_QUERY_SUDBIR_VERSION"
mkdir -p "$ranker_data_dir"
rm -rf "${ranker_data_dir:?}"/*
cp -rf "$tower_prod_dir/"* "$ranker_data_dir/"
rm -f "$ranker_data_dir/"*tfrecord*

echo "=== Post-training artifact deployment completed successfully! ==="
