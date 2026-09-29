#!/bin/bash

export TMP=../TMP11

conda activate tfx_py313

export SAVED_MODEL_DIR="$TMP/bin/rs_pipeline/Pusher/pushed_model/21"
export OUT_DIR="$TMP/retrieval_files"
export EMBEDDINGS_DIR=$OUT_DIR

echo "writing user and movie embeddings to $OUT_DIR"
python src/test/python/movie_lens_tfx/WriteRetrievalInputs.py \
   WriteRetrievalInputs.test_write_movie_embeddings \
   --saved_model_path=$SAVED_MODEL_DIR \
   --output_dir_path=$OUT_DIR

python src/test/python/movie_lens_tfx/WriteRetrievalInputs.py \
   WriteRetrievalInputs.test_write_user_embeddings \
   --saved_model_path=$SAVED_MODEL_DIR \
   --output_dir_path=$OUT_DIR

conda deactivate
conda activate py_311

cd ../retrieval

echo "writing user recommendations to $OUT_DIR"
#reads the environment variables set above
python3 -m unittest src.test.python.movie_lens_retrieval.write_user_recommendations_and_negatives.TestRetrieval.test_write_recommendations_and_timestamps


