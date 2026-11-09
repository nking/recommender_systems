# use only once: src/test/python/movie_lens_tfx/RenumberIds.py

## handle splits and formats of ratings files ####
python3 src/test/python/movie_lens_tfx/write_train_val_test_splits_by_user.py

cp -f bin/full/ratings_*dat src/main/resources/ml-1m/

cp -f bin/full/ratings_*dat src/test/resources/ml-1m/
cp -f bin/full/ratings_*array_record src/test/resources/ml-1m/

cp -f bin/small/ratings_*dat src/test/resources/ml-1m/small/
cp -f bin/small/ratings_*array_record src/test/resources/ml-1m/small/

cp -f bin/tiny/ratings_*dat src/test/resources/ml-1m/tiny/
cp -f bin/tiny/ratings_*array_record src/test/resources/ml-1m/tiny/

python3 src/test/python/movie_lens_tfx/write_tiny3_tiered_liked_splits.py
cp -f bin/tiny3/ratings_*dat src/test/resources/ml-1m/tiny3/
cp -f bin/tiny3/ratings_*array_record src/test/resources/ml-1m/tiny3/

python3 src/test/python/movie_lens_tfx/write_tiny2_splits.py
cp -f bin/tiny2/ratings_*dat src/test/resources/ml-1m/tiny2/

## ===== handle movie_tiers #######
python -3 unittest src/test/python/movie_lens_tfx/WriteRetrievalInputs.test_write_movie_head_torso_tail_tiers

cp -f bin/movie_tiers/* src/test/resources/

## ====== copy to other projects.  they are sibling directories to this project's ####
export TMP_OUT=../retrieval/src/test/resources/data 
cp -f bin/full/ratings*array_record $TMP_OUT/
cp -f bin/small/ratings*array_record $TMP_OUT/small/

echo "copy data to ranker"
TMP_OUT=../ranker/src/test/resources/data 
mkdir -p $TMP_OUT/small $TMP_OUT/tiny $TMP_OUT/tiny3

cp -f bin/full/ratings*array_record $TMP_OUT/
cp -f bin/full/ratings*parquet $TMP_OUT/

cp -f bin/small/ratings*array_record $TMP_OUT/small/
cp -f bin/small/ratings*parquet $TMP_OUT/small/

cp -f bin/tiny/ratings*array_record $TMP_OUT/tiny/
cp -f bin/tiny/ratings*parquet $TMP_OUT/tiny/

cp -f bin/tiny3/ratings*array_record $TMP_OUT/tiny3/
cp -f bin/tiny3/ratings*parquet $TMP_OUT/tiny3/

cp -rf bin/movie_tiers/* $TMP_OUT/

echo "=== Copying Files from Ranker to Re-Ranker Project ==="
if [ ! -d "../reranker" ]; then
    echo "Directory '../reranker' does not exist.  Checkout the project from github if want to use it" >&2
    exit 0
fi

TMP_SRC=../ranker/src/test/resources/data 
TMP_OUT=../reranker/src/test/resources/data 
mkdir -p $TMP_OUT/small $TMP_OUT/tiny $TMP_OUT/tiny3
cp -f $TMP_SRC/*parquet $TMP_OUT/

cp -f $TMP_SRC/ratings*parquet $TMP_OUT/small/
cp -f $TMP_SRC/ratings*parquet $TMP_OUT/tiny/
cp -f $TMP_SRC/ratings*parquet $TMP_OUT/tiny3/

cp -rf $TMP_SRC/movie_tiers.json $TMP_OUT/

#the users and movies files are copied to rerankerhere, but
## could be placed in recommender_systems/pre_training_data_copy_to_ranker.sh
cp -rf $TMP_SRC/users.bin $TMP_OUT/
