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

export PIPELINE_DATA_DIR="$(realpath -m ../kaggle_reco_sys/TMP70/bin/rs_pipeline)"
export OUTPUT_BASE_DIR="$(realpath -m ../kaggle_reco_sys/TMP70)"
export TAG="newLoss"

unset PIPELINE_DATA_DIR_BASELINE
unset OUTPUT_BASE_DIR_BASELINE
unset TAG_BASELINE
#if have a baseline model such as previous bes to compare PIPELINE_DATA_DIR to, set these vars
export PIPELINE_DATA_DIR_BASELINE="$(realpath -m ../kaggle_reco_sys/TMP62/bin/rs_pipeline)"
export OUTPUT_BASE_DIR_BASELINE="$(realpath -m ../kaggle_reco_sys/TMP62)"
export TAG_BASELINE="oldLoss"

#======= generate the train, val, test curves ====================
# writes to ${OUTPUT_BASE_DIR}/pngs
python3 -m unittest src.test.python.movie_lens_tfx.write_tensorboard_pngs_test.WriteTensorboardToPng

# ======== run post-training analysis
eval "$(conda shell.bash hook)"
conda activate py_311
if [ "$CONDA_DEFAULT_ENV" != "py_311" ]; then
    echo "Error: Failed to activate Conda environment."
    exit 1
fi

cd ../retrieval

# writes to ${OUTPUT_BASE_DIR}/post_training_analysis
## to compare to previous best model use:
if [ -n "$PIPELINE_DATA_DIR_BASELINE" ]; then
    echo "compare models $TAG and $TAG_BASELINE"
    python3 src/test/python/movie_lens_retrieval/post_training_compare_models.py \
    --analyze_and_compare \
    --PIPELINE_DATA_DIR_1=$PIPELINE_DATA_DIR \
    --OUTPUT_BASE_DIR_1=$OUTPUT_BASE_DIR \
    --TAG_1=$TAG \
    --PIPELINE_DATA_DIR_2=$PIPELINE_DATA_DIR_BASELINE \
    --OUTPUT_BASE_DIR_2=$OUTPUT_BASE_DIR_BASELINE \
    --TAG_2=$TAG_BASELINE \
    --OUTPUT_COMPARE_DIR="../${TAG_BASELINE}_${TAG}_Compare" \
    --quiet
else
    echo "analyze model at $PIPELINE_DATA_DIR"
    python3 -m unittest src.test.python.movie_lens_retrieval.post_training_analysis.TestAnalysis
fi

#ls -lctr $OUTPUT_BASE_DIR/post_training_analysis/
