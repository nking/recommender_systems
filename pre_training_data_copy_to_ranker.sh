#!/bin/bash

#############################################################
#    script to write array_record and parquet formats of the`
#        split ml-1m files for the retrieval and ranker projects
#        which are assumed to be sibling directories to this one
#
#    Follow this script, after model training,
#        with post_training.sh and post_training2.sh
#
# Usage:
#    ./pre_training_data_copy_to_ranker.sh
#  or
#      bash < pre_training_data_copy_to_ranker.sh
#############################################################

export OUTPUT_BASE_DIR="$(realpath -m ../TMPDATA)"
#if directory doesn't exist or is empty:
if [ ! -d "$OUTPUT_BASE_DIR" ] || [ -z "$(ls -A "$OUTPUT_BASE_DIR" 2>/dev/null)" ]; then
    python3 -m unittest src.test.python.movie_lens_tfx.write_ml1m_to_other_formats.WriteOtherFormats
fi

# copy the array_record and parquet to ranker
SRC_DIR="$OUTPUT_BASE_DIR/other_formats/ml-1m"
DEST_DIR="$(realpath -m ../ranker/src/test/resources/data/)"
echo "  copy from $SRC_DIR"
echo "      to $DEST_DIR"
mkdir -p $DEST_DIR
cp -rf $SRC_DIR/* $DEST_DIR/
echo "  copy to ranker/tmp_mounts"
python3 ../ranker/scripts/init-tmp_mounts.py

echo "copy array_record files to retrieval project"
DEST_DIR="$(realpath -m ../retrieval/src/test/resources/data/)"
echo "      to $DEST_DIR"
mkdir -p $DEST_DIR
cp -rf $SRC_DIR/*.array_record $DEST_DIR/
cp -f src/test/resources/ml-1m//users.bin $DEST_DIR/
