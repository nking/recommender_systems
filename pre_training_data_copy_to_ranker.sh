#!/bin/bash

#############################################################
#    script to write array_record and parquet formats of the`
#        split ml-1m files for the retrieval and ranker projects
#        which are assumed to be sibling directories to this one
#
#    after model training,
#        see post_training.sh and post_training2.sh 
#############################################################

export OUTPUT_BASE_DIR="../TMP_DATA"
python3 -m unittest src.test.python.movie_lens_tfx.write_ml1m_to_other_formats.WriteOtherFormats

#TODO:
# copy the array record formats to retrieval
# copy the array_record and parquet to ranker
