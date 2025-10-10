# recommender_systems
repository for refactoring of code at https://www.kaggle.com/code/nicholeasuniquename/recommender-systems to use MLOps

NOT COMPLETE YET

see library versions compatible with tfx 1.16.0
https://pypi.org/project/tfx/1.16.0/
those are installed to a conda virtual environment based on python 3.10

======= 
ingest:

  Twp custom components were made to choose from:
  1) a python function custom component:
    MovieLensExampleGen from ingest_movie_lens_component.py
  2) a fully custom component:
    IngestMovieLensComponent from ingest_movie_lens_custom_component.py


   IngestMovieLensComponent in ingest_movie
