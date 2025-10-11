# recommender_systems
repository for refactoring of code at https://www.kaggle.com/code/nicholeasuniquename/recommender-systems to use MLOps

NOT COMPLETE YET

see library versions compatible with tfx 1.16.0
#see dependencies https://github.com/tensorflow/transform
those are installed to a conda virtual environment based on python 3.10

to create a virtual environment to install the TFX compatible
libraries, can use conda or virtualenv.
(1) for conda, 
  see: https://www.kaggle.com/code/nicholeasuniquename/a-virtual-environment-w-earlier-version-of-python
(2) for virtualenv
  python3 -m pip install --user virtualenv
  python3 -m virtualenv -p python3.10 /path/to/envs/python_310_tfx
  source /path/to/envs/python_310_tfx/bin/activate
  
the virutal environments are activated within a shell, and are not
currently selectable in the Kaggle notebook 
after ipykernel install and register.  The kernels are selectable
in Google Cloud jupyter notebooks (in Vertex AI workbenches)
and presumably in AWS SageMaker Studio notebooks, and Azure ML Studio.

once within a shell using activated virtual env having python 3.10:

pip -q install pyarrow==10.0.1
pip -q install apache-beam==2.59.0
pip -q install tensorflow==2.16.1
pip -q install tensorflow-transform==1.16.0
pip -q install tfx==1.16.0
pip -q install tensorflow-data-validation==1.16.0
pip -q install pytest

======= 
ingest:

  Twp custom components were made to choose from:
  1) a python function custom component:
    MovieLensExampleGen from ingest_movie_lens_component.py
  2) a fully custom component:
    IngestMovieLensComponent from ingest_movie_lens_custom_component.py

