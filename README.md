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
  
the virtual environments are activated within a shell, and are not
currently selectable in the Kaggle notebook 
after ipykernel install and register.  The kernels are selectable
in Google Cloud jupyter notebooks (in Vertex AI workbenches)
and presumably in AWS SageMaker Studio notebooks, and Azure ML Studio.

once within a shell using activated virtual env having python 3.10:

if not using kaggle, make sure your platform glibxx libraries are
updated because pyfarmhash needs GLIBCXX_3.4.32
    sudo apt-get update
    sudo apt-get install --reinstall libstdc++6
can find the versions with:
    strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX | grep 32
if you do not see 3.4.32 within that list, then do the same "strings" check
on the host system (not virtual env).  
if not found there either, try to update the host system similarly for 
libstdc++6.
if have 3.4.32 within the host system list, you can use this to copy over
the more complete host library:
   cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ~/miniconda3/envs/tfx_py310/lib/

Then, the activated virtural envirnoment needs these packages:

pip -q install pyarrow==10.0.1
pip -q install apache-beam==2.59.0
pip -q install tensorflow==2.16.1
pip -q install tensorflow-transform==1.16.0
pip -q install tfx==1.16.0
pip -q install tensorflow-data-validation==1.16.0
pip -q install pytest

============= 
Miscellaneous project information:

ingest components:

  Two custom components were made to choose from:
  1) a python function custom component:
    MovieLensExampleGen from ingest_movie_lens_component.py
  2) a fully custom component:
    IngestMovieLensComponent from ingest_movie_lens_custom_component.py

=======

local testing, no CI/CD yet (pipeline uses scripts...):
  pycharm:
    using right click menu, mark the source tree directory:
      src/main/python/movie_lens_tfx
    using right click menu, mark the test tree directory:
      src/test/python/movie_lens_tfx
    then pycharm tests will correctly resolve paths.
  bash or other shell environment:
    the directory that python is executed from becomes the CWD, 
    and so sibling scripts in the source directory cannot find
    one another because they are looking for scripts relative to
    the CWD.
    workarounds that are non-invasive are:
      - use pytest by setting properties in pytest.ini
      - use pytest.ini along with pyproject.toml which is in the
        source tree top directory src/main/python/movie_lens_tfx/
      looks like one solution might be to list all the
      scripts and methods to expose in the pyproject.toml
      
        
  
