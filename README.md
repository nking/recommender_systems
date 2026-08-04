# recommender_systems

This is a project holding a TFX MLOps pipeline to train a
Two-Tower DNN (bi-encoder) using a contrastive listwise 
loss with item sampling bias corrections and a term to suppress
item popularity bias.   In the hyperparameter tuning stage,
the objective to choose the best model was constructed from 
NDCG@20 on in batch positive and negatives and NDCG@20 on only the tail
of candidates - a composite of the 2 in order to learn niche users better.
The validation dataset was used to choose the best model.

The results are in docs/mlops/ subdirectory.

The kaggle notebook is at:
https://www.kaggle.com/code/nicholeasuniquename/recommender-systems to use MLOps

The main pipeline, 

    run_kaggle_pipelines.py 

runs the MLOps for the TwoTowerDNN with bias corrections (default) or without.
The in-batch negatives and item popularity bias corrections
follow the Yi et al. 2019 "Sampling-bias-corrected neural modeling
for large corpus item recommendations".
The model trains Query and Candidate models that produce
embeddings that live in the same embedding reference space, 
hence can be used to look-up one another when stored in an 
approximate nearest neighbor indexer like Scann or Faiss
(the Retrieval project builds the retriever, but even more
robust is the end-to-end inference in the ranker project
referenced below).

This repository is source code used in Kaggle notebooks:

https://www.kaggle.com/code/nicholeasuniquename/recommender-systems-with-tfx-pipelines/

https://www.kaggle.com/code/nicholeasuniquename/recommender-systems/

Using TFX requires a careful control of library versions for compatibility.
Older versions of this project used tfx 1.16.0 which required older versions
of python and the TF stack, but in the past year TFX has released a few
versions, the latest as of Aug 2026 is 1.21.0.
#see dependencies https://github.com/tensorflow/transform and compatibility
matrix as https://pypi.org/project/tfx/

To create a virtual environment to install the TFX compatible
libraries, can use conda or virtualenv.
(1) for conda, 
  see: https://www.kaggle.com/code/nicholeasuniquename/a-virtual-environment-w-earlier-version-of-python

  conda create -q --name tfx_py311 python=3.11 -y

  conda activate tfx_py311

  python 3.11 was chosen for compatibility with ml-metadata, installable
  as a whell on kaggle which is running ubuntu 22.04.
      ml-metadata==1.17.1
  might need numpy numpy==1.26.4? for python_version < '3.13'

(2) for virtualenv

  python3 -m pip install --user virtualenv

  python3 -m virtualenv -p python3.11 /path/to/envs/python_311_tfx

  source /path/to/envs/python_311_tfx/bin/activate
  
the virtual environments are activated within a shell, and are not
currently selectable in the Kaggle notebook 
after ipykernel install and register.  The kernels are selectable
in Google Cloud jupyter notebooks (in Vertex AI workbenches)
and presumably in AWS SageMaker Studio notebooks, and Azure ML Studio.

Once within a shell using activated virtual env having python 3.11:

if not using kaggle, make sure your platform glibxx libraries are
updated because pyfarmhash needs GLIBCXX_3.4.32

    sudo apt-get update

    sudo apt-get install --reinstall libstdc++6

can find the versions with:

    strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX | grep 32

If you do not see 3.4.32 within that list, then do the same "strings" check
on the host system (not virtual env).  
if not found there either, try to update the host system similarly for 
libstdc++6.

If you have 3.4.32 within the host system list, you can use this to copy over
the more complete host library:

   cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ~/miniconda3/envs/tfx_py313/lib/

Then, the activated virtual environment needs these packages:

the dependencies can be installed most easily with:

   pip install --editable .

A good resource for looking at version compatability with TFX 1.21.0
is https://github.com/tensorflow/tfx/blob/v1.21.0/test_constraints.txt

For other versions of TFX, need to use a different tag than v1.21.0

============= 

Miscellaneous project information:

ingest components:

  Two custom components were made to choose from:

  1) a python function custom component:

    MovieLensExampleGen from ingest_movie_lens_component.py

  2) a fully custom component:

    IngestMovieLensComponent from ingest_movie_lens_custom_component.py

Then a third python function component was made so that splits
could be performed before use in the pipeline.

    MovieLensSplitExampleGen from ingest_already_split_movie_lens_component.py

=======

setup.py is used because need to package the ingest component with
all of its sibling imports for the pipeline.

pyproject.toml is also present with same information.

in project base directory, within activated virtual environment:

  pip install --editable . 

local testing, no CI/CD, but the tests are invocable in IDE
or using a bash shell.

  for CI/CD, can add favorite software... GitHubActions come with
  this repository, GitLab, CircelCI, Jenkins, 
  Google Cloud's CloudBuid and CloudDeploy, 
  AWS CodePipelines, AzureDevOps

  pycharm:

    using right click menu, mark the source tree directory:
      src/main/python

    using right click menu, mark the test tree directory:
      src/test/python/movie_lens_tfx

    then pycharm tests will correctly resolve paths.

  bash or other shell environment:

    python and pytest can be used from the project's base
    directory
  
===========================================================

(1) This project trains the Two-Tower bi-encoder using TF/Keras3/TFX stack.

    https://github.com/nking/recommender_systems

    https://www.kaggle.com/code/nicholeasuniquename/recommender-systems-with-tfx-pipelines

    https://www.kaggle.com/code/nicholeasuniquename/recommender-systems    

(2) retriever tinkers with how to use the embedding models and
other algorithms that an be used for cold starts, etc.

    https://github.com/nking/retrieval
    
(3) ranker trains a JAX AI stack  cross-encoder needed for more accurate
personalization after the fast retrieval stage.
The ranker project trains a cross-encoder that uses
a Graph Attention Transformer V2 layer followed by a dens score layer.  
The project also creates a rust server for scalable end-to-end inference that makes
gRPC calls to the TFS deployed Query model trained in (1) 
and the TFS deployed trained ranker cross-encoder.
The project also stages scripts for how to use data-parallel training 
over multiple-hosts in a K8s cluster to perform HPO (and train and test) of the model.

    https://github.com/nking/ranker

    https://www.kaggle.com/code/nicholeasuniquename/ranker-cross-encoder-w-gatv2/

(4) re-ranker is a cross encoder to score candidate
inputs that can have originated from different recommendations.
The project fine-tunes a pre-trained T5 distilled LLM from hugging
face to build a pytorch  list-wise learning to rank, re-ranker.
It hasn't been added to the end-to-end inference in (3) because 
I haven't time to do it yet.

    https://github.com/nking/reranker

    https://www.kaggle.com/code/nicholeasuniquename/re-ranker-fine-tuning-of-pre-trained


