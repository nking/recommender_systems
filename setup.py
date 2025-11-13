from setuptools import setup, find_packages

setup(
  name='movie_lens_tfx',
  version='0.1.0',
  packages=find_packages(where="src/main/python",
    include=['movie_lens_tfx', 'movie_lens_tfx.misc',
            'movie_lens_tfx.utils', 'movie_lens_tfx.ingest_component',
            'movie_lens_tfx.ingest_pyfunc_component', 'movie_lens_tfx']),
  package_dir={'': 'src/main/python'},
  install_requires = [
    "pyarrow==10.0.1",
    "apache-beam==2.59.0",
    "tensorflow==2.16.1",
    "tensorflow-transform==1.16.0",
    "tfx==1.16.0",
    "tensorflow-data-validation==1.16.1",
    "tensorflow-metadata==1.16.1",
    "ml-metadata==1.16.0",
    "python-snappy==0.7.3",
  ],
  extra_requires = ["test": ["pytest", "nannyml>=0.13.1"]]
  classifiers=[ 'Natural Language :: English',
               'Programming Language :: Python :: 3.10 :: Only',
               'Development Status :: 1 - Development/Unstable'
  ],
  url='https://www.kaggle.com/code/nicholeasuniquename/recommender-systems-with-tfx-pipelines',
  license='MIT',
  author='Nichole King',
  author_email='',
  description='TFX pipelines for Kaggle recommender systems project'
)
