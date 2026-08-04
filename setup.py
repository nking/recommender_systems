from setuptools import setup, find_packages

setup(
  name='movie_lens_tfx',
  version='0.2.0',
  packages=find_packages(where="src/main/python",
    include=['movie_lens_tfx', 'movie_lens_tfx.misc',
            'movie_lens_tfx.utils', 'movie_lens_tfx.ingest_component',
            'movie_lens_tfx.ingest_pyfunc_component',
            'movie_lens_tfx.ingest_already_split_pyfunc_component',
            'movie_lens_tfx.bulk_infer_component', 'movie_lens_tfx']),
  package_dir={'': 'src/main/python'},
  install_requires = [
    "pyarrow==18.1.0",
    "python-snappy==0.7.3",
    "numpy==2.1.0",
    "apache-beam==2.73.0",
    "tensorflow==2.21.0",
    "tensorflow-transform==1.21.0",
    "tensorboard==2.21.0",
    "tfx==1.21.0",
    "tensorflow-data-validation==1.21.0",
    "tensorflow-metadata==1.21.0",
    "ml-metadata==1.21.0",
    "pandas==2.2.3",
    "array-record==0.5.1",
    "msgpack==1.2.1",
  ],
  extras_require={"test": ["pytest", "nannyml>=0.13.1","polars>=1.35.2", "plotly>=6.4.0", "kaleido>=1.2.0"]},
  classifiers=[ 'Natural Language :: English',
               'Programming Language :: Python :: 3.11 :: Only',
               'Development Status :: 1 - Development/Unstable'
  ],
  url='https://www.kaggle.com/code/nicholeasuniquename/recommender-systems-with-tfx-pipelines',
  license='MIT',
  author='Nichole King',
  author_email='',
  description='TFX pipelines for Kaggle recommender systems project'
)
