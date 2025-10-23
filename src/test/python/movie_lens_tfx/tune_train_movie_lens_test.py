
import shutil

from tfx.dsl.io import fileio
from tfx.orchestration import metadata
from tfx.components import StatisticsGen, SchemaGen

from ingest_movie_lens_component import *
#import trainer_movie_lens

from ml_metadata.proto import metadata_store_pb2
from ml_metadata.metadata_store import metadata_store
from tune_train_movie_lens import *

from helper import *

from absl import logging
tf.get_logger().propagate = False
logging.set_verbosity(logging.DEBUG)
logging.set_stderrthreshold(logging.DEBUG)

class TuneTrainTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.infiles_dict_ser, self.output_config_ser, self.split_names = \
      get_test_data()
    self.user_id_max = 6040
    self.movie_id_max = 3952
    self.n_genres = N_GENRES
    self.n_age_groups = N_AGE_GROUPS
    self.n_occupations = 21
    self.name = 'test run of ratings transform'

  def test_tune_and_train(self):
    test_num = "1"
    ratings_example_gen = (MovieLensExampleGen(
      infiles_dict_ser=self.infiles_dict_ser,
      output_config_ser = self.output_config_ser))

    statistics_gen = StatisticsGen(examples = ratings_example_gen.outputs['output_examples'])

    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=True)

    if get_kaggle():
      tr_dir = "/kaggle/working/"
    else:
      tr_dir = os.path.join(get_project_dir(), "src/main/python/movie_lens_tfx")

    ratings_transform = tfx.components.Transform(
      examples=ratings_example_gen.outputs['output_examples'],
      schema=schema_gen.outputs['schema'],
      module_file=os.path.join(tr_dir, 'transform_movie_lens.py'))

    tuner_custom_config = {
      'user_id_max' : self.user_id_max,
      'movie_id_max' : self.movie_id_max,
      'n_genres' : self.n_genres,
      'n_age_groups' : self.n_age_groups,
      'run_eagerly' : True,
      "use_bias_corr" : False,
      'incl_genres':True,
    }
    
    tuner = tfx.components.Tuner(
      module_file=os.path.join(tr_dir, 'tune_train_movie_lens.py'),
      examples=ratings_transform.outputs['transformed_examples'],
      #schema is already in the transform graph
      transform_graph=ratings_transform.outputs['transform_graph'],
      #args: splits, num_steps.  splits defaults are assumed if none given
      train_args=tfx.proto.TrainArgs(num_steps=20),
      eval_args=tfx.proto.EvalArgs(num_steps=5),
      custom_config=tuner_custom_config,
    )
    #'user_id_max' 'movie_id_max' 'n_genres' 'run_eagerly'

    #see https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local.py
    #trainer = trainer_movie_lens.MovieLensTrainer(
    trainer=tfx.components.Trainer(
      module_file=os.path.join(tr_dir, 'tune_train_movie_lens.py'),
      examples=ratings_transform.outputs['transformed_examples'],
      transform_graph=ratings_transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      # If Tuner is in the pipeline, Trainer can take Tuner's output
      # best_hyperparameters artifact as input and utilize it in the user module
      # code.
      #
      # If there isn't Tuner in the pipeline, either use Importer to import
      # a previous Tuner's output to feed to Trainer, or directly use the tuned
      # hyperparameters in user module code and set hyperparameters to None
      # here.
      #
      # Example of Importer,
      #   hparams_importer = Importer(
      #     source_uri='path/to/best_hyperparameters.txt',
      #     artifact_type=HyperParameters).with_id('import_hparams')
      #   ...
      #   hyperparameters = hparams_importer.outputs['result'],
      hyperparameters=(tuner.outputs['best_hyperparameters']),
      train_args=tfx.proto.TrainArgs(num_steps=5),
      eval_args=tfx.proto.EvalArgs(num_steps=5)
    )

    #Tuner Component `outputs` contains:
    #- `model`: Channel of type [`standard_artifacts.Model`][tfx.v1.types.standard_artifacts.Model] for trained model.
    #- `model_run`: Channel of type [`standard_artifacts.ModelRun`][tfx.v1.types.standard_artifacts.ModelRun], as the working
    #              dir of models, can be used to output non-model related output
    #              (e.g., TensorBoard logs).

    #TODO: continue with components;
    # see https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local.py

    components = [ratings_example_gen, statistics_gen, schema_gen,
                  ratings_transform, tuner, trainer]

    PIPELINE_NAME = 'TestPythonTransformPipeline'
    #output_data_dir = os.path.join(os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',self.get_temp_dir()),self._testMethodName)
    output_data_dir = os.path.join(get_bin_dir(), test_num, self._testMethodName)
    PIPELINE_ROOT = os.path.join(output_data_dir, PIPELINE_NAME)
    #remove results from previous test runs:
    try:
      shutil.rmtree(PIPELINE_ROOT)
    except OSError as e:
      pass
    METADATA_PATH = os.path.join(PIPELINE_ROOT, 'tfx_metadata', 'metadata.db')
    os.makedirs(os.path.join(PIPELINE_ROOT, 'tfx_metadata'), exist_ok=True)

    ENABLE_CACHE = False

    #metadata_connection_config = metadata_store_pb2.ConnectionConfig()
    #metadata_connection_config.sqlite.SetInParent()
    #metadata_connection = metadata.Metadata(metadata_connection_config)
    metadata_connection_config = metadata.sqlite_metadata_connection_config(METADATA_PATH)

    beam_pipeline_args = [
      '--direct_running_mode=multi_processing',
      '--direct_num_workers=0'
    ]

    #imple is tfx.v1.dsl.Pipeline  where tfx is aliased in import
    my_pipeline = tfx.dsl.Pipeline(
      pipeline_name=PIPELINE_NAME,
      pipeline_root=PIPELINE_ROOT,
      components=components,
      enable_cache=ENABLE_CACHE,
      metadata_connection_config=metadata_connection_config,
      beam_pipeline_args=beam_pipeline_args,
    )

    tfx.orchestration.LocalDagRunner().run(my_pipeline)

    #metadata_connection = metadata.Metadata(metadata_connection_config)
    store = metadata_store.MetadataStore(metadata_connection_config)
    artifact_types = store.get_artifact_types()
    logging.debug(f"MLMD store artifact_types={artifact_types}")
    artifacts = store.get_artifacts()
    logging.debug(f"MLMD store artifacts={artifacts}")
    #id: 8
    #type_id: 15
    #uri:
    # "/kaggle/working/bin/transform_1/test_MovieLensExampleGen/
    #   TestPythonTransformPipeline/Transform/transformed_examples/4"

    executions = store.get_executions()
    logging.debug(f"MLMD store executions={executions}")
    self.assertLessEqual(4, len(artifact_types))
    self.assertLessEqual(4, len(artifacts))
    self.assertLessEqual(4, len(executions))
    # executions has custom_properties.key: "infiles_dict_ser"
    #    and custom_properties.key: "output_config_ser"
    artifact_count = len(artifacts)
    execution_count = len(executions)
    self.assertGreaterEqual(artifact_count, execution_count)
    self.assertGreaterEqual(artifact_count, execution_count)

    artifact_uri = artifacts[0].uri
    for split_name in self.split_names:
      dir_path = f'{artifact_uri}/{get_split_dir_name(split_name)}'
      file_paths = [os.path.join(dir_path, name) for name in
                    os.listdir(dir_path)]
      #  file_paths = get_output_files(ratings_example_gen, 'output_examples', split_name)
      self.assertGreaterEqual(len(file_paths), 1)

    logging.debug(f"ratings_transform.id={ratings_transform.id}") #StatisticsGen
    logging.debug(f"ratings_transform={ratings_transform}")
    self.assertTrue(fileio.exists(os.path.join(PIPELINE_ROOT, ratings_transform.id)))

    #https://github.com/tensorflow/tfx/blob/e537507b0c00d45493c50cecd39888092f1b3d79/tfx/types/standard_artifacts.py#L487
    #TransformCache
    #TransformGraph
    transform_graph_list = store.get_artifacts_by_type("TransformGraph")
    logging.debug(f"transform_graph_list={transform_graph_list}")
    latest_transform_graph_artifact = sorted(transform_graph_list, \
      key=lambda x: x.create_time_since_epoch, reverse=True)[0]
    #or use last_update_time_since_epoch
    transform_graph_uri = latest_transform_graph_artifact.uri
    logging.debug(f"transform_graph_uri={transform_graph_uri}")

    #/kaggle/working/bin/transform_1/test_MovieLensExampleGen/TestPythonTransformPipeline/
    #   MovieLensExampleGen/output_examples/1/
    #   Split-<train, eval, or test>/data_tfrecord-0000?-of-00004.tfrecord

    #component outputs contains: https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Transform#example
    #transform_graph: Channel of type standard_artifacts.TransformGraph,
    #   includes an exported Tensorflow graph suitable for both training and serving.
    #transformed_examples: Channel of type standard_artifacts.Examples
    #   for materialized transformed examples, which includes transform splits as specified in splits_config

    executions = store.get_executions_by_type(type_name="Transform")
    for execution in executions:
      events = store.get_events_by_execution_ids([execution.id])
      for event in events:
        if event.type == metadata_store_pb2.Event.OUTPUT:
          artifact = store.get_artifacts_by_id([event.artifact_id])[0]
          if artifact.type_id == store.get_artifact_type('Examples').id:
            logging.debug(f"artifact={artifact}\nuri={artifact.uri}")

    #!find /kaggle/working/bin -type f -iname "transformed_examples*.gz"
    """
    logging.debug(f"component transformed_examples uri={ratings_transform.outputs['transformed_examples'].get()[0].uri}")
    logging.debug(f"component post_transform_schema uri={ratings_transform.outputs['post_transform_schema'].get()[0].uri}")
    stats_path_train = os.path.join(transform_graph_uri, get_split_dir_name("train"), 'FeatureStats.pb')
    stats_path_eval = os.path.join(transform_graph_uri, get_split_dir_name("eval"), 'FeatureStats.pb')
    stats_path_test = os.path.join(transform_graph_uri, get_split_dir_name("test"),'FeatureStats.pb')
    self.assertTrue(os.path.exists(stats_path_train))
    self.assertTrue(os.path.exists(stats_path_eval))
    self.assertTrue(os.path.exists(stats_path_test))

    tfrecord_filenames = [os.path.join(stats_path_train, name) for name in os.listdir(stats_path_train)]
    dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")
    #might need to parse the data
    for tfrecord in dataset.take(5):
      example = tf.train.Example()
      example.ParseFromString(tfrecord.numpy())
      logging.debug(f"a transform example={example}")
    """
