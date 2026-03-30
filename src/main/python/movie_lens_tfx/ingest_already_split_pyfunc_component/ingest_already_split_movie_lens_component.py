import pprint

from tfx.types import standard_artifacts, artifact_utils, \
    channel_utils
from tfx.dsl.component.experimental import annotations
from tfx.dsl.component.experimental.decorators import component
# from tfx.dsl.components.component import component

from movie_lens_tfx.utils.ingest_movie_lens_beam import *

from tfx import v1 as tfx

import tensorflow as tf

tf.get_logger().propagate = False
logging.set_verbosity(logging.WARNING)
logging.set_stderrthreshold(logging.WARNING)
pp = pprint.PrettyPrinter()

logging.debug(f"TensorFlow version: {tf.__version__}")
logging.debug(f"TFX version: {tfx.__version__}")


# when use_beam=True, when the TFX pipeline is compiled and run with a
# Beam orchestrator, TFX automatically injects a Beam pipeline into
# that argument, there is no need to supply it directly.

# A parameter is an argument (int, float, bytes, or unicode string)

@component(use_beam=True)
def MovieLensSplitExampleGen(
        # name: tfx.dsl.components.Parameter[str],
        infiles_dict_train_ser: tfx.dsl.components.Parameter[str],
        infiles_dict_val_ser: tfx.dsl.components.Parameter[str],
        infiles_dict_test_ser: tfx.dsl.components.Parameter[str],
        # output_config_ser: tfx.dsl.components.Parameter[str], #supplied internally with train, val, test splite
        output_examples: tfx.dsl.components.OutputArtifact[
            standard_artifacts.Examples],
        beam_pipeline: annotations.BeamComponentParameter[
            beam.Pipeline] = None):
    """
    ingest the ratings, movies, and users files, left join them on ratings,
    and split them into the given buckets in output_config.
    
    Args:
      :param infiles_dict_train_ser: a string created from using base64 and pickle on the infiles_dict created with
        movie_lens_utils.create_infiles_dict where its input arguments are made
        from movie_lens_utils.create_infile_dict
      : param infiles_dict_val_ser
      : param infiles_dict_test_ser
      :param output_examples:
        ChannelParameter(type=standard_artifacts.Examples),
      :param beam_pipeline: injected into method by TFX.  do not supply
        this value
    """
    logging.info("MovieLensSplitExampleGen")
    
    try:
        infiles_dict_train = deserialize(infiles_dict_train_ser)
        infiles_dict_val = deserialize(infiles_dict_val_ser)
        infiles_dict_test = deserialize(infiles_dict_test_ser)
    except Exception as ex:
        err = f"error with deserialize(infiles_dict_..._ser)"
        logging.error(f'{err} : {ex}')
        raise ValueError(f'{err} : {ex}')
    
    for infiles_dict in [infiles_dict_train, infiles_dict_val,
        infiles_dict_test]:
        err = infiles_dict_formedness_error(infiles_dict)
        if err:
            logging.error(err)
            raise ValueError(err)
    
    # creates Split-eval  Split-test  Split-train directories under MovieLensSplitExampleGen
    # fr=or names split_names = ['train', 'eval', 'test']
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1),
                example_gen_pb2.SplitConfig.Split(name='test', hash_buckets=1)
            ]
        )
    )
    
    split_names = ["train", "eval", "test"]
    
    if not output_examples:
        examples_artifact = standard_artifacts.Examples()
        examples_artifact.splits = split_names
        output_examples = channel_utils.as_channel([examples_artifact])
    else:
        logging.debug(f"output_examples was passed in to component")
    
    if isinstance(output_examples, list):
        output_uri = artifact_utils.get_single_instance( output_examples).uri
        for artifact in output_examples:
            # this is just json.dumps after some type checking
            artifact.split_names = artifact_utils.encode_split_names(
                split_names)
            if "version" in infiles_dict:
                artifact.version = infiles_dict["version"]
                artifact.span = 0
    else:
        output_uri = output_examples.uri
        output_examples.split_names = artifact_utils.encode_split_names(
            split_names)
        if "version" in infiles_dict:
            output_examples.version = infiles_dict["version"]
            output_examples.span = 0
    
    logging.debug(f"output_examples TYPE={type(output_examples)}")
    logging.debug(f"output_examples={output_examples}")
    logging.debug(f"split_names={split_names}")
    
    with beam_pipeline as pipeline:
        # beam.PCollection, List[Tuple[str, Any]
        ratings_dict = {}
        for infiles_dict, split_name in zip([infiles_dict_train, infiles_dict_val,infiles_dict_test], split_names):
        
            ratings_pc, column_name_type_list = \
                pipeline | f"IngestAndJoin_{random.randint(0, 1000000000)}" \
                >> IngestAndJoin(infiles_dict=infiles_dict)
            
            logging.debug(f'column_name_type_list={column_name_type_list}')
            
            # create tf.train.Examples from PCollection before split:
            ratings_examples = ratings_pc \
                | f'ToTFExample_{random.randint(0, 1000000000000)}' \
                >> beam.Map(create_example, column_name_type_list)
    
            ratings_dict[split_name] = ratings_examples
        
        # write to TFRecords
        for name, example in ratings_dict.items():
            prefix_path = get_file_prefix_path(output_uri, name)
            logging.debug(f"prefix_path={prefix_path}")
            example | f"Serialize_{random.randint(0, 1000000000000)}" \
            >> beam.Map(lambda x: x.SerializeToString()) \
            | f"write_to_tfrecord_{random.randint(0, 1000000000000)}" \
            >> beam.io.tfrecordio.WriteToTFRecord( \
                file_path_prefix=prefix_path, file_name_suffix='.gz')
        logging.info('output_examples written as TFRecords')
