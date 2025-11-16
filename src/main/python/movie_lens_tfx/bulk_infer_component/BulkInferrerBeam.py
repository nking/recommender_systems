# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TFX bulk_inferrer executor."""
## edits made in this project for running a saved_model

import importlib
import os
from typing import Any, Callable, Dict, List, Optional, Union

from absl import logging
import apache_beam as beam
import tensorflow as tf
from tfx import types
from tfx.components.bulk_inferrer import prediction_to_example_utils
from tfx.components.util import model_utils
from tfx.components.util import tfxio_utils
from tfx.dsl.components.base import base_beam_executor
from tfx.proto import bulk_inferrer_pb2
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import path_utils
from tfx.utils import proto_utils
#from tfx_bsl.public.beam import run_inference
import movie_lens_tfx.bulk_infer_component.tfx_bsl_public_run_inference as run_inference
from tfx_bsl.public.proto import model_spec_pb2
from tfx_bsl.tfxio import record_based_tfxio

from typing import Optional, Union

from tfx import types
from tfx.components.bulk_inferrer import executor
from tfx.dsl.components.base import base_beam_component
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import data_types
from tfx.proto import bulk_inferrer_pb2
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs

from tensorflow_serving.apis import prediction_log_pb2

##======================
import apache_beam as beam
from typing import Any, Iterable, Text, Dict
from tfx_bsl.public.beam.run_inference import ModelHandler
from tensorflow_serving.apis import prediction_log_pb2

# Workarounds for importing extra dependencies. Do not add more.
for name in ['tensorflow_text', 'tensorflow_recommenders']:
  try:
    importlib.import_module(name)
  except ImportError:
    logging.info('%s is not available.', name)


_PREDICTION_LOGS_FILE_NAME = 'prediction_logs'
_EXAMPLES_FILE_NAME = 'examples'
_TELEMETRY_DESCRIPTORS = ['BulkInferrer']


class BulkInferrerBeamExecutor(base_beam_executor.BaseBeamExecutor):
  """TFX bulk inferer executor."""

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    """Runs batch inference on a given model with given input examples.

    Args:
      input_dict: Input dict from input key to a list of Artifacts.
        - examples: examples for inference.
        - model: exported model.
        - model_blessing: model blessing result, optional.
      output_dict: Output dict from output key to a list of Artifacts.
        - output: bulk inference results.
      exec_properties: A dict of execution properties.
        - model_spec: JSON string of bulk_inferrer_pb2.ModelSpec instance.
        - data_spec: JSON string of bulk_inferrer_pb2.DataSpec instance.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    if output_dict.get(standard_component_specs.INFERENCE_RESULT_KEY):
      inference_result = artifact_utils.get_single_instance(
          output_dict[standard_component_specs.INFERENCE_RESULT_KEY])
    else:
      inference_result = None
    if output_dict.get(standard_component_specs.OUTPUT_EXAMPLES_KEY):
      output_examples = artifact_utils.get_single_instance(
          output_dict[standard_component_specs.OUTPUT_EXAMPLES_KEY])
    else:
      output_examples = None

    if 'examples' not in input_dict:
      raise ValueError('\'examples\' is missing in input dict.')
    if 'model' not in input_dict:
      raise ValueError('Input models are not valid, model '
                       'need to be specified.')
    if standard_component_specs.MODEL_BLESSING_KEY in input_dict:
      model_blessing = artifact_utils.get_single_instance(
          input_dict[standard_component_specs.MODEL_BLESSING_KEY])
      if not model_utils.is_model_blessed(model_blessing):
        logging.info('Model on %s was not blessed', model_blessing.uri)
        return
    else:
      logging.info('Model blessing is not provided, exported model will be '
                   'used.')

    model = artifact_utils.get_single_instance(
        input_dict[standard_component_specs.MODEL_KEY])
    model_path = path_utils.serving_model_path(
        model.uri, path_utils.is_old_model_artifact(model))
    logging.info('Use exported model from %s.', model_path)

    data_spec = bulk_inferrer_pb2.DataSpec()
    proto_utils.json_to_proto(
        exec_properties[standard_component_specs.DATA_SPEC_KEY], data_spec)

    output_example_spec = bulk_inferrer_pb2.OutputExampleSpec()
    if exec_properties.get(standard_component_specs.OUTPUT_EXAMPLE_SPEC_KEY):
      proto_utils.json_to_proto(
          exec_properties[standard_component_specs.OUTPUT_EXAMPLE_SPEC_KEY],
          output_example_spec)

    self._run_model_inference(
        data_spec, output_example_spec,
        input_dict[standard_component_specs.EXAMPLES_KEY], output_examples,
        inference_result, self._get_inference_spec(model_path, exec_properties))

  def _get_inference_spec(
      self, model_path: str,
      exec_properties: Dict[str, Any]) -> model_spec_pb2.InferenceSpecType:
    model_spec = bulk_inferrer_pb2.ModelSpec()
    proto_utils.json_to_proto(
        exec_properties[standard_component_specs.MODEL_SPEC_KEY], model_spec)
    saved_model_spec = model_spec_pb2.SavedModelSpec(
        model_path=model_path,
        tag=model_spec.tag,
        signature_name=model_spec.model_signature_name)
    result = model_spec_pb2.InferenceSpecType()
    result.saved_model_spec.CopyFrom(saved_model_spec)
    return result

  def _run_model_inference(
      self,
      data_spec: bulk_inferrer_pb2.DataSpec,
      output_example_spec: bulk_inferrer_pb2.OutputExampleSpec,
      examples: List[types.Artifact],
      output_examples: Optional[types.Artifact],
      inference_result: Optional[types.Artifact],
      inference_endpoint: model_spec_pb2.InferenceSpecType,
  ) -> None:
    """Runs model inference on given examples data.

    Args:
      data_spec: bulk_inferrer_pb2.DataSpec instance.
      output_example_spec: bulk_inferrer_pb2.OutputExampleSpec instance.
      examples: List of `standard_artifacts.Examples` artifacts.
      output_examples: Optional output `standard_artifacts.Examples` artifact.
      inference_result: Optional output `standard_artifacts.InferenceResult`
        artifact.
      inference_endpoint: Model inference endpoint.
    """

    example_uris = {}
    for example_artifact in examples:
      for split in artifact_utils.decode_split_names(
          example_artifact.split_names):
        if data_spec.example_splits:
          if split in data_spec.example_splits:
            example_uris[split] = artifact_utils.get_split_uri(
                [example_artifact], split)
        else:
          example_uris[split] = artifact_utils.get_split_uri([example_artifact],
                                                             split)

    payload_format, _ = tfxio_utils.resolve_payload_format_and_data_view_uri(
        examples)

    tfxio_factory = tfxio_utils.get_tfxio_factory_from_artifact(
        examples,
        _TELEMETRY_DESCRIPTORS,
        schema=None,
        read_as_raw_records=True,
        # We have to specify this parameter in order to create a RawRecord TFXIO
        # but we won't use the RecordBatches so the column name of the raw
        # records does not matter.
        raw_record_column_name='unused')

    if output_examples:
      output_examples.split_names = artifact_utils.encode_split_names(
          sorted(example_uris.keys()))

    with self._make_beam_pipeline() as pipeline:
      data_list = []
      for split, example_uri in example_uris.items():
        #tfxio type is <tfx_bsl.tfxio.raw_tf_record.RawTfRecordTFXIO object at 0x7b4008142aa0>
        tfxio = tfxio_factory([io_utils.all_files_pattern(example_uri)])
        assert isinstance(tfxio, record_based_tfxio.RecordBasedTFXIO), (
            'Unable to use TFXIO {} as it does not support reading raw records.'
            .format(type(tfxio)))
        # pylint: disable=no-value-for-parameter
        data = (pipeline
                | 'ReadData[{}]'.format(split) >> tfxio.RawRecordBeamSource()
                | 'RunInference[{}]'.format(split) >> _RunInference(
                    payload_format, inference_endpoint))
        
        if output_examples:
          output_examples_split_uri = artifact_utils.get_split_uri(
              [output_examples], split)
          logging.info('Path of output examples split `%s` is %s.', split,
                       output_examples_split_uri)
          _ = (
              data
              | 'WriteExamples[{}]'.format(split) >> _WriteExamples(
                  output_example_spec, output_examples_split_uri))
          # pylint: enable=no-value-for-parameter

        data_list.append(data)

      if inference_result:
        _ = (
            data_list
            | 'FlattenInferenceResult' >> beam.Flatten(pipeline=pipeline)
            | 'WritePredictionLogs' >> beam.io.WriteToTFRecord(
                os.path.join(inference_result.uri, _PREDICTION_LOGS_FILE_NAME),
                file_name_suffix='.gz',
                coder=beam.coders.ProtoCoder(prediction_log_pb2.PredictionLog)))

    if output_examples:
      logging.info('Output examples written to %s.', output_examples.uri)
    if inference_result:
      logging.info('Inference result written to %s.', inference_result.uri)


def _MakeParseFn(
    payload_format: int
) -> Union[Callable[[bytes], tf.train.Example], Callable[
    [bytes], tf.train.SequenceExample]]:
  """Returns a function to parse bytes to payload."""
  if payload_format == example_gen_pb2.PayloadFormat.FORMAT_TF_EXAMPLE:
    return tf.train.Example.FromString
  elif (payload_format ==
        example_gen_pb2.PayloadFormat.FORMAT_TF_SEQUENCE_EXAMPLE):
    return tf.train.SequenceExample.FromString
  else:
    raise NotImplementedError(
        'Payload format %s is not supported.' %
        example_gen_pb2.PayloadFormat.Name(payload_format))

#NLK: adding a load function for saved_model
def _load_override_fn(model_path, tags):
  logging.debug(f'invoking _load_override_fn')
  #Callable[[str, Sequence[str]], Any]
  return tf.saved_model.load(model_path)
  
#NLK: cannot pass this to run_inference.RunInference, but keeping here for reference
def _custom_inference_fn(
    loaded_model: tf.Module,
    features_dict: beam.pvalue.PCollection
) -> beam.pvalue.PCollection:
    """Performs inference using the loaded model and input features."""
    # The loaded_model is what was returned by your load_override_fn (or the default loader)
    infer = loaded_model.signatures["serving_default"]
    INPUT_KEY = list(infer.structured_input_signature[1].keys())[0]
    predictions = infer(**{INPUT_KEY: features_dict})
    return predictions

@beam.ptransform_fn
@beam.typehints.with_output_types(prediction_log_pb2.PredictionLog)
def _RunInference(
    pipeline: beam.pvalue.PCollection,
    payload_format: int,
    inference_endpoint: model_spec_pb2.InferenceSpecType
) -> beam.pvalue.PCollection:
  """Runs model inference on given examples data."""
  inferences = None
  try:
    #NLK: modification for tf_Example as serialized string, not parsed
    if inference_endpoint.saved_model_spec is not None:
      logging.debug(f'run inference with serialized tf_example')
      inferences = (pipeline | 'RunInference' >> run_inference.RunInference(
          inference_spec_type=inference_endpoint,
          load_override_fn=_load_override_fn))
  except Exception as e:
    pass
  if inferences is None:
    logging.debug(f'run default inference PTransforms')
    inferences = (pipeline | 'ParseExamples' >> beam.Map(_MakeParseFn(payload_format))
      | 'RunInference2' >> run_inference.RunInference(
      inference_spec_type=inference_endpoint, load_override_fn=_load_override_fn))
  #inferences | "print_inferences" >> beam.Map(lambda _: print(f"INFERENCE:{inferences.result()}"))
  return inferences


@beam.ptransform_fn
@beam.typehints.with_input_types(prediction_log_pb2.PredictionLog)
def _WriteExamples(prediction_log: beam.pvalue.PCollection,
                   output_example_spec: bulk_inferrer_pb2.OutputExampleSpec,
                   output_path: str) -> beam.pvalue.PDone:
  """Converts `prediction_log` to `tf.train.Example` and materializes."""
  return (prediction_log
          | 'ConvertToExamples' >> beam.Map(
              prediction_to_example_utils.convert,
              output_example_spec=output_example_spec)
          | 'WriteExamples' >> beam.io.WriteToTFRecord(
              os.path.join(output_path, _EXAMPLES_FILE_NAME),
              file_name_suffix='.gz',
              coder=beam.coders.ProtoCoder(tf.train.Example)))

class BulkInferrerBeam(base_beam_component.BaseBeamComponent):
  """A TFX component to do batch inference on a model with unlabelled examples.

  BulkInferrer consumes examples data and a model, and produces the inference
  results to an external location as PredictionLog proto.

  BulkInferrer will infer on validated model.

  ## Example
  ```
    # Uses BulkInferrer to inference on examples.
    bulk_inferrer = BulkInferrer(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'])
  ```

  Component `outputs` contains:

   - `inference_result`: Channel of type [`standard_artifacts.InferenceResult`][tfx.v1.types.standard_artifacts.InferenceResult]
                         to store the inference results.
   - `output_examples`: Channel of type [`standard_artifacts.Examples`][tfx.v1.types.standard_artifacts.Examples]
                        to store the output examples. This is optional
                        controlled by `output_example_spec`.

  See [the BulkInferrer
  guide](../../../guide/bulkinferrer) for more details.
  """

  SPEC_CLASS = standard_component_specs.BulkInferrerSpec
  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(BulkInferrerBeamExecutor)

  def __init__(
      self,
      examples: types.BaseChannel,
      model: Optional[types.BaseChannel] = None,
      model_blessing: Optional[types.BaseChannel] = None,
      data_spec: Optional[Union[bulk_inferrer_pb2.DataSpec,
                                data_types.RuntimeParameter]] = None,
      model_spec: Optional[Union[bulk_inferrer_pb2.ModelSpec,
                                 data_types.RuntimeParameter]] = None,
      output_example_spec: Optional[Union[bulk_inferrer_pb2.OutputExampleSpec,
                                          data_types.RuntimeParameter]] = None):
    """Construct an BulkInferrer component.

    Args:
      examples: A [BaseChannel][tfx.v1.types.BaseChannel] of type [`standard_artifacts.Examples`][tfx.v1.types.standard_artifacts.Examples], usually
        produced by an ExampleGen component. _required_
      model: A [BaseChannel][tfx.v1.types.BaseChannel] of type [`standard_artifacts.Model`][tfx.v1.types.standard_artifacts.Model], usually produced
        by a [Trainer][tfx.v1.components.Trainer] component.
      model_blessing: A [BaseChannel][tfx.v1.types.BaseChannel] of type [`standard_artifacts.ModelBlessing`][tfx.v1.types.standard_artifacts.ModelBlessing],
        usually produced by a ModelValidator component.
      data_spec: bulk_inferrer_pb2.DataSpec instance that describes data
        selection.
      model_spec: bulk_inferrer_pb2.ModelSpec instance that describes model
        specification.
      output_example_spec: bulk_inferrer_pb2.OutputExampleSpec instance, specify
        if you want BulkInferrer to output examples instead of inference result.
    """
    if output_example_spec:
      output_examples = types.Channel(type=standard_artifacts.Examples)
      inference_result = None
    else:
      inference_result = types.Channel(type=standard_artifacts.InferenceResult)
      output_examples = None

    spec = standard_component_specs.BulkInferrerSpec(
        examples=examples,
        model=model,
        model_blessing=model_blessing,
        data_spec=data_spec or bulk_inferrer_pb2.DataSpec(),
        model_spec=model_spec or bulk_inferrer_pb2.ModelSpec(),
        output_example_spec=output_example_spec,
        inference_result=inference_result,
        output_examples=output_examples)
    super().__init__(spec=spec)