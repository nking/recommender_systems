from ml_metadata.proto import metadata_store_pb2
from ml_metadata.metadata_store import metadata_store
import os

PIPELINE_ROOT = "../TMP10/bin/rs_pipeline"

METADATA_PATH = os.path.join(PIPELINE_ROOT, 'tfx_metadata', 'metadata.db')

if not os.path.exists(METADATA_PATH):
    raise ValueError(f"cannot find {METADATA_PATH}")

metadata_connection_config = metadata_store_pb2.ConnectionConfig()
metadata_connection_config.sqlite.filename_uri = METADATA_PATH
store = metadata_store.MetadataStore(metadata_connection_config)


# Verify by counting recorded artifacts and executions
print('===================================================================')
print(f"Total Artifacts recorded: {len(store.get_artifacts())}")

pusher_type = store.get_execution_type("tfx.components.pusher.component.Pusher")
pushed_artifacts = store.get_artifacts_by_type('PushedModel')
if not pushed_artifacts:
    print("No PushedModel artifacts found.")
else:
    latest_pushed = pushed_artifacts[-1]
    print(f"Pushed Model ID: {latest_pushed.id}")
    print(f"Pushed Model URI: {latest_pushed.uri}")

    # 3. Find the Pusher execution that outputted this PushedModel
    events = store.get_events_by_artifact_ids([latest_pushed.id])
    pusher_execution_id = [e.execution_id for e in events if e.type == metadata_store_pb2.Event.OUTPUT][0]

    # 4. Find the input artifact that fed into that Pusher execution (this is the Trainer model)
    execution_events = store.get_events_by_execution_ids([pusher_execution_id])
    trainer_artifact_id = [e.artifact_id for e in execution_events if e.type == metadata_store_pb2.Event.INPUT][0]
    
    trainer_artifact = store.get_artifacts_by_id([trainer_artifact_id])[0]
    print(f"Corresponding Trainer Model URI: {trainer_artifact.uri}")
executions = store.get_executions_by_type(pusher_type.name)
latest_execution = max(executions, key=lambda x: x.last_update_time_since_epoch)
# Get all events associated with this execution
events = store.get_events_by_execution_ids( [latest_execution.id])
output_artifact_ids = [
    event.artifact_id for event in events
    if event.type == metadata_store_pb2.Event.OUTPUT
]
artifacts = store.get_artifacts_by_id(output_artifact_ids)

print('===================================================================')
print(f'Pusher executions: {executions}')
#print(f'Pusher events: {events}')

print('===================================================================')
artifacts = store.get_artifacts_by_type('PushedModel')
print(f'pushed model artifacts={artifacts}')
print('===================================================================')

print(f"Total Artifacts recorded: {store.get_artifacts()}")

print('===================================================================')
pushed_artifacts = store.get_artifacts_by_type('PushedModel')
if not pushed_artifacts:
    print("No PushedModel artifacts found.")
else:
    latest_pushed = pushed_artifacts[-1]
    print(f"Pushed Model ID: {latest_pushed.id}")
    print(f"Pushed Model URI: {latest_pushed.uri}")

    # 3. Find the Pusher execution that outputted this PushedModel
    events = store.get_events_by_artifact_ids([latest_pushed.id])
    pusher_execution_id = [e.execution_id for e in events if e.type == metadata_store_pb2.Event.OUTPUT][0]

    # 4. Find the input artifact that fed into that Pusher execution (this is the Trainer model)
    execution_events = store.get_events_by_execution_ids([pusher_execution_id])
    trainer_artifact_id = [e.artifact_id for e in execution_events if e.type == metadata_store_pb2.Event.INPUT][0]
    
    trainer_artifact = store.get_artifacts_by_id([trainer_artifact_id])[0]
    print(f"Corresponding Trainer Model URI: {trainer_artifact.uri}")
print('===================================================================')
pushed_artifacts = store.get_artifacts_by_type('PushedModel')
if not pushed_artifacts:
    print("No PushedModel artifacts found.")
else:
    latest_pushed = pushed_artifacts[-1]
    
    # 2. Find the Pusher execution that outputted it
    p_events = store.get_events_by_artifact_ids([latest_pushed.id])
    pusher_exec_id = [e.execution_id for e in p_events if e.type == metadata_store_pb2.Event.OUTPUT][0]

    # 3. Find all inputs to the Pusher execution
    pusher_inputs = store.get_events_by_execution_ids([pusher_exec_id])
    
    for event in pusher_inputs:
        if event.type == metadata_store_pb2.Event.INPUT:
            art = store.get_artifacts_by_id([event.artifact_id])[0]
            art_type = store.get_artifact_types_by_id([art.type_id])[0].name
            
            # If the input directly points to a Model artifact
            if art_type == 'Model':
                print(f"Pusher Directly Consumed Model URI: {art.uri}")
            
            # If the input is a ModelBlessing, trace that blessing back to its Trainer model input
            elif art_type == 'ModelBlessing':
                blessing_events = store.get_events_by_artifact_ids([art.id])
                eval_exec_ids = [e.execution_id for e in blessing_events if e.type == metadata_store_pb2.Event.OUTPUT]
                
                if eval_exec_ids:
                    eval_inputs = store.get_events_by_execution_ids([eval_exec_ids[0]])
                    for ev_event in eval_inputs:
                        if ev_event.type == metadata_store_pb2.Event.INPUT:
                            in_art = store.get_artifacts_by_id([ev_event.artifact_id])[0]
                            in_art_type = store.get_artifact_types_by_id([in_art.type_id])[0].name
                            if in_art_type == 'Model':
                                print(f"Definitive Pushed Trainer Model URI: {in_art.uri}")

print('===================================================================')
