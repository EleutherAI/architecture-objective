# Need to install seqio
# gcloud auth application-default login


MODULE_IMPORT=t5x.configs.dataset.pile.task
TASK_NAME=pile_t2t_span_corruption
JOB_NAME=pilet2tspancorruption # the name must consist of only the characters [-a-z0-9], starting with a letter and ending with a letter or number
BUCKET=gs://bigscience/pile/$TASK_NAME # Don't know is cache needs to be task specific or not ...
PROJECT=bigscience
REGION=europe-west1

seqio_cache_tasks \
 --module_import=$MODULE_IMPORT \
 --tasks=${TASK_NAME} \
 --output_cache_dir=${BUCKET}/cache \
 --setup_file t5x/configs/dataset/pile/dataflow_setup.py \
 --pipeline_options="--runner=DataflowRunner,--project=$PROJECT,--region=$REGION,--job_name=$JOB_NAME,--staging_location=$BUCKET/binaries,--temp_location=$BUCKET/tmp"