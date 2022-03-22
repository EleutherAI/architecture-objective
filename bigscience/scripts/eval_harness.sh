pushd ~/code/t5x

ORIGINAL_EXPERIMENT_NAME=$1
# The idea is whether to load the nc_dec as c_dec or vice-versa. "true" will switch
OPPOSITE_ARCHITECTURE=$2

if [[ $ORIGINAL_EXPERIMENT_NAME == *t0_adapt* ]]
then
  CHECKPOINT_STEP=141072 # 131072 (pretrain) + 10000 (t0 adapt)
else
  CHECKPOINT_STEP=131072
fi

# Logs
LOGS_PATH="~/logs"
mkdir -p $LOGS_PATH

if [[ $ORIGINAL_EXPERIMENT_NAME == c_dec* ]]
then
  if [[ $OPPOSITE_ARCHITECTURE != true ]]
  then
    MODEL_GIN_FILE=c_dec_xxl.gin
  else
    echo "Using opposite architecture"
    MODEL_GIN_FILE=nc_dec_xxl.gin
  fi
fi
if [[ $ORIGINAL_EXPERIMENT_NAME == nc_dec* ]]
then
  if [[ $OPPOSITE_ARCHITECTURE != true ]]
  then
    MODEL_GIN_FILE=nc_dec_xxl.gin
  else
    echo "Using opposite architecture"
    MODEL_GIN_FILE=c_dec_xxl.gin
  fi
fi
if [[ $ORIGINAL_EXPERIMENT_NAME == enc_dec* ]]
then
  if [[ $OPPOSITE_ARCHITECTURE != true ]]
  then
    MODEL_GIN_FILE=enc_dec_xxl.gin
  else
    echo "Cannot have opposite architecture for enc dec."
    exit 1
  fi
fi
if [[ $MODEL_GIN_FILE == "" ]]
then
  echo "Incorrect experiment name $ORIGINAL_EXPERIMENT_NAME, does not start with c_dec/nc_dec/enc_dec"
  exit
fi

echo "Load model gin: $MODEL_GIN_FILE"

EXPERIMENT_NAME=$ORIGINAL_EXPERIMENT_NAME"_eai_eval_"$CHECKPOINT_STEP
MODEL_GIN_FILE=bigscience/gins/$MODEL_GIN_FILE
EVAL_OUTPUT_DIR=gs://bigscience-t5x/arch_objective_exps_v2/eai_eval/"$EXPERIMENT_NAME".json

HF_DATASETS_OFFLINE=0 PYTHONPATH=$(pwd)/bigscience/gins python3 $(pwd)/t5x/eval_harness.py \
   --gin_file_="$MODEL_GIN_FILE" \
   --gin_file_="bigscience/gins/eval_harness.gin" \
   --gin.INFER_OUTPUT_DIR="'.'"  \
   --gin.DROPOUT_RATE=0.0 \
   --gin.CHECKPOINT_PATH="'gs://bigscience-t5x/arch_objective_exps_v2/$ORIGINAL_EXPERIMENT_NAME/checkpoint_$CHECKPOINT_STEP'" \
   --results_path $EVAL_OUTPUT_DIR \
   --tasks=anli_r1,anli_r2,anli_r3,arc_challenge,arc_easy,boolq,cb,copa,headqa_en,hellaswag,lambada,logiqa,mathqa,mc_taco,mrpc,multirc,openbookqa,piqa,prost,pubmedqa,qnli,qqp,race,rte,sciq,sst,triviaqa,webqs,wic,winogrande,wnli,wsc \
   2>&1 | tee $LOGS_PATH/eai_eval_$EXPERIMENT_NAME.txt
