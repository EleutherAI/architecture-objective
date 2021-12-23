python3 -c "import jax; print(jax.device_count()); print(jax.local_device_count())"

# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
EXPERIMENT_NAME=$1
MODEL_DIR="gs://bigscience-t5x/arch_objective_exps_v2/$EXPERIMENT_NAME"

# directory where the T5X repo is cloned.
T5X_DIR="/home/thomas/code/t5x"
export PYTHONPATH=${T5X_DIR}/bigscience/gins

# Logs
LOGS_PATH="/home/thomas/logs"
mkdir -p $LOGS_PATH

HF_DATASETS_OFFLINE=1 python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="bigscience/gins/$EXPERIMENT_NAME.gin" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  2>&1 | tee $LOGS_PATH/pretrain_$EXPERIMENT_NAME.txt

# sh bigscience/scripts/pretrain.sh c_dec_c4_full_lm

