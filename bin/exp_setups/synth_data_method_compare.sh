## Environment setup arguments

#####################################################
## Base paths
#####################################################

CUR_DIR=$(pwd)
BASE_OUT_DIR=${CUR_DIR}/exp_logs
SYNTH_DATA_ROOT_DIR=${CUR_DIR}/data/synthetic
RAW_DATA_ROOT_DIR=${CUR_DIR}/data/processed

#####################################################
## Hyperparameters
#####################################################

### Training config ###
TRAIN_CONFIG_NAME=tinyllama_1B_cpt
LR_SCHEDULER="cosine"
WARMUP=1000  
SEQ_LEN=2048

GLOBAL_BATCH_SIZE=96
TRAIN_NUM_GPU=4

# PER_GPU_BATCH_SIZE=12  # for A100 / H100 80GB
PER_GPU_BATCH_SIZE=24  # for H200 144GB

#####################################################
## Experiment setups
#####################################################

EXP_NAME_GROUP="synth_data_method_compare" 
WARMSTART_EXP_NOTES="Compare synthetic data generation with different baselines"
WARMSTART_NUM_TOKENS=8000000000  # ~1.6B total tokens, ~ 5 epochs


DATA_TYPE=${1:-"latent_thoughts"}
shift  # Shift to remove the first argument so we can process any additional args

echo "DATA_TYPE: ${DATA_TYPE}"

if [[ $DATA_TYPE == "latent_thoughts" ]]; then
    WARMSTART_EXP_NAME_EXTRA_SUFFIX="_latent_thought"
    WARMSTART_DATA_SETUP="data.root_dir=${SYNTH_DATA_ROOT_DIR} data.sources='{\"finemath_4plus_final_500m_warmstart_split_with_gpt_latent_thoughts_latents\": 1.0}'"
    WARMSTART_LATENT_TYPE_SWEEP=("random")
elif [[ $DATA_TYPE == "raw_repeat" ]]; then
    WARMSTART_EXP_NAME_EXTRA_SUFFIX="_raw_repeat"
    WARMSTART_DATA_SETUP="data.root_dir=${SYNTH_DATA_ROOT_DIR} data.sources='{\"finemath_4plus_final_500m_warmstart_split_with_gpt_latent_thoughts_latents\": 1.0}'"
    WARMSTART_LATENT_TYPE_SWEEP=("null")
elif [[ $DATA_TYPE == "raw_fresh" ]]; then
    WARMSTART_EXP_NAME_EXTRA_SUFFIX="_raw_fresh"
    WARMSTART_DATA_SETUP="data.root_dir=${RAW_DATA_ROOT_DIR} data.sources='{\"finemath_4plus_final_fixed_train_split\": 1.0}'"
    WARMSTART_LATENT_TYPE_SWEEP=("null")
elif [[ $DATA_TYPE == "wrap_cot" ]]; then
    MIX_RAW_RATIO=0.0
    WARMSTART_EXP_NAME_EXTRA_SUFFIX="_wrap_cot_mix=${MIX_RAW_RATIO}"
    WARMSTART_DATA_SETUP="data.root_dir=${SYNTH_DATA_ROOT_DIR} data.sources='{\"finemath_4plus_final_500m_warmstart_split_with_wrap_cot_latents\": 1.0}' data.mix_in_raw_ratio=${MIX_RAW_RATIO}"
    WARMSTART_LATENT_TYPE_SWEEP=("pure")
elif [[ $DATA_TYPE == "wrap_baseline" ]]; then
    MIX_RAW_RATIO=0.0
    WARMSTART_EXP_NAME_EXTRA_SUFFIX="_wrap_baseline_mix=${MIX_RAW_RATIO}"
    WARMSTART_DATA_SETUP="data.root_dir=${SYNTH_DATA_ROOT_DIR} data.sources='{\"finemath_4plus_final_500m_warmstart_split_with_wrap_baseline_latents\": 1.0}' data.mix_in_raw_ratio=${MIX_RAW_RATIO}"
    WARMSTART_LATENT_TYPE_SWEEP=("pure")
else
    echo "Invalid DATA_TYPE: ${DATA_TYPE}"
    exit 1
fi


WARMSTART_LR_SWEEP=(1e-4)
WARMSTART_EXTRA_KWARGS="checkpoint.must_eval_steps=[500,1250]"   # this will override the default in the script
WARMSTART_EXP_NAME=train_${EXP_NAME_GROUP}_warmstart
TRAIN_WARMSTART_TRIALS=(0)