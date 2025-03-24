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
LR=1e-4
SEQ_LEN=2048

# PER_GPU_BATCH_SIZE=12  # for A100 / H100 80GB
PER_GPU_BATCH_SIZE=24  # for H200 144GB

WARMSTART_GLOBAL_BATCH_SIZE=96
WARMSTART_TRAIN_NUM_GPU=4
BOOTSTRAP_GLOBAL_BATCH_SIZE=192
BOOTSTRAP_TRAIN_NUM_GPU=8

if [[ ${RUN_SCRIPT} == "run_train_warmstart.sh" ]]; then
    # Training config
    GLOBAL_BATCH_SIZE=${WARMSTART_GLOBAL_BATCH_SIZE}
    TRAIN_NUM_GPU=${WARMSTART_TRAIN_NUM_GPU}
elif [[ ${RUN_SCRIPT} == "run_train_bootstrap.sh" ]]; then
    # Training config
    GLOBAL_BATCH_SIZE=${BOOTSTRAP_GLOBAL_BATCH_SIZE}
    TRAIN_NUM_GPU=${BOOTSTRAP_TRAIN_NUM_GPU}

    # Parse data setup and bootstrap iteration
    DATA_SETUP=${1:-"bootstrap_latent"}
    shift
    BOOTSTRAP_ITERATION=${1:-1}
    shift
elif [[ ${RUN_SCRIPT} == "run_bootstrap_latents.sh" ]]; then
    # Parse bootstrap iteration
    BOOTSTRAP_ITERATION=${1:-1}
    shift
elif [[ ${RUN_SCRIPT} == "run_finetune_eval.sh" ]]; then
    DATA_SETUP=${1:-"bootstrap_latent"}
    shift
    BOOTSTRAP_ITERATION=${1:-1}
    shift
else
    echo "Invalid RUN_SCRIPT: ${RUN_SCRIPT}"
    exit 1
fi


#####################################################
## Experiment setups
#####################################################

EXP_NAME_GROUP="bootstrap_fixed_data" 

### Warmstart setups ###
WARMSTART_EXP_NOTES="Warmstart training run for bootstrapping on fixed data, cosine lr schedule"
WARMSTART_DATA_SETUP="data.root_dir=${SYNTH_DATA_ROOT_DIR} data.sources='{\"finemath_4plus_final_500m_warmstart_split_with_gpt_latent_thoughts_latents\": 1.0}'"
WARMSTART_LATENT_TYPE_SWEEP=("random")
WARMSTART_LR_SWEEP=(${LR})
TRAIN_WARMSTART_TRIALS=(0 1 2)

WARMSTART_NUM_TOKENS=800000000  # ~0.8B total tokens, ~240M raw tokens,  ~4096 steps
WARMSTART_EXP_NAME_EXTRA_SUFFIX="_240m_raw"
WARMSTART_EXP_NAME=train_${EXP_NAME_GROUP}_warmstart
WARMSTART_EXTRA_KWARGS="checkpoint.eval.every=1000"   # this will override the default in the script


## Bootstrap setups ##
TRAIN_BOOTSTRAP_EXP_NAME=train_${EXP_NAME_GROUP}_bootstrap
TRAIN_BOOTSTRAP_EXP_NOTES="Training with self-bootstrap latents and compare with raw data baselines on FineMath, bootstrap on fixed data and restart training from scratch at each iteration"

BOOTSTRAP_SOURCE_DATASET_NAME=finemath_4plus_final_2b_separated_bootstrap_split
TRAIN_BOOTSTRAP_NUM_RAW_TOKENS=$(( 1920000000 * 2 ))
TRAIN_BOOTSTRAP_NUM_TOTAL_TOKENS=$(( 6500000000 * 2 ))  # ~2 epoch
MONTE_CARLO_SAMPLES=4

### Models to bootstrap from ###
INIT_BOOTSTRAP_MODEL_EXP_NAME=${WARMSTART_EXP_NAME}
INIT_BOOTSTRAP_MODEL_RUN_NAME="${WARMSTART_EXP_NAME}_latent=random_opt=${LR_SCHEDULER}_lr=${LR}${WARMSTART_EXP_NAME_EXTRA_SUFFIX}_trial_0"
INIT_BOOTSTRAP_MODEL_CKPT_STEPS=$(( (WARMSTART_NUM_TOKENS / (WARMSTART_GLOBAL_BATCH_SIZE * SEQ_LEN) ) ))
INIT_BOOTSTRAP_MODEL_LR=${LR}

INIT_BOOTSTRAP_MODEL_DIR=${CUR_DIR}/exp_logs/${INIT_BOOTSTRAP_MODEL_EXP_NAME}/${INIT_BOOTSTRAP_MODEL_RUN_NAME}
INIT_BOOTSTRAP_MODEL_PATH=${INIT_BOOTSTRAP_MODEL_DIR}/checkpoints/$(printf "%010d" ${INIT_BOOTSTRAP_MODEL_CKPT_STEPS})

BOOTSTRAP_MODEL_CKPT_STEPS=$(( (TRAIN_BOOTSTRAP_NUM_TOTAL_TOKENS / (BOOTSTRAP_GLOBAL_BATCH_SIZE * SEQ_LEN) ) )) 
if [[ ${BOOTSTRAP_ITERATION} -gt 1 ]]; then
    PREV_ITER=$((BOOTSTRAP_ITERATION - 1))
    CUR_ITER_BOOTSTRAP_MODEL_DIR=${CUR_DIR}/exp_logs/${TRAIN_BOOTSTRAP_EXP_NAME}/${TRAIN_BOOTSTRAP_EXP_NAME}_setup=bootstrap_latents_iter=${PREV_ITER}_mc=${MONTE_CARLO_SAMPLES}_scratch_trial_0
    CUR_ITER_BOOTSTRAP_MODEL_PATH=${CUR_ITER_BOOTSTRAP_MODEL_DIR}/checkpoints/$(printf "%010d" ${CUR_ITER_BOOTSTRAP_MODEL_CKPT_STEPS})
    CUR_ITER_BOOTSTRAP_MODEL_CKPT_STEPS=${BOOTSTRAP_MODEL_CKPT_STEPS}
else
    CUR_ITER_BOOTSTRAP_MODEL_DIR=${INIT_BOOTSTRAP_MODEL_DIR}
    CUR_ITER_BOOTSTRAP_MODEL_PATH=${INIT_BOOTSTRAP_MODEL_PATH}
    CUR_ITER_BOOTSTRAP_MODEL_CKPT_STEPS=${INIT_BOOTSTRAP_MODEL_CKPT_STEPS}
fi


### Generate bootstrap latents setups ###
GEN_NUM_GPU=1

GEN_DATA_CONFIG_NAME="bootstrap_latents_cpt"
GEN_DATA_SUFFIX="with_bootstrap_latents" # fixed the chunking to be text-specific

CHUNK_SEED=42
MAX_GEN_LEN=1024

GEN_NUM_PARALLEL=200  # set to be fixed for shard caching
GEN_SHARD_START=0
GEN_SHARD_END=${GEN_NUM_PARALLEL}
GEN_SHARD_EXCLUDE=()
GEN_EXTRA_KWARGS=""

GEN_DATASET_NAME=${BOOTSTRAP_SOURCE_DATASET_NAME}_${GEN_DATA_SUFFIX}_setup_${EXP_NAME_GROUP}_iter_${BOOTSTRAP_ITERATION}_mc_${MONTE_CARLO_SAMPLES}

### Train bootstrap setups ###

INIT_CKPT_PATH=null  # retrain from scratch
TRAIN_BOOTSTRAP_EXTRA_KWARGS="checkpoint.eval.every=2500 checkpoint.eval.keep=1"  
TRAIN_BOOTSTRAP_TRIALS=(0 1 2)

if [[ $DATA_SETUP == "bootstrap_latent" ]]; then
    SETUP_KEY="bootstrap_latents_iter=${BOOTSTRAP_ITERATION}_mc=${MONTE_CARLO_SAMPLES}_scratch"
    DATA_ROOT_DIR=${SYNTH_DATA_ROOT_DIR}
    DATASET_SUFFIX="_${GEN_DATA_SUFFIX}_setup_${EXP_NAME_GROUP}_iter_${BOOTSTRAP_ITERATION}_mc_${MONTE_CARLO_SAMPLES}"
    LATENT_TYPE="random"
    NUM_TOKENS=${TRAIN_BOOTSTRAP_NUM_TOTAL_TOKENS}
elif [[ $DATA_SETUP == "raw_token_match_baseline" ]]; then
    SETUP_KEY="raw_token_matched_scratch"
    DATA_ROOT_DIR=${RAW_DATA_ROOT_DIR}
    LATENT_TYPE=null
    NUM_TOKENS=${TRAIN_BOOTSTRAP_NUM_RAW_TOKENS}
    TRAIN_BOOTSTRAP_EXTRA_KWARGS="${TRAIN_BOOTSTRAP_EXTRA_KWARGS} checkpoint.eval.every=1250"
elif [[ $DATA_SETUP == "raw_flops_match_baseline" ]]; then
    SETUP_KEY="raw_flops_matched_scratch"
    DATA_ROOT_DIR=${RAW_DATA_ROOT_DIR}
    LATENT_TYPE=null
    NUM_TOKENS=${TRAIN_BOOTSTRAP_NUM_TOTAL_TOKENS}
    TRAIN_BOOTSTRAP_EXTRA_KWARGS="${TRAIN_BOOTSTRAP_EXTRA_KWARGS}"
elif [[ -n ${DATA_SETUP} ]]; then
    echo "Invalid DATA_SETUP: ${DATA_SETUP}"
    exit 1
fi

