#!/bin/bash

# Set the running script
RUN_SCRIPT=$(basename $(realpath $0))

# Check for minimum required arguments
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <mode> <eval_setup_script> [additional_args...]"
    echo "  mode: 'cat', 'run', or 'launch'"
    echo "  eval_setup_script: the exp setup script (e.g., exp_setup_pretrain.sh)"
    echo "  additional_args: optional arguments passed to the setup script"
    exit 1
fi

# Import environment variables
EXP_SETUP_SCRIPT_DIR=$(dirname $(realpath $0))/exp_setups

# Parse the first two arguments
MODE=$1
EVAL_SETUP_SCRIPT=$2

# Validate mode
if [[ "$MODE" != "cat" && "$MODE" != "run" && "$MODE" != "launch" ]]; then
    echo "Error: Mode must be 'cat', 'run', or 'launch'"
    exit 1
fi

# Shift twice to remove the first two arguments
shift 2

echo "Running '${RUN_SCRIPT}' script with setup '${EVAL_SETUP_SCRIPT}' in '${MODE}' mode"
if [[ $# -gt 0 ]]; then
    echo "Additional arguments to the setup script: $@"
fi

# Source the setup script with all remaining arguments
EVAL_SETUP_SCRIPT=${EXP_SETUP_SCRIPT_DIR}/$EVAL_SETUP_SCRIPT
source ${EVAL_SETUP_SCRIPT} "$@"

# Load sphinx config
source ${EXP_SETUP_SCRIPT_DIR}/slurm.sh

if [[ -z ${TRAIN_BOOTSTRAP_TRIALS} ]]; then
    TRAIN_BOOTSTRAP_TRIALS=(0)
fi

# Training setup
if [[ $((GLOBAL_BATCH_SIZE % (PER_GPU_BATCH_SIZE * TRAIN_NUM_GPU) )) -ne 0 ]]; then
    echo "GLOBAL_BATCH_SIZE must be divisible by PER_GPU_BATCH_SIZE * TRAIN_NUM_GPU"
    exit 1
fi
GRAD_ACC_STEPS=$(( (GLOBAL_BATCH_SIZE / (PER_GPU_BATCH_SIZE * TRAIN_NUM_GPU) ) ))

NUM_STEPS=$(( (NUM_TOKENS / (GLOBAL_BATCH_SIZE * SEQ_LEN) ) ))
echo "SETUP_KEY: ${SETUP_KEY} | DATASET_SUFFIX: ${DATASET_SUFFIX} | DATA_ROOT_DIR: ${DATA_ROOT_DIR} | LATENT_TYPE: ${LATENT_TYPE}"
echo "TRAIN_NUM_GPU: ${TRAIN_NUM_GPU}, NUM_TOKENS: ${NUM_TOKENS}, NUM_STEPS: ${NUM_STEPS}, GLOBAL_BATCH_SIZE: ${GLOBAL_BATCH_SIZE}, PER_GPU_BATCH_SIZE: ${PER_GPU_BATCH_SIZE}, SEQ_LEN: ${SEQ_LEN}, GRAD_ACC_STEPS: ${GRAD_ACC_STEPS}"

# Set up lr for bootstrap training
if [[ ${LR_SCHEDULER} == "wsd" ]] && [[ ${TRAIN_BOOTSTRAP_REWARMUP:-"true"} == "false" ]]; then 
    LR_WARMUP=0
    LOAD_OPT_FROM_INIT_CKPT=true
else
    LR_WARMUP=${WARMUP}
    LOAD_OPT_FROM_INIT_CKPT=false
fi

# Launch jobs
NUM_JOBS=0
for TRIAL in ${TRAIN_BOOTSTRAP_TRIALS[@]}; do
    TRIAL_SEED=$((TRIAL + 42))
    echo "TRIAL: ${TRIAL}, SEED: ${TRIAL_SEED}"

    RUN_NAME=${TRAIN_BOOTSTRAP_EXP_NAME}_setup=${SETUP_KEY}
    RUN_NAME="${RUN_NAME}_trial_${TRIAL}"

    SLURM_DIR=${BASE_OUT_DIR}/${TRAIN_BOOTSTRAP_EXP_NAME}/slurm
    mkdir -p ${SLURM_DIR}
    script_path="${SLURM_DIR}/${RUN_NAME}.sh"
    out_path="${SLURM_DIR}/${RUN_NAME}.out"

    DUMP_DIR=${BASE_OUT_DIR}/${TRAIN_BOOTSTRAP_EXP_NAME}/${RUN_NAME}
    mkdir -p ${DUMP_DIR}

    DATASET_NAME=${BOOTSTRAP_SOURCE_DATASET_NAME}${DATASET_SUFFIX}
    
    # Create run script
    echo "#!/bin/bash" > $script_path
    PORT=$((29500 + RANDOM % 500))

    cmd="torchrun --nproc-per-node ${TRAIN_NUM_GPU} --master-port ${PORT} -m apps.main.train config=configs/train/${TRAIN_CONFIG_NAME}.yaml"
    cmd="${cmd} name=${RUN_NAME} dump_dir=${DUMP_DIR}"
    cmd="${cmd} logging.wandb.name=${RUN_NAME} logging.wandb.id=${RUN_NAME} logging.wandb.dir=${DUMP_DIR}"
    cmd="${cmd} logging.wandb.tags=[${TRAIN_BOOTSTRAP_EXP_NAME}] logging.wandb.notes='${TRAIN_BOOTSTRAP_EXP_NOTES}'" 
    cmd="${cmd} steps=${NUM_STEPS} grad_acc_steps=${GRAD_ACC_STEPS}"
    cmd="${cmd} data.seq_len=${SEQ_LEN} data.batch_size=${PER_GPU_BATCH_SIZE}"
    cmd="${cmd} distributed.dp_shard=${TRAIN_NUM_GPU}"
    cmd="${cmd} data.root_dir=${DATA_ROOT_DIR}"
    cmd="${cmd} data.sources='{\"${DATASET_NAME}\": 1.0}'"
    cmd="${cmd} data.latent_type=${LATENT_TYPE}"
    cmd="${cmd} async_eval_gpus=1"
    if [[ ${INIT_CKPT_PATH} != "null" ]]; then
        cmd="${cmd} checkpoint.init_ckpt_path=${INIT_CKPT_PATH}"
    fi
    cmd="${cmd} checkpoint.dump.keep=1"
    cmd="${cmd} checkpoint.load_optim_from_init_ckpt=${LOAD_OPT_FROM_INIT_CKPT}"
    cmd="${cmd} optim.scheduler=${LR_SCHEDULER} optim.lr=${INIT_BOOTSTRAP_MODEL_LR} optim.warmup=${LR_WARMUP}"
    if [[ -n ${TRAIN_BOOTSTRAP_WSD_LR_DECAY} && ${LR_SCHEDULER} == "wsd" ]]; then
        cmd="${cmd} optim.decay_fraction=${TRAIN_BOOTSTRAP_WSD_LR_DECAY}"
    fi
    if [[ -n ${LR_MIN_RATIO} ]]; then
        cmd="${cmd} optim.lr_min_ratio=${LR_MIN_RATIO}"
    fi
    cmd="${cmd} data.prefetch_size=512"   # should not exceed the dataset size
    cmd="${cmd} seed=${TRIAL_SEED} model.seed=${TRIAL_SEED} data.seed=${TRIAL_SEED}"
    cmd="${cmd} ${EVAL_CONFIG_ARGS} ${TRAIN_BOOTSTRAP_EXTRA_KWARGS}"
    echo "${cmd}" >> $script_path

    # Handle different modes
    if [[ $MODE == "cat" ]]; then
        cat $script_path
    elif [[ $MODE == "run" ]]; then
        . $script_path
    elif [[ $MODE == "launch" ]]; then
        NUM_CPUS=$((TRAIN_NUM_GPU * 16))
        NUM_MEM=$((TRAIN_NUM_GPU * 64))
        
        sbatch -o ${out_path} --account=${TRAIN_ACCOUNT} -p ${TRAIN_PARTITION} --gpus-per-node=$TRAIN_NUM_GPU --mem=${NUM_MEM}G -c ${NUM_CPUS} --time=20:00:00 ${TRAIN_EXTRA_FLAGS} ${script_path}
    fi
    
    NUM_JOBS=$(($NUM_JOBS + 1))

    echo "----------------------------------------"
done

echo "Submitted $NUM_JOBS jobs"
