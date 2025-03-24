#!/bin/bash


#!/bin/bash

# Set the running script
RUN_SCRIPT=$(basename $(realpath $0))

# Check for minimum required arguments
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <mode> <eval_dataset> <eval_setup_script> [additional_args...]"
    echo "  mode: 'cat', 'run', or 'launch'"
    echo "  eval_dataset: the dataset to evaluate on"
    echo "  eval_setup_script: the exp setup script (e.g., exp_setup_pretrain.sh)"
    echo "  additional_args: optional arguments passed to the setup script"
    exit 1
fi

# Import environment variables
EXP_SETUP_SCRIPT_DIR=$(dirname $(realpath $0))/exp_setups

# Parse the first two arguments
MODE=$1
EVAL_DATASET=$2
EVAL_SETUP_SCRIPT=$3

# Validate mode
if [[ "$MODE" != "cat" && "$MODE" != "run" && "$MODE" != "launch" ]]; then
    echo "Error: Mode must be 'cat', 'run', or 'launch'"
    exit 1
fi

# Shift twice to remove the first two arguments
shift 3

echo "Running '${RUN_SCRIPT}' script with setup '${EVAL_SETUP_SCRIPT}' in '${MODE}' mode"
if [[ $# -gt 0 ]]; then
    echo "Additional arguments to the setup script: $@"
fi

# Source the setup script with all remaining arguments
EVAL_SETUP_SCRIPT=${EXP_SETUP_SCRIPT_DIR}/$EVAL_SETUP_SCRIPT
source ${EVAL_SETUP_SCRIPT} "$@"

# Load sphinx config
source ${EXP_SETUP_SCRIPT_DIR}/slurm.sh


# Set environment variables

EVAL_GROUP_NAME="finetune_eval_on_${EVAL_DATASET}"
EVAL_CONFIG_FILE="./configs/finetune/${EVAL_DATASET}_finetune_vllm.yaml"

UPSTREAM_TRIAL=(0 1 2)
DOWNSTREAM_SEED=(42 43 44 45 46)

FINETUNE_NUM_GPU=1

PWD=$(pwd)


# Process all iterations, upstream trials, and downstream seeds
NUM_JOBS=0

for SEED in "${DOWNSTREAM_SEED[@]}"; do
    for TRIAL in "${UPSTREAM_TRIAL[@]}"; do
        echo ""
        echo ""
        echo "Processing setup: $SETUP_KEY, upstream trial: $TRIAL, downstream seed: $SEED"

        # Update model path in config to match the iteration and upstream trial
        EXP_RUN_DIR="${PWD}/exp_logs/${TRAIN_BOOTSTRAP_EXP_NAME}/${TRAIN_BOOTSTRAP_EXP_NAME}_setup=${SETUP_KEY}_trial_${TRIAL}"
        CKPT_BASE_DIR="${EXP_RUN_DIR}/checkpoints"

        if [[ -d "$CKPT_BASE_DIR" ]]; then
            # find the latest checkpoint
            CKPT_STEPS=$(ls -1 "$CKPT_BASE_DIR" | tail -n 1)
            CKPT_DIR="${CKPT_BASE_DIR}/${CKPT_STEPS}"
            CONSOLIDATED_DIR="${CKPT_DIR}/consolidated"
            MODEL_PATH="${CKPT_DIR}/hf"
        else
            echo "CKPT_BASE_DIR does not exist: $CKPT_BASE_DIR"
            continue
        fi

        # Base dump directory for this model
        BASE_DUMP_DIR="${EXP_RUN_DIR}/${EVAL_GROUP_NAME}/${CKPT_STEPS}"
        EXP_NAME="${EVAL_GROUP_NAME}_setup=${SETUP_KEY}_trial${TRIAL}_seed${SEED}"
        DUMP_DIR="${BASE_DUMP_DIR}/seed=${SEED}"

        if [[ -f "${DUMP_DIR}/eval.completed" ]]; then
            echo "Eval already completed for ${MODEL_PATH}"
            continue
        fi
    
        echo "  Model path: $MODEL_PATH"
        echo "  DUMP_DIR: $DUMP_DIR"

        mkdir -p $DUMP_DIR
        out_path="${DUMP_DIR}/finetune_eval.out"
        err_path="${DUMP_DIR}/finetune_eval.err"
        script_path="${DUMP_DIR}/finetune_eval.sh"

        EXTRA_ARGS="dump_dir=$DUMP_DIR model_name_or_path=$MODEL_PATH run_name=$EXP_NAME seed=$SEED"
        if [[ $MODEL_PATH == *"raw"* ]]; then
            # for raw baseline, use only special token delimiter without the latent prefix
            EXTRA_ARGS="${EXTRA_ARGS} only_special_token_delimiter=true"
        fi
        # EXTRA_ARGS="${EXTRA_ARGS} vllm_gpu_memory_utilization=0.3"

        convert_cmd="python setup/convert_consolidated_lingua_ckpt_to_hf.py --input_dir=${CONSOLIDATED_DIR} --output_dir=${MODEL_PATH} --tokenizer_path=${PWD}/exp_logs/pretrained_hf_ckpts/TinyLlama/TinyLlama_v1.1-embd-resized/tokenizer"
        train_cmd="python -u -m apps.main.finetune config=$EVAL_CONFIG_FILE $EXTRA_ARGS"

        echo "#!/bin/bash" > $script_path
        echo "export OMP_NUM_THREADS=1" >> $script_path
        echo "export VLLM_ATTENTION_BACKEND=XFORMERS" >> $script_path
        echo "$FINETUNE_EXTRAT_ENV_VARS" >> $script_path
        echo "$convert_cmd" >> $script_path
        echo "$train_cmd" >> $script_path
        chmod +x $script_path
        
        # Handle different modes
        if [[ $MODE == "cat" ]]; then
            cat $script_path
        elif [[ $MODE == "run" ]]; then
            . $script_path
        elif [[ $MODE == "launch" ]]; then
            NUM_CPUS=$((FINETUNE_NUM_GPU * 16))
            NUM_MEM=$((FINETUNE_NUM_GPU * 64))
            
            sbatch -o ${out_path} -e ${err_path} --account=${FINETUNE_ACCOUNT} -p ${FINETUNE_PARTITION} --gpus-per-node=$FINETUNE_NUM_GPU --mem=${NUM_MEM}G -c ${NUM_CPUS} --time=1:30:00 ${FINETUNE_EXTRA_FLAGS} ${script_path}
        fi

        ((NUM_JOBS++))
        
    done
done

echo "Launched $NUM_JOBS experiments"