#!/bin/bash

# Set the running script
RUN_SCRIPT=$(basename $(realpath $0))

# Check for minimum required arguments
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <mode> <eval_setup_script> [additional_args...]"
    echo "  mode: 'download', 'cat,' 'run,' 'launch,' or 'merge'"
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
if [[ "$MODE" != "download" && "$MODE" != "cat" && "$MODE" != "run" && "$MODE" != "launch" && "$MODE" != "merge" ]]; then
    echo "Error: Mode must be 'download', 'cat', 'run', 'launch', or 'merge'"
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


if [[ "$MODE" == "download" ]]; then
    echo "Downloading datasets from Hugging Face..."
    python setup/download_cached_datasets.py --subsets ${GEN_DATASET_NAME} --dump-dir ${SYNTH_DATA_ROOT_DIR}
else
    TARGET_DATASET_BASE_DIR=${SYNTH_DATA_ROOT_DIR}/${GEN_DATASET_NAME}
    SLURM_DIR=${TARGET_DATASET_BASE_DIR}/slurm

    mkdir -p ${TARGET_DATASET_BASE_DIR}
    mkdir -p ${SLURM_DIR}

    # Format the base command
    base_cmd="python apps/main/gen_latents.py config=configs/gen/${GEN_DATA_CONFIG_NAME}.yaml"
    base_cmd="${base_cmd} src_dataset_name=${BOOTSTRAP_SOURCE_DATASET_NAME}"
    base_cmd="${base_cmd} tgt_dataset_base_dir=${SYNTH_DATA_ROOT_DIR}"
    base_cmd="${base_cmd} tgt_dataset_name=${GEN_DATASET_NAME}"
    base_cmd="${base_cmd} generation_method=self_bootstrap"
    if [[ -n ${BOOTSTRAP_DATA_GEN_MODEL_DIR} && -n ${BOOTSTRAP_DATA_GEN_MODEL_CKPT_STEPS} ]]; then
        base_cmd="${base_cmd} generation_kwargs.model_dir=${BOOTSTRAP_DATA_GEN_MODEL_DIR}"
        base_cmd="${base_cmd} generation_kwargs.step=${BOOTSTRAP_DATA_GEN_MODEL_CKPT_STEPS}"
    else
        base_cmd="${base_cmd} generation_kwargs.model_dir=${CUR_ITER_BOOTSTRAP_MODEL_DIR}"
        base_cmd="${base_cmd} generation_kwargs.step=${CUR_ITER_BOOTSTRAP_MODEL_CKPT_STEPS}"
    fi
    base_cmd="${base_cmd} generation_kwargs.use_vllm=true"
    base_cmd="${base_cmd} generation_kwargs.generator.max_gen_len=${MAX_GEN_LEN}"
    if [[ $MONTE_CARLO_SAMPLES -gt 1 ]]; then
        base_cmd="${base_cmd} generation_kwargs.num_total_samples=${MONTE_CARLO_SAMPLES}"
        base_cmd="${base_cmd} generation_kwargs.apply_monte_carlo=true"
    fi
    base_cmd="${base_cmd} chunk_seed=${CHUNK_SEED}"
    base_cmd="${base_cmd} overwrite=true"
    base_cmd="${base_cmd} ${GEN_EXTRA_KWARGS}"


    CACHE_DIR=${SYNTH_DATA_ROOT_DIR}/${GEN_DATASET_NAME}/cache

    # Step 1: Generate latents for each slice on different jobs
    if [[ $MODE != "merge" ]]; then
            NUM_JOBS=0
            for NUM_SLICE in $(seq ${GEN_SHARD_START} $((GEN_SHARD_END - 1))); do
            if [[ -n "${GEN_SHARD_EXCLUDE[@]}" && " ${GEN_SHARD_EXCLUDE[*]} " == *" ${NUM_SLICE} "*  ]]; then
                echo "Skipping shard ${NUM_SLICE} because it is in the exclude list"
                continue
            fi

            CACHE_FILE=${CACHE_DIR}/result_shard00_slice$(printf "%02d" ${NUM_SLICE})_of_${GEN_NUM_PARALLEL}.jsonl

            if [[ -f ${CACHE_FILE} ]]; then
                echo "Skipping shard ${NUM_SLICE} because it is already finished"
                continue
            fi

            echo "Submitting shard ${NUM_SLICE} of ${GEN_NUM_PARALLEL}"

            JOB_NAME=slice_${NUM_SLICE}_of_${GEN_NUM_PARALLEL}
            

            script_path="${SLURM_DIR}/${JOB_NAME}.sh"
            out_path="${SLURM_DIR}/${JOB_NAME}.out"

            # Create run script
            echo "#!/bin/bash" > $script_path
            cat << 'EOF' >> $script_path
# Check hostname and adjust max seq nums if needed
current_hostname=$(hostname)
total_gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
if [[ $TRAIN_CONFIG_NAME == "llama_1B_cpt" && $total_gpu_mem -lt 60000 ]]; then   # 40GiB A100 & 49GiB A6000
    extra_args=" generation_kwargs.generator.max_num_seqs=80"
elif [[ $total_gpu_mem -lt 30000 ]]; then   # 24 GiB A5000 & RTX3090
    extra_args=" generation_kwargs.generator.gpu_memory_utilization=0.85" 
else
    extra_args=""
fi
EOF

            cmd="${base_cmd} data_slice=\\\"${NUM_SLICE}:$((NUM_SLICE + 1)):${GEN_NUM_PARALLEL}\\\""
            cmd="${cmd} merge_shards=false"   # do not merge shards for now
            echo "${cmd}\${extra_args}" >> $script_path

            # Handle different modes
            if [[ $MODE == "cat" ]]; then
                cat $script_path
                echo ""
            elif [[ $MODE == "run" ]]; then
                . $script_path
            elif [[ $MODE == "launch" ]]; then
                MEM=32G
                NUM_CPUS=8
                sbatch -o ${out_path} -p ${GEN_PARTITION} --account=${GEN_ACCOUNT} --gpus-per-node=$GEN_NUM_GPU --mem=${MEM} -c ${NUM_CPUS} ${GEN_EXTRA_FLAGS} ${script_path}
            fi
            
            NUM_JOBS=$(($NUM_JOBS + 1))
        done

        echo "Submitted $NUM_JOBS jobs"
    fi


    if [[ $MODE == "merge" ]]; then
        # Step 2: Merge the generated data with the a small trick: the merge can be done in a single process with the same number of data slices
        merge_cmd="${base_cmd} num_processes=${GEN_NUM_PARALLEL}"

        echo $merge_cmd
        ${merge_cmd}
    fi
fi