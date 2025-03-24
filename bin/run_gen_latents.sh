SOURCE_DATASET_NAME=finemath_4plus_final_500m_warmstart_split
TARGET_DATASET_BASE_DIR=data/synthetic/
NUM_PROCESSES=20

SYNTH_METHOD=${1}
DOWNLOAD_PREPROCESSED_DATASETS=${2:-"false"}

if [[ "${SYNTH_METHOD}" == "latent_thoughts" ]]; then
    ## Latent Thoughts
    CONFIG_NAME=gpt_latent_thoughts
    TARGET_DATASET_NAME=${SOURCE_DATASET_NAME}_with_${CONFIG_NAME}_latents
    NUM_REPEATS=1
elif [[ "${SYNTH_METHOD}" == "wrap_baseline" ]]; then
    ## WRAP Baseline
    CONFIG_NAME=wrap_baseline
    TARGET_DATASET_NAME=${SOURCE_DATASET_NAME}_with_${CONFIG_NAME}_latents
    NUM_REPEATS=5
elif [[ "${SYNTH_METHOD}" == "wrap_cot" ]]; then
    ## WRAP variant with CoT
    CONFIG_NAME=wrap_cot
    TARGET_DATASET_NAME=${SOURCE_DATASET_NAME}_with_${CONFIG_NAME}_latents
    NUM_REPEATS=4
fi

mkdir -p ${TARGET_DATASET_BASE_DIR}

if [[ "$DOWNLOAD_PREPROCESSED_DATASETS" == "false" ]]; then
    python apps/main/gen_latents.py config=configs/gen/${CONFIG_NAME}.yaml \
        src_dataset_name=${SOURCE_DATASET_NAME} \
        tgt_dataset_base_dir=${TARGET_DATASET_BASE_DIR} \
        tgt_dataset_name=${TARGET_DATASET_NAME} \
        num_processes=${NUM_PROCESSES} \
        num_repeats=${NUM_REPEATS}
elif [[ "$DOWNLOAD_PREPROCESSED_DATASETS" == "true" ]]; then
     python setup/download_cached_datasets.py --subsets ${TARGET_DATASET_NAME} --dump-dir ${TARGET_DATASET_BASE_DIR}
else
    echo "Invalid value for DOWNLOAD_PREPROCESSED_DATASETS: $DOWNLOAD_PREPROCESSED_DATASETS"
    exit 1
fi