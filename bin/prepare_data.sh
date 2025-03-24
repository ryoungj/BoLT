### FineMath #####

CUR_DIR=$(pwd)
BASE_DATA_DIR="./data/processed"
TRAIN_SPLIT_DATASET_NAME=finemath_4plus_final_fixed_train_split
NUM_SPLITS=20

# Get the SETUP argument, default to "all" if not provided
SETUP=${1:-"all"}
DOWNLOAD_PREPROCESSED_DATASETS=${2:-"false"}


if [[ "$SETUP" == "all" || "$SETUP" == "warmstart" ]]; then
    SPLIT_DATASET_NAME=finemath_4plus_final_500m_warmstart_split

    if [[ "$DOWNLOAD_PREPROCESSED_DATASETS" == "false" ]]; then
        python $CUR_DIR/apps/main/prepare_data.py --data-config-dir ${CUR_DIR}/configs/dataset --src-dataset-name ${TRAIN_SPLIT_DATASET_NAME} --tgt-dataset-name ${SPLIT_DATASET_NAME} \
            --mode subset_split --split-slice 0:1:${NUM_SPLITS} 
    elif [[ "$DOWNLOAD_PREPROCESSED_DATASETS" == "true" ]]; then
        python setup/download_cached_datasets.py --subsets ${SPLIT_DATASET_NAME} --dump-dir ${BASE_DATA_DIR}

        # Create dataset config files that may be used in dataset splitting
        cp configs/dataset/${SPLIT_DATASET_NAME}_cached.json configs/dataset/${SPLIT_DATASET_NAME}.json
    else
        echo "Invalid value for DOWNLOAD_PREPROCESSED_DATASETS: $DOWNLOAD_PREPROCESSED_DATASETS"
        exit 1
    fi
fi

if [[ "$SETUP" == "all" || "$SETUP" == "bootstrap_fixed_data" ]]; then
    SPLIT_DATASET_NAME=finemath_4plus_final_2b_separated_bootstrap_split

    if [[ "$DOWNLOAD_PREPROCESSED_DATASETS" == "false" ]]; then
        python $CUR_DIR/apps/main/prepare_data.py --data-config-dir ${CUR_DIR}/configs/dataset --src-dataset-name ${TRAIN_SPLIT_DATASET_NAME} --tgt-dataset-name ${SPLIT_DATASET_NAME} \
            --mode subset_split --split-slice $((NUM_SPLITS-7)):$((NUM_SPLITS-3)):${NUM_SPLITS} 
    elif [[ "$DOWNLOAD_PREPROCESSED_DATASETS" == "true" ]]; then
        python setup/download_cached_datasets.py --subsets ${SPLIT_DATASET_NAME} --dump-dir ${BASE_DATA_DIR}

        # Create dataset config files that may be used in dataset splitting
        cp configs/dataset/${SPLIT_DATASET_NAME}_cached.json configs/dataset/${SPLIT_DATASET_NAME}.json
    else
        echo "Invalid value for DOWNLOAD_PREPROCESSED_DATASETS: $DOWNLOAD_PREPROCESSED_DATASETS"
        exit 1
    fi
fi
