#!/bin/bash

BASE_DATA_DIR="./data/processed"

DOWNLOAD_PREPROCESSED_DATASETS=${1:-false}

VAL_SPLIT_DATASET_NAME=finemath_4plus_final_fixed_val_split
TRAIN_SPLIT_DATASET_NAME=finemath_4plus_final_fixed_train_split


if [[ "$DOWNLOAD_PREPROCESSED_DATASETS" == "false" ]]; then
    echo "Downloading and pre-processing dataset"

    # Download and preprocess dataset
    python setup/download_hf_data.py --dataset finemath-4plus --datadir "$BASE_DATA_DIR"

    # Prepare shuffled dataset
    python apps/main/prepare_data.py --data-config-dir configs/dataset --src-dataset-name finemath_4plus_raw --tgt-dataset-name finemath_4plus_final_shuffled \
        --num-chunks 1 --num-val-samples 0 --memory 128 --mode final_shuffled --tmp-dir ./data/tmp --tgt-dataset-basedir ${BASE_DATA_DIR}
    
    # Split the train/val dataset locally
    NUM_SPLITS=1000

    # Split the val dataset
    python apps/main/prepare_data.py --data-config-dir configs/dataset --src-dataset-name finemath_4plus_final_shuffled --tgt-dataset-name ${VAL_SPLIT_DATASET_NAME} \
        --mode subset_split --split-slice 0:1:${NUM_SPLITS}

    # Split the train dataset
    python apps/main/prepare_data.py --data-config-dir configs/dataset --src-dataset-name finemath_4plus_final_shuffled --tgt-dataset-name ${TRAIN_SPLIT_DATASET_NAME} \
        --mode subset_split --split-slice 1:${NUM_SPLITS}:${NUM_SPLITS} 

    # Rename the val dataset to the correct name
    mv ${BASE_DATA_DIR}/${VAL_SPLIT_DATASET_NAME}/finemath_4plus_final_shuffled.chunk.00.jsonl ${BASE_DATA_DIR}/${VAL_SPLIT_DATASET_NAME}/finemath_4plus_final_shuffled.val.jsonl
elif [[ "$DOWNLOAD_PREPROCESSED_DATASETS" == "true" ]]; then
    echo "Downloading pre-processed val dataset from HuggingFace"

    # Download the pre-processed dataset from HuggingFace
    # Only download the val set as the specific training set will be downloaded later later
    python setup/download_cached_datasets.py --subsets ${VAL_SPLIT_DATASET_NAME} --dump-dir ${BASE_DATA_DIR}

    # Rename the val dataset to the correct name
    mv ${BASE_DATA_DIR}/${VAL_SPLIT_DATASET_NAME}/${VAL_SPLIT_DATASET_NAME}.chunk.00.jsonl ${BASE_DATA_DIR}/${VAL_SPLIT_DATASET_NAME}/finemath_4plus_final_shuffled.val.jsonl

    # Create dataset config files that may be used in dataset splitting
    cp configs/dataset/${VAL_SPLIT_DATASET_NAME}_cached.json configs/dataset/${VAL_SPLIT_DATASET_NAME}.json
else
    echo "Invalid value for DOWNLOAD_PREPROCESSED_DATASETS: $DOWNLOAD_PREPROCESSED_DATASETS"
    exit 1
fi