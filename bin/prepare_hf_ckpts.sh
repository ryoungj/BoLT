#!/bin/bash

DTYPE="float32"
BASE_CKPT_DIR="./exp_logs/pretrained_hf_ckpts"
MODELS=(TinyLlama/TinyLlama_v1.1)

for MODEL in "${MODELS[@]}"; do
    python setup/download_hf_ckpt.py \
        --model "$MODEL" \
        --output "$BASE_CKPT_DIR/$MODEL-embd-resized" \
        --dtype "$DTYPE" \
        --add-special-tokens
done
