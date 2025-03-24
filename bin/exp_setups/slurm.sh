# Modify the following sphinx setups to run your experiments!

## Training: typically a node with 8 H200 GPUs
TRAIN_ACCOUNT="miso"
TRAIN_PARTITION="miso"
TRAIN_EXTRA_FLAGS=""


## Eval: can be distributed across multiple nodes with standard gpus like A100 and A6000
EVAL_ACCOUNT="nlp"
EVAL_PARTITION="jag-standard"
EVAL_EXCLUDE="jagupard19,jagupard20,jagupard26,jagupard27,jagupard28,jagupard29,jagupard30,jagupard31"
EVAL_EXTRA_FLAGS=""
EVAL_CONFIG_ARGS="slurm.account=${EVAL_ACCOUNT} slurm.partition=${EVAL_PARTITION} slurm.exclude=${EVAL_EXCLUDE} ${EVAL_EXTRA_FLAGS}"


## Generation: can be distributed over multiple nodes with idle gpus like A100, A6000, A5000, RTX3090, etc.
GEN_ACCOUNT="nlp"
GEN_PARTITION="sc-loprio"
GEN_EXTRA_FLAGS="--exclude=jagupard[19-20,26-27],iliad[1-4],john[1-15,17]"


## Finetuning: typically 1 A100/H100 GPU
FINETUNE_ACCOUNT="nlp"
FINETUNE_PARTITION="sphinx"
FINETUNE_EXTRAT_ENV_VARS="export LD_LIBRARY_PATH=/nlp/scr/yjruan/miniconda3/envs/latent-lingua/lib:\$LD_LIBRARY_PATH
export WANDB_PROJECT="finetune-eval"
export WANDB_DIR=/juice5/scr5/yjruan/wandb
export WANDB_CACHE_DIR=/juice5/scr5/yjruan/wandb_cache
export TRITON_CACHE_DIR=/juice5/scr5/yjruan/triton_cache
export TMPDIR=/juice5/scr5/yjruan/tmp
export TORCH_COMPILE_DIR=/juice5/scr5/yjruan/compile_cache
export TORCHINDUCTOR_CACHE_DIR=/juice5/scr5/yjruan/inductor_cache"
FINETUNE_EXTRA_FLAGS="--exclude=sphinx[1-2]"
