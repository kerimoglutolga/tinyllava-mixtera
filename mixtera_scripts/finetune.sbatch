#!/bin/bash

#SBATCH --account=a-a09
#SBATCH --job-name=mixtera_finetune
#SBATCH --output=./finetune.log
#SBATCH --error=./finetune.err
#SBATCH --partition=normal
#SBATCH --environment=tinyllava
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --time=10:00:00

# Must have TinyLLaVA and Mixtera installed in the environment with all requirements
source /iopsstor/scratch/cscs/tkerimog/tinyllama_env/bin/activate

# Mixtera configuration
export MIXTERA_SERVER_ADDR=$(hostname)
export MIXTERA_SERVER_DIR=/iopsstor/scratch/cscs/tkerimog/tinyllava/mixtera_server
export MIXTERA_SERVER_PORT=12345
export MIXTERA_JOB_ID="mixtera_tinyllava_$(date +'%Y%m%d_%H%M%S')" 
export MIXTERA_CHUNK_SIZE=512 # 4 * global_batch_size as a heuristic
export MIXTERA_MIXTURE='{}'  
export MIXTERA_MODE='finetune'

# Start Mixtera server
/iopsstor/scratch/cscs/tkerimog/tinyllama_env/bin/python -u -m mixtera.network.server.entrypoint \
    $MIXTERA_SERVER_DIR \
    --host $MIXTERA_SERVER_ADDR\
    --port $MIXTERA_SERVER_PORT &

sleep 5

# TinyLLaVA configuration
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/tkerimog/tinyllava/.cache
export HF_HOME=/iopsstor/scratch/cscs/tkerimog/tinyllava/.cache
export NUM_WORKERS=8

FINETUNE_DATA_PATH=/iopsstor/scratch/cscs/tkerimog/tinyllava/data/text_files/llava_v1_5_mix665k_cleaned.json # finetune annotation file path
FINETUNE_IMAGE_PATH=/iopsstor/scratch/cscs/tkerimog/tinyllava/data # finetune image dir
LLM_VERSION=apple/OpenELM-270M-Instruct
VT_VERSION=google/siglip-so400m-patch14-384
VT_VERSION2=""
CN_VERSION=mlp2x_gelu
CONV_VERSION=llama
VERSION=elm_base
TRAIN_RECIPE=common
MODEL_MAX_LENGTH=2048
MAX_STEPS=5194 # number of update steps for training i.e. batch_size * gradient_accumulation_steps samples will be processed per step
PRETRAINED_MODEL_PATH="/iopsstor/scratch/cscs/tkerimog/tinyllava/TinyLLaVA_Factory_new/outputs/tiny-llava-OpenELM-270M-Instruct-siglip-so400m-patch14-384-elm_base-pretrain"
OUTPUT_DIR="/iopsstor/scratch/cscs/tkerimog/tinyllava/"

cd /iopsstor/scratch/cscs/tkerimog/tinyllava/TinyLLaVA_Factory

bash scripts/train/openelm/finetune_openelm.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" \
"$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$MAX_STEPS" "$NUM_WORKERS" "$PRETRAINED_MODEL_PATH" "$OUTPUT_DIR"