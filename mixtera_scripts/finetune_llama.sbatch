#!/bin/bash

#SBATCH --job-name=tinyllava_mixtera_finetune
#SBATCH --output=./finetune_tinyllama.log
#SBATCH --error=./finetune_tinyllama.err
#SBATCH --partition=normal
#SBATCH --environment=tinyllava
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --time=05:59:00

# Must have TinyLLaVA and Mixtera installed in the environment with all requirements
source /iopsstor/scratch/cscs/tkerimog/tinyllama_env/bin/activate

# Mixtera configuration
export MIXTERA_SERVER_ADDR=$(hostname)
export MIXTERA_SERVER_DIR=/iopsstor/scratch/cscs/tkerimog/tinyllava/mixtera_server
export MIXTERA_SERVER_PORT=12345
export MIXTERA_JOB_ID="mixtera_tinyllava_$(date +'%Y%m%d_%H%M%S')" 
export MIXTERA_CHUNK_SIZE=256 # 4 * global_batch_size as a heuristic
export MIXTERA_MIXTURE='{}' 

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

FINETUNE_DATA_PATH=/iopsstor/scratch/cscs/tkerimog/tinyllava/data/text_files/llava_v1_5_mix665k.json # finetune annotation file path
FINETUNE_IMAGE_PATH=/iopsstor/scratch/cscs/tkerimog/tinyllava/data # finetune image dir
LLM_VERSION=TinyLlama/TinyLlama-1.1B-Chat-v1.0 # llm path in huggingface
VT_VERSION=google/siglip-so400m-patch14-384 # vision tower path in huggingface
VT_VERSION2="" # if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu # connector type, other options are: qformer, resampler, etc
CONV_VERSION=llama #chat template, other options are: phi, llama, gemmma, etc
VERSION=mixtera # experiment name for recording different runnings
TRAIN_RECIPE=common # training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=2048
MAX_STEPS=543 # number of update steps for training i.e. batch_size * gradient_accumulation_steps samples will be processed per step
PRETRAINED_MODEL_PATH="/iopsstor/scratch/cscs/tkerimog/tinyllava/tiny-llava-TinyLlama-1.1B-Chat-v1.0-siglip-so400m-patch14-384-mixtera-pretrain-1"

cd /iopsstor/scratch/cscs/tkerimog/tinyllava/TinyLLaVA_Factory

bash scripts/train/finetune.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" \
"$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$MAX_STEPS" "$NUM_WORKERS" "$PRETRAINED_MODEL_PATH"