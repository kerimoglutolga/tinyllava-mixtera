#!/bin/bash

#SBATCH --account=a-a09
#SBATCH --job-name=mixtera_pretrain
#SBATCH --output=./pretrain.log
#SBATCH --error=./pretrain.err
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
export MIXTERA_CHUNK_SIZE=1024 # 4 * global_batch_size as a heuristic
export MIXTERA_MIXTURE='{}'  
export MIXTERA_MODE='pretrain'

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

DATA_PATH=/iopsstor/scratch/cscs/tkerimog/tinyllava/data/text_files/blip_laion_cc_sbu_558k.json
IMAGE_PATH=/iopsstor/scratch/cscs/tkerimog/tinyllava/data/llava/llava_pretrain/images #
LLM_VERSION=apple/OpenELM-270M-Instruct # llm path in huggingface
VT_VERSION=google/siglip-so400m-patch14-384 # vision tower path in huggingface
VT_VERSION2="" # if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu # connector type, other options are: qformer, resampler, etc
CONV_VERSION=qwen2_base #chat template, other options are: phi, llama, gemmma, etc
VERSION=mixtera-open_elm # experiment name for recording different runnings
TRAIN_RECIPE=common # training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=2048
MAX_STEPS=2179 # number of update steps for training i.e. batch_size * gradient_accumulation_steps samples will be processed per step
OUTPUT_DIR="/iopsstor/scratch/cscs/tkerimog/tinyllava"

cd /iopsstor/scratch/cscs/tkerimog/tinyllava/TinyLLaVA_Factory

bash scripts/train/pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" \
"$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$MAX_STEPS" "$NUM_WORKERS" "$OUTPUT_DIR"