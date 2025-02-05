#!/bin/bash

#SBATCH --job-name=eval_benchmark_toolkit
#SBATCH --output=./eval_benchmark_toolkit_mixtera-coco05_gqa01_ocrvqa04-finetune.log
#SBATCH --error=./eval_benchmark_toolkit_mixtera-coco05_gqa01_ocrvqa04-finetune.err
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

# TinyLLaVA configuration
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/tkerimog/tinyllava/.cache
export HF_HOME=/iopsstor/scratch/cscs/tkerimog/tinyllava/.cache
export NUM_WORKERS=8

MODEL_PATH="/iopsstor/scratch/cscs/tkerimog/tinyllava/tiny-llava-TinyLlama-1.1B-Chat-v1.0-siglip-so400m-patch14-384-mixtera-coco05_gqa01_ocrvqa04-finetune/checkpoint-1386"
MODEL_NAME="tiny-llava-TinyLlama-1.1B-Chat-v1.0-siglip-so400m-patch14-384-mixtera-coco05_gqa01_ocrvqa04-finetune"
CONV_MODE=llama

cd /iopsstor/scratch/cscs/tkerimog/tinyllava/TinyLLaVA_Factory

CUDA_VISIBLE_DEVICES=0 bash scripts/eval/pope.sh "$MODEL_PATH" "$MODEL_NAME" "$CONV_MODE" 
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mme.sh "$MODEL_PATH" "$MODEL_NAME" "$CONV_MODE" 
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmvet.sh "$MODEL_PATH" "$MODEL_NAME" "$CONV_MODE" 
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmmu.sh "$MODEL_PATH" "$MODEL_NAME" "$CONV_MODE" 
