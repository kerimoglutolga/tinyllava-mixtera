#!/bin/bash

#SBATCH --account=a-a09
#SBATCH --job-name=eval_image_question_answering
#SBATCH --output=./eval_image_question_answering_mixtera-llava_balanced.log
#SBATCH --error=./eval_image_question_answering_mixtera-llava_balanced.err
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

MODEL_PATH="tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"
MODEL_NAME="TinyLLaVA-Phi-2-SigLIP-3.1B"
CONV_MODE=phi

cd /iopsstor/scratch/cscs/tkerimog/tinyllava/TinyLLaVA_Factory

CUDA_VISIBLE_DEVICES=0 bash scripts/eval/sqa.sh "$MODEL_PATH" "$MODEL_NAME" "$CONV_MODE" 
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/textvqa.sh "$MODEL_PATH" "$MODEL_NAME" "$CONV_MODE" 
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/vqav2.sh "$MODEL_PATH" "$MODEL_NAME" "$CONV_MODE" 
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/gqa.sh "$MODEL_PATH" "$MODEL_NAME" "$CONV_MODE" 
