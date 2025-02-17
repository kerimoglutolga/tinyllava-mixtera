#!/bin/bash

#SBATCH --account=a-a09
#SBATCH --job-name=mixtera_finetune
#SBATCH --output=./finetune_base.log
#SBATCH --error=./finetune_base.err
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

# TinyLLaVA configuration
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/tkerimog/tinyllava/.cache
export HF_HOME=/iopsstor/scratch/cscs/tkerimog/tinyllava/.cache
export NUM_WORKERS=8

DATA_PATH=/iopsstor/scratch/cscs/tkerimog/tinyllava/data/text_files/blip_laion_cc_sbu_558k.json #pretrain annotation file path
FINETUNE_DATA_PATH=/iopsstor/scratch/cscs/tkerimog/tinyllava/data/text_files/llava_v1_5_mix665k_cleaned.json #finetune annotation file path
IMAGE_PATH=/iopsstor/scratch/cscs/tkerimog/tinyllava/data/llava/llava_pretrain/images #pretrain image dir
FINETUNE_IMAGE_PATH=/iopsstor/scratch/cscs/tkerimog/tinyllava/data #finetune image dir

LLM_VERSION=microsoft/phi-2 # llm path in huggingface
VT_VERSION=google/siglip-so400m-patch14-384 #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=phi #chat template, other options are: phi, llama, gemmma, etc
VERSION=base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=3072 #max model length for llm
PRETRAINED_MODEL_PATH=/iopsstor/scratch/cscs/tkerimog/tinyllava/tiny-llava-phi-2-siglip-so400m-patch14-384-base-pretrain
OUTPUT_DIR=/iopsstor/scratch/cscs/tkerimog/tinyllava/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune

cd /iopsstor/scratch/cscs/tkerimog/tinyllava/TinyLLaVA_Factory

export HF_TOKEN=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
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN

#bash scripts/train/pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
bash scripts/train/finetune.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$PRETRAINED_MODEL_PATH" "$OUTPUT_DIR"

