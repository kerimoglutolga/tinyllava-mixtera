#!/bin/bash

# Assign the arguments to variables
DATA_PATH="$1"
IMAGE_PATH="$2"
LLM_VERSION="$3"
VT_VERSION="$4"
VT_VERSION2="$5"
CN_VERSION="$6"
VERSION="$7"
TRAIN_RECIPE="$8"
MODEL_MAX_LENGTH="$9"
MAX_STEPS="${10}"
NUM_WORKERS="${11}"
OUTPUT_DIR="${12}"

VT_VARIANT="${VT_VERSION#*/}"
LLM_VARIANT="${LLM_VERSION#*/}"

deepspeed --include localhost:0 --master_port 29501 tinyllava/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --data_path  $DATA_PATH\
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --conv_version pretrain \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $VT_VERSION \
    --vision_tower2 "$VT_VERSION2" \
    --connector_type $CN_VERSION \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation flash_attention_2 \
    --fp16 True \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_llm frozen \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --output_dir /iopsstor/scratch/cscs/tkerimog/tinyllava/tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2179 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers $NUM_WORKERS \
    --lazy_preprocess True \
    --report_to wandb \
    --tokenizer_use_fast False \
    --run_name tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-pretrain \
    --lr_scheduler_type "cosine" \
    --max_steps $MAX_STEPS \
    --dispatch_batches True \
    --split_batches True \
