#!/bin/bash

MODEL_PATH="/iopsstor/scratch/cscs/tkerimog/tinyllava/tiny-llava-TinyLlama-1.1B-Chat-v1.0-siglip-so400m-patch14-384-base-finetune/checkpoint-5197"
MODEL_NAME="tiny-llava-TinyLlama-1.1B-Chat-v1.0-siglip-so400m-patch14-384-base-finetune"
EVAL_DIR="/iopsstor/scratch/cscs/tkerimog/tinyllava/data/eval"

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $EVAL_DIR/textvqa/train_images \
    --answers-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode llama

python -m tinyllava.eval.eval_textvqa \
    --annotation-file $EVAL_DIR/textvqa/TextVQA_0.5.1_val.json \
    --result-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl

