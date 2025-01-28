#!/bin/bash

MODEL_PATH="/iopsstor/scratch/cscs/tkerimog/tinyllava/tiny-llava-TinyLlama-1.1B-Chat-v1.0-siglip-so400m-patch14-384-base-finetune/checkpoint-5197"
MODEL_NAME="tiny-llava-TinyLlama-1.1B-Chat-v1.0-siglip-so400m-patch14-384-base-finetune"
EVAL_DIR="/iopsstor/scratch/cscs/tkerimog/tinyllava/data/eval"

python -m tinyllava.eval.model_vqa_science \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/scienceqa/llava_test_CQM-A.json \
    --image-folder $EVAL_DIR/scienceqa/images/test \
    --answers-file $EVAL_DIR/scienceqa/answers/$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode llama

python tinyllava/eval/eval_science_qa.py \
    --base-dir $EVAL_DIR/scienceqa \
    --result-file $EVAL_DIR/scienceqa/answers/$MODEL_NAME.jsonl \
    --output-file $EVAL_DIR/scienceqa/answers/"$MODEL_NAME"_output.jsonl \
    --output-result $EVAL_DIR/scienceqa/answers/"$MODEL_NAME"_result.json

