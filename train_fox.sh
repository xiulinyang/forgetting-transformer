#!/bin/bash

OUTPUT_DIR="./forgetting_gate/"  # You can set this to any other path
WANDB_DIR="./output/wandb"  # You can set this to any other path
mkdir -p $OUTPUT_DIR
mkdir -p $WANDB_DIR
fabric run train.py \
    --devices 4 \
    --num-nodes 1 \
    --node-rank 0 \
    --main-address localhost \
    --main-port 1234 \
    +experiment/pile/forgetting_transformer=forgetting_gate \
    seed=42 \
    exp=forgetting_gate_exp \
    tag=ft_l2_h256_h4_seq2048_adam \
    output_dir=$OUTPUT_DIR \
    data_dir=$DATA_DIR \
    wandb.log_dir=$WANDB_DIR \
    wandb.mode=offline \
    resume=true