#!/bin/bash

OUTPUT_DIR="./alibi/"  # You can set this to any other path
WANDB_DIR="./output/wandb"  # You can set this to any other path
mkdir -p $OUTPUT_DIR
mkdir -p $WANDB_DIR

torchrun --nproc_per_node=1 --master_port=29500 train.py \
  +experiment/pile/forgetting_transformer=alibi \
  fabric.devices=1 \
  fabric.precision=16-mixed \
  seed=42 \
  exp=alibi_exp \
  tag=ft_l2_h256_h4_seq2048_adam \
  output_dir=./alibi/ \
  wandb.log_dir=./output/wandb \
  wandb.mode=online \
  resume=false


