#!/bin/bash

OUTPUT_DIR="./forgetting_gate/"  # You can set this to any other path
WANDB_DIR="./output/wandb"  # You can set this to any other path
mkdir -p $OUTPUT_DIR
mkdir -p $WANDB_DIR

torchrun --nproc_per_node=4 --master_port=1234 train.py \
  +experiment/pile/forgetting_transformer=forgetting_gate \
  fabric.devices=4 \
  fabric.precision=16-mixed \
  seed=42 \
  exp=forgetting_gate_exp \
  tag=ft_l2_h256_h4_seq2048_adam \
  output_dir=./forgetting_gate/ \
  data_dir=$DATA_DIR \
  wandb.log_dir=./output/wandb \
  wandb.mode=online \
  resume=false


