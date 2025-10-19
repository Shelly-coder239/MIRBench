#!/bin/bash
# ==============================================
#  Curriculum Learning Fine-tuning with ms-swift
#  (samples are pre-sorted by difficulty: easy → hard)
#  Reference: https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html
# ==============================================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8

swift sft \
    --model ../llava-onevision-qwen2-7b-ov-hf/ \
    --train_type full \
    --dataset ../total_sft.json \
    --split_dataset_ratio 0 \                
    --shuffle False \                  
    --dataset_num_proc 8 \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 4 \
    --save_steps 5000 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 32768 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --deepspeed zero3 \
    --report_to tensorboard \
    --bf16 True                                     
