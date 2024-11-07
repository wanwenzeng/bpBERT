#!/bin/bash

# DDP running
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 run_tfbs.py \
 --hdf5_file **your .h5 file path** \
 --bert_model **your pretrained model dir** \
 --output_dir models/tfbs/ \
 --n_tasks 40 \
 --max_seq_length 1002 \
 --do_train \
 --train_batch_size 1 \
 --eval_batch_size 3 \
 --learning_rate 5e-5 \
 --num_train_epochs 30 \
 --warmup_proportion 0 \
 --gradient_accumulation_steps 1 \
 --comment tfbs \
 --ngram_list kmer45678_0.1_GRCh38.txt \
 --kmer_len_list 4_5_6_7_8 \
 --max_ngram_in_sequence 1500 \
 --early_stop_patience 5 \
 --ckpt_steps 2000 \
 --multi_gpu