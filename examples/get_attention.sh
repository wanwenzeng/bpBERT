#!/bin/bash

while IFS= read -r exapriment_param; do
    CUDA_VISIBLE_DEVICES=0 python3 get_attention.py \
        --bws_file ./bws_list_40.txt \
        --fasta_file **your fasta file path** \
        --bert_model **your model dir** \
        --output_dir attention \
        --peak_bed_dir **your peak region beds dir for all experiments, can get from ../bpBERT/intervals.py** \
        --n_tasks 40 \
        --max_seq_length 1002 \
        --kmer_list kmer45678_0.1_GRCh38.txt \
        --kmer_len_list 4_5_6_7_8 \
        --max_kmer_in_sequence 1500 \
        --eval_batch_size 1 \
        --num_workers 1 \
        --exapriment "$exapriment_param"
    wait
done < all_experiments.txt
