#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 analyze_snp.py \
--bert_model **your model dir** \
--data_file ./gwas/ALL/snp_pred_fold_change_33.tsv \
--output_dir ./gwas/ALL \
--n_tasks 40 \
--rsid rs6428370 \
--alt T \
--task_id 33 \
--max_ngram_in_sequence 1500 \
--max_seq_length 1002 \
--batch_size 1 \
--ngram_list kmer45678_0.1_GRCh38.txt \
--kmer_len_list 4_5_6_7_8