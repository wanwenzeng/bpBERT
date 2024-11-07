#!/bin/bash

python3 hdf5.py \
 --beds_file ./beds_list_40.txt \
 --bws_file ./bws_list_40.txt \
 --fasta_file **your fasta file path** \
 --hdf5_file data.h5 \
 --exclude_chr 'chrX' 'chrY'