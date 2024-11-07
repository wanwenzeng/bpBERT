#!/bin/bash

while IFS= read -r tf_param; do

    python3 find_motif_from_attn.py \
        --data_dir ./figures/1000bp \
        --window_size 24 \
        --min_len 7 \
        --min_n_motif 3 \
        --align_all_ties \
        --verbose \
        --save_file_dir test \
        --tf "$tf_param"

    wait
done < all_experiments.txt
