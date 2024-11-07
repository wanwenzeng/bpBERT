# bpBERT

bpBERT is a deep learning model based on BERT, designed for predicting transcription factor binding sites on DNA sequences with base-pair resolution. This model aims to reveal the regulatory syntax of DNA sequences and assess the functional impact of genetic variants, making it applicable for research in gene regulation and genomics.

![bpBERT Model Architecture](docs/fig1_bpBERT-overview.png)

## Environment setup
We recommend you to build a python virtual environment with Anaconda. 
```bash
conda create -n bpbert python=3.9.7

conda activate bpbert

conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge

conda install -c bioconda pybedtools bedtools pybigwig pysam htslib

pip install tensorboardX tqdm kipoi_utils h5py scikit-learn matplotlib boto3 requests numpy==1.22.4
```

## Pretrain
You can pretrain bpBERT from scratch with DDP using the script:
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 run_pretrain_online.py \
     --bert_model bpBERT-base \
     --scratch \
     --training_data sample_data.txt \
     --output_dir ../models/pretrain/ \
     --train_batch_size 1 \
     --epochs 40 \
     --learning_rate 3.5e-5 \
     --kmer_list kmer45678_0.1_GRCh38.txt \
     --max_kmer_in_sequence 750 \
     --warmup_proportion 0 \
     --comment pretrain \
     --seed 36 \
     --kmer_dict_size 8735 \
     --multi_gpu \
     --max_seq_len 202 
```

## Finetune
You can finetune bpBERT on tfbs task with DDP using the script:
```bash
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
```

You can evaluate the performance of bpBERT using the script:
```bash
CUDA_VISIBLE_DEVICES=0 python3 run_tfbs_resume_epoch_lenEmb_40tfs.py \
 --hdf5_file **your .h5 file path** \
 --bert_model **your finetuned model dir** \
 --output_dir eval_results/ \
 --n_tasks 40 \
 --do_eval \
 --max_seq_length 1002 \
 --max_ngram_in_sequence 1500 \
 --eval_batch_size 16 \
 --from_pretrained \
 --ngram_list kmer45678_0.1_GRCh38.txt \
 --kmer_len_list 4_5_6_7_8
 ```

## Motif analysis
You can use fientuned bpBERT to find important motifs:
```bash
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
```

And you should compute the attention scores before you run motif finding:
```bash
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
```

## SNPs analysis
You can compute the importance scores and the mutation scores of the SNPs you are intrested in using the script:
```bash
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
```

