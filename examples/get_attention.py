from __future__ import absolute_import, division, print_function
from copy import deepcopy
import imp
import sys
sys.path.append("..")
import gzip
import logging
import math
from random import shuffle
from copy import deepcopy
from pathlib import Path
from kipoi_utils.data_utils import numpy_collate_concat
from pybedtools import BedTool, Interval
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset,
                              TensorDataset)
from tqdm import tqdm, trange
import numpy as np
import seaborn as sns
import os
import argparse
import datetime
from collections import OrderedDict,defaultdict
from kipoi_utils.external.flatten_json import flatten, unflatten

logger = logging.getLogger(__name__)

from bpBERT.modeling import SeqModelForMotif
from bpBERT.extract import extract_bigwig_stranded, extract_fasta
from bpBERT.tokenization import BertTokenizer
from bpBERT.kmer_utils import bpBERTKmerDict, KMER_DICT_NAME
from bpBERT.functions import softmax, mean

from tracks import filter_tracks, plot_tracks, to_neg
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
def flatten_list(l):
    """Flattens a nested list
    """
    return [x for nl in l for x in nl]

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # 创建文件夹，包括所有必要的父文件夹
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def get_attn_profile_score(attention_scores,profile):
    # to obtain the profile-weighted attntion score
    # Convert profile to tensor for calculation
    profile_tensor = torch.tensor(profile, dtype=torch.float32)
    profile_tensor = profile_tensor.to(attention_scores.device)
    # print('attention_scores',attention_scores.device,attention_scores.shape)
    # print('profile_tensor',profile_tensor.device,profile_tensor.shape)

    # Initializes the weight tensor, which stores the final weight value for each token
    avg_profile_attention = torch.zeros_like(profile_tensor, dtype=torch.float32)
    avg_profile_attention2 = torch.zeros_like(profile_tensor, dtype=torch.float32)
    avg_head_attention = attention_scores.mean(dim=1)  # Take the average shape of the 12 heads，[bs,1002,1002]


    # Calculate the weight value for each token
    for token_idx in range(1002):
        # Step 1: The attention weights of the current token and all tokens are extracted, and the weights of all heads are averaged
        
        # print('avg_head_attention',avg_head_attention.shape)
        avg_attention_for_token = avg_head_attention[:,:, token_idx]
        
        # avg_attention_for_token = F.softmax(avg_attention_for_token, dim=0)

        # Step 2: Multiply the attention weight with the profile (read count)
        # print(avg_attention_for_token,profile_tensor)
        res = torch.einsum('bi,bij->bj', avg_attention_for_token, profile_tensor)
        # weighted_read_counts = avg_attention_for_token * profile_tensor  # Element-by-element multiplication
        # print(res.shape)
        
        avg_profile_attention[:,token_idx,:] = res
        # avg_attention_sum1[token_idx] = sum(avg_head_attention[0,token_idx, :])
        # print(token_idx,sum(avg_head_attention[0,token_idx, :]).item(),sum(avg_attention_for_token).item(),weighted_read_counts.sum().item())

        # Step 3: Sum the product to get the final weight of the token
        res2 = torch.einsum('bi,bij->bj', avg_head_attention[:,token_idx, :], profile_tensor)
        avg_profile_attention2[:,token_idx,:] = res2
        # print("res",res[0,12],res2[0,12],sum(avg_head_attention[0,:, 700]),sum(avg_head_attention[0,700, :]))


    avg_profile_attention = avg_profile_attention.cpu().numpy()  
    avg_profile_attention2 = avg_profile_attention2.cpu().numpy()
    
    return avg_profile_attention,avg_profile_attention2


def get_attn_score(attention,idx):
    # to obtain the attntion score of each token for the pos&neg idx positions
    idx = torch.tensor(idx)

    cls_attention = [attention[i, :, index[0], :]+attention[i, :, index[1], :] for i,index in enumerate(idx)]
    cls_attention = torch.stack(cls_attention).mean(dim=1) 
    avg_cls_attention = cls_attention.cpu().numpy()
    norms = np.linalg.norm(avg_cls_attention, axis=1, keepdims=True)
    norms[norms == 0] = 1

    normalized_avg_cls_attention = avg_cls_attention / norms
    # print('normalized_avg_cls_attention',normalized_avg_cls_attention.shape)

    return normalized_avg_cls_attention

def get_attn_score_single(attention,idx):
    # to obtain the attntion score of each token for the idx position
    cls_attention = attention[:, :, idx, :]
    avg_cls_attention = cls_attention.mean(dim=1)

    avg_cls_attention = avg_cls_attention.cpu().numpy()
    norms = np.linalg.norm(avg_cls_attention, axis=1, keepdims=True)
    norms[norms == 0] = 1

    normalized_avg_cls_attention = avg_cls_attention / norms

    return normalized_avg_cls_attention


class TfbsDataset(Dataset):
    def __init__(self, intervals,max_seq_length, tokenizer, kmer_dict,kmer_len_list,fasta_file,bws_file):
        super(TfbsDataset, self).__init__()
        self.intervals = intervals
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.kmer_dict = kmer_dict
        self.kmer_len_list = kmer_len_list
        self.fasta_file = fasta_file
        self.bws_file = bws_file

        print("The number of data is {}".format(len(self.intervals)))
    
    def __len__(self):
        return len(self.intervals)

    def __getitem__(self, index):
        feature = self.intervals[index]
        feature = feature.strip('\n')
        features = feature.split('	')
        chrom = features[0]
        start = int(features[1])
        stop = int(features[2])

        enhancer = ''
        max_seq_length = self.max_seq_length

        interval = Interval(chrom, start, stop)
        interval_str = f"{interval.chrom}:{interval.start}-{interval.end}({enhancer})"
        
        seq = extract_fasta(self.fasta_file, [interval])
        # prepare bigwigs
        target_wigs = OrderedDict()
        iiidx = 0
        for line in open(self.bws_file, encoding='UTF-8'):
            line = line.rstrip().split('\t')
            target_wigs[line[0]] = [line[1], line[2]]
            iiidx = iiidx+1
        n_tasks = len(target_wigs)

        profile = []
        for task, wig_files in target_wigs.items():
            bws_stranded = extract_bigwig_stranded(wig_files, [interval])
            profile.append(bws_stranded)
        profile = np.array(profile)
        profile = np.transpose(profile, axes=(1,2,3,0))

        total_count_transform = lambda x: np.log(1 + x)
        counts = total_count_transform(np.sum(profile,axis=1))

        profile_one = profile[0].tolist()
        counts_one = counts.tolist()
        channels = 2
        if len(profile_one) >= max_seq_length - 1:
            profile_one = profile_one[0:(max_seq_length - 2)]
        profile_one.insert(0, [[0.0] * n_tasks for _ in range(channels)])
        profile_one.append([[0.0] * n_tasks for _ in range(channels)])

        while len(profile_one) < max_seq_length:
            profile_one.append([[0.0] * n_tasks for _ in range(channels)])

        profile_one = torch.tensor(profile_one, dtype=torch.float)
        counts_one = torch.tensor(counts_one, dtype=torch.float)

        features = convert_seq_to_features(seq, self.max_seq_length, self.tokenizer, self.kmer_dict,self.kmer_len_list)

        return features+(profile_one,counts_one)


def convert_seq_to_features(seq, max_seq_length, tokenizer, kmer_dict,kmer_len_list):
    # convert seq to features
    tokens = []
    for i, word in enumerate(seq):
        token = tokenizer.tokenize(word)
        tokens.extend(token)

    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
    
    ntokens = []
    ntokens.append("[CLS]")
    for i, token in enumerate(tokens):
        ntokens.append(token)
    ntokens.append("[SEP]")
    
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
            
    # ----------- code for kmer BEGIN-----------
    kmer_matches = []
    #  Filter the kmer segment from 6 to 13 to check whether there is a kmer
    for p in kmer_len_list:
        for q in range(0, len(ntokens) - p + 1):
            character_segment = ntokens[q:q + p]
            # j is the starting position of the kmer
            # i is the length of the current kmer
            character_segment = tuple(character_segment)
            if character_segment in kmer_dict.kmer_to_id_dict:
                kmer_index = kmer_dict.kmer_to_id_dict[character_segment]
                kmer_matches.append([kmer_index, q, p])

    # shuffle(kmer_matches)

    max_kmer_in_seq_proportion = math.ceil((len(ntokens) / max_seq_length) * kmer_dict.max_kmer_in_seq)
    if len(kmer_matches) > max_kmer_in_seq_proportion:
        kmer_matches = kmer_matches[:max_kmer_in_seq_proportion]

    idx_range = list(range(0,len(kmer_matches)))
    shuffle(idx_range)
    ignore_kmer = []
    if len(kmer_matches) > max_kmer_in_seq_proportion:
        ignore_kmer=idx_range[max_kmer_in_seq_proportion:]

    kmer_ids = []
    kmer_positions = []
    kmer_lengths = []

    for index,item in enumerate(kmer_matches):
        if index not in ignore_kmer:
            item = list(item)
            kmer_ids.append(item[0])
            kmer_positions.append(item[1])
            kmer_lengths.append(item[2])

    kmer_mask_array = np.zeros(kmer_dict.max_kmer_in_seq, dtype=bool)
    kmer_mask_array[:len(kmer_ids)] = 1

    # record the masked positions
    kmer_positions_matrix = np.zeros(shape=(max_seq_length, kmer_dict.max_kmer_in_seq), dtype=np.int32)
    for i in range(len(kmer_ids)):
        kmer_positions_matrix[kmer_positions[i]:kmer_positions[i] + kmer_lengths[i], i] = 1.0

    # Zero-pad up to the max kmer in seq length.
    padding = [0] * (kmer_dict.max_kmer_in_seq - len(kmer_ids))
    kmer_ids += padding
    kmer_lengths += padding
    kmer_positions += padding

    # ----------- code for kmer END-----------

    features = (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(kmer_ids, dtype=torch.long),
                torch.tensor(kmer_positions, dtype=torch.long),
                torch.tensor(kmer_positions_matrix, dtype=torch.long),
                torch.tensor(kmer_lengths, dtype=torch.long),
                torch.tensor(kmer_mask_array, dtype=torch.long))
    return features

def main():
    parser = argparse.ArgumentParser()
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    parser.add_argument("--bws_file",type=str,required=True,help="The list of profile .bw files")
    parser.add_argument("--fasta_file",type=str,required=True,help="The reference genome .fasta file")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bpBERT-based"
                        "bert-base-multilingual-cased, bert-base-chinese.")

    parser.add_argument("--output_dir",
                       default='./results/result-tokenlevel-{}'.format(now_time),
                       type=str,
                       help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--n_tasks", type=int, required=True, help="Number of tasks.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--kmer_list", type=str, default="kmer.txt")
    parser.add_argument("--kmer_len_list", type=str, default='4_5_6_7_8')
    parser.add_argument("--peak_bed_dir", type=str, default='test.bed')
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--max_kmer_in_sequence", type=int, default=900)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    info_list = []
    with open('./datalist_40.tsv','r',encoding='utf-8', newline='') as f:
        chr_list = f.readlines()
        # print(chr_list)
        for line in chr_list[1:]:
            info = line.strip().split('	')
            assert len(info) == 6
            experiment,alignments,IDR_thresholded_peaks,description,TF,Cell = info
            # print(experiment,alignments,IDR_thresholded_peaks,description,TF,Cell)
            info_list.append((experiment,alignments,IDR_thresholded_peaks,description,TF,Cell))

    
    tf_dict = {s[0]: (index,s[1],s[2],s[3],s[4],s[5]) for index, s in enumerate(info_list)}
    tf_sign = tf_dict[args.experiment][0]
    print('tf_dict',tf_sign,args.experiment,tf_dict[args.experiment][-2],tf_dict[args.experiment][-1],tf_dict)
    args.peak_bed_dir = f'{args.peak_bed_dir}/{args.experiment}.bed'
    args.output_dir = f'./figures/1000bp/{tf_dict[args.experiment][-1]}/{tf_dict[args.experiment][-2]}_{args.experiment}/{args.output_dir}/'

    


    create_folder_if_not_exists(args.output_dir)
    kmer_len_list_str = args.kmer_len_list.split('_')
    kmer_len_list = []
    print(kmer_len_list_str)
    for item in kmer_len_list_str:
        # print()
        kmer_len = int(item)
        kmer_len_list.append(kmer_len)
    print(kmer_len_list)

    # Prepare model tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    kmer_dict = bpBERTKmerDict(args.kmer_list, tokenizer=tokenizer,max_kmer_in_seq=args.max_kmer_in_sequence)

    model = SeqModelForMotif.from_pretrained(args.bert_model, args.n_tasks,output_attentions=True)
    model.to("cuda")
    args.device = torch.device("cuda")
    model.eval()


    with open(args.peak_bed_dir,'r',encoding='utf-8', newline='') as f:
        chr_list = f.readlines()
        print(chr_list[0])

        eval_dataset = TfbsDataset(chr_list,args.max_seq_length, tokenizer, kmer_dict,kmer_len_list,args.fasta_file,args.bws_file)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_dataset)
        # eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,num_workers=args.num_workers, pin_memory=True, batch_size=args.eval_batch_size)

        # Eval!
        logging.info("***** Running evaluation *****")
        logging.info("  Num examples = %d", len(eval_dataset))
        logging.info("  Batch size = %d", args.eval_batch_size)
        

        num_samples = len(chr_list)
        print("all",num_samples)

        batch_size = args.eval_batch_size
        # all_avg_cls_attention = np.zeros([num_samples,args.max_seq_length])
        all_avg_cls_attention_peak = np.zeros([num_samples,args.max_seq_length])
        # all_avg_profile_attention = np.zeros([num_samples,args.max_seq_length,40])
        all_input_ids = np.zeros([num_samples,args.max_seq_length])
        # # all_gt_profile = np.zeros([num_samples,args.max_seq_length,2,args.n_tasks])
        # all_pred_profile = np.zeros([num_samples,args.max_seq_length,2,args.n_tasks])
        # # all_gt_counts = np.zeros([num_samples,2,args.n_tasks])
        # all_pred_counts = np.zeros([num_samples,2,args.n_tasks])
        count = 0

        for index, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):

            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, kmer_ids, kmer_positions,kmer_positions_matrix, kmer_lengths, kmer_masks, profile, counts = batch
            # print(profile)

            # print(index,input_ids.shape,input_mask.shape,kmer_ids.shape,kmer_positions.shape,kmer_positions_matrix.shape,kmer_lengths.shape,kmer_masks.shape,profile.shape,counts.shape)
            targets = {"profile": profile, "counts": counts}


            with torch.no_grad():
                logits, all_attentions, attention_kmer,loss = model(input_ids=input_ids,
                    input_kmer_ids=kmer_ids,
                    kmer_position_matrix=kmer_positions_matrix,
                    kmer_position_ids=kmer_positions,
                    kmer_type_ids=kmer_lengths,
                    targets=targets,
                    # attention_mask_label=l_mask,
                    attention_mask=input_mask,
                    kmer_attention_mask=kmer_masks,
                    mode="validate")
                attention = all_attentions[-1]
                tf_logits = logits['profile'][:,:,:,tf_sign]
                logits_posneg = np.sum(logits['profile'], axis=2)
                # print(logits['profile'].shape,logits_posneg.shape)
                max_indices = np.argmax(tf_logits, axis=1)
                # print(max_indices.shape,max_indices)

                # avg_profile_attention,avg_profile_attention2 = get_attn_profile_score(attention,logits_posneg)
                normalized_avg_cls_attention_peak = get_attn_score(attention,max_indices)
                # normalized_avg_cls_attention = get_attn_score_single(attention,0)

                # print(avg_cls_attention.shape,logits['counts'],logits['profile'])
                # all_avg_cls_attention.append(avg_cls_attention)
                # print(all_avg_cls_attention[index*batch_size:index*batch_size+input_ids.shape[0],:].shape,all_avg_cls_attention[index*batch_size:index*batch_size+input_ids.shape[0],:])
                all_avg_cls_attention_peak[index*batch_size:index*batch_size+input_ids.shape[0],:] = normalized_avg_cls_attention_peak
                # all_avg_cls_attention[index*batch_size:index*batch_size+input_ids.shape[0],:] = normalized_avg_cls_attention
                # all_avg_profile_attention[index*batch_size:index*batch_size+input_ids.shape[0],:] = avg_profile_attention
                # print(all_avg_cls_attention.shape)
                # print(all_avg_cls_attention[index*batch_size:index*batch_size+input_ids.shape[0],:].shape,all_avg_cls_attention[index*batch_size:index*batch_size+input_ids.shape[0],:])
                # all_input_ids.append(input_ids.cpu().numpy())

                # print(all_input_ids[index*batch_size:index*batch_size+input_ids.shape[0],:].shape,all_input_ids[index*batch_size:index*batch_size+input_ids.shape[0],:])
                all_input_ids[index*batch_size:index*batch_size+input_ids.shape[0],:] = input_ids.cpu().numpy()
                # print(all_input_ids[index*batch_size:index*batch_size+input_ids.shape[0],:].shape,all_input_ids[index*batch_size:index*batch_size+input_ids.shape[0],:])

                # # print('all_gt_profile',targets['profile'].shape,all_gt_profile[index*batch_size:index*batch_size+input_ids.shape[0],:,:,:])
                # all_gt_profile[index*batch_size:index*batch_size+input_ids.shape[0],:,:,:] = targets['profile'].cpu().numpy()
                # # print('all_gt_profile',targets['profile'].shape,all_gt_profile[index*batch_size:index*batch_size+input_ids.shape[0],:,:,:])

                # print('all_pred_profile',all_pred_profile[index*batch_size:index*batch_size+input_ids.shape[0],:,:,:].shape,all_pred_profile[index*batch_size:index*batch_size+input_ids.shape[0],:,:,:])
                # all_pred_profile[index*batch_size:index*batch_size+input_ids.shape[0],:,:,:] = logits['profile']
                # print('all_pred_profile',all_pred_profile[index*batch_size:index*batch_size+input_ids.shape[0],:,:,:].shape,all_pred_profile[index*batch_size:index*batch_size+input_ids.shape[0],:,:,:])


                # # print('all_gt_counts',all_gt_counts[index*batch_size:index*batch_size+input_ids.shape[0],:,:])
                # all_gt_counts[index*batch_size:index*batch_size+input_ids.shape[0],:,:] = targets['counts'].cpu().numpy()
                # # print('all_gt_counts',all_gt_counts[index*batch_size:index*batch_size+input_ids.shape[0],:,:])

                # # print('all_pred_counts',all_pred_counts[index*batch_size:index*batch_size+input_ids.shape[0],:,:])
                # all_pred_counts[index*batch_size:index*batch_size+input_ids.shape[0],:,:] = logits['counts']
                # # print('all_pred_profile',all_pred_profile[index*batch_size:index*batch_size+input_ids.shape[0],:,:])

                # all_gt_profile.append(targets['profile'].cpu().numpy())
                # all_pred_profile.append(logits['profile'])
                # all_gt_counts.append(targets['counts'].cpu().numpy())
                # all_pred_counts.append(logits['counts'])
                if index%100 == 0:
                    print('Saving at:',index*batch_size,index*batch_size+input_ids.shape[0])
                    np.save(f"{args.output_dir}/avg_cls_attention_peak.npy", all_avg_cls_attention_peak)
                    # np.save(f"{args.output_dir}/avg_cls_attention.npy", all_avg_cls_attention)
                    # np.save(f"{args.output_dir}/avg_profile_attention.npy", all_avg_profile_attention)
                    np.save(f"{args.output_dir}/all_input_ids.npy", all_input_ids)
            
        print(f"Final avg_cls_attention_peak shape: {all_avg_cls_attention_peak.shape}")
        # print(f"Final avg_cls_attention shape: {all_avg_cls_attention.shape}")
        # print(f"Final avg_profile_attention shape: {all_avg_profile_attention.shape}")
        print(f"Final all_input_ids shape: {all_input_ids.shape}")
        # # print(f"Final all_gt_profile shape: {all_gt_profile.shape}")
        # print(f"Final all_pred_profile shape: {all_pred_profile.shape}")
        # # print(f"Final all_gt_counts shape: {all_gt_counts.shape}")
        # print(f"Final all_pred_counts shape: {all_pred_counts.shape}")

        # save to numpy
        np.save(f"{args.output_dir}/avg_cls_attention_peak.npy", all_avg_cls_attention_peak)
        # np.save(f"{args.output_dir}/avg_cls_attention.npy", all_avg_cls_attention)
        # np.save(f"{args.output_dir}/avg_profile_attention.npy", all_avg_profile_attention)
        np.save(f"{args.output_dir}/all_input_ids.npy", all_input_ids)
        # np.save(f"{args.output_dir}/all_gt_profile.npy", all_gt_profile)
        # np.save(f"{args.output_dir}/all_pred_profile.npy", all_pred_profile)
        # # np.save(f"{args.output_dir}/all_gt_counts.npy", all_gt_counts)
        # np.save(f"{args.output_dir}/all_pred_counts.npy", all_pred_counts)
        
        print('tf_dict',tf_sign,args.experiment,tf_dict[args.experiment][-2],tf_dict[args.experiment][-1])

        

if __name__ == '__main__':
    main()
