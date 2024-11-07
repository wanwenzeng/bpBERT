# coding: utf-8
# Copyright 2019 Sinovation Ventures AI Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run finetuning on bpBERT model."""

import sys
sys.path.append("..")

import logging
logging.getLogger().setLevel(logging.INFO)

from tensorboardX import SummaryWriter

import argparse
import json
import os
import math
import random 
from random import shuffle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import datetime
from collections import OrderedDict, defaultdict
from kipoi_utils.external.flatten_json import flatten, unflatten
from kipoi_utils.data_utils import numpy_collate_concat
from copy import deepcopy
import shutil
import h5py

from bpBERT.tokenization import BertTokenizer
from bpBERT.optimization_trans import AdamW, get_linear_schedule_with_warmup
from bpBERT.kmer_utils import bpBERTKmerDict, NGRAM_DICT_NAME
from bpBERT.modeling import MutitaskModel, bpBERTConfig, ProfileHead, ScalarHead
from bpBERT.file_utils import WEIGHTS_NAME, CONFIG_NAME, PYTORCH_PRETRAINED_BERT_CACHE
from bpBERT.functions import softmax, mean
from scipy.stats import pearsonr
from bpBERT.metrics import average_dict
logger = logging.getLogger(__name__)


def convert_example_to_features(seq, profile, counts, max_seq_length, tokenizer, kmer_dict,kmer_len_list):

    seq_len, channels, n_tasks = profile.shape
    profile_one = profile.tolist()
    counts_one = counts.tolist()
    
    tokens = []
    for i, word in enumerate(seq):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        profile_one = profile_one[0:(max_seq_length - 2)]
    
    ntokens = []
    ntokens.append("[CLS]")
    profile_one.insert(0, [[0.0] * n_tasks for _ in range(channels)])
    for i, token in enumerate(tokens):
        ntokens.append(token)
    ntokens.append("[SEP]")
    profile_one.append([[0.0] * n_tasks for _ in range(channels)])
    
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    label_mask = [[[1] * n_tasks for _ in range(channels)] for i in range(len(input_ids))]

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        label_mask.append([[0] * n_tasks for _ in range(channels)])
        profile_one.append([[0.0] * n_tasks for _ in range(channels)])

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(label_mask) == max_seq_length
    assert len(profile_one) == max_seq_length
            
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
                torch.tensor(kmer_mask_array, dtype=torch.long),
                torch.tensor(label_mask, dtype=torch.long),
                torch.tensor(profile_one, dtype=torch.float),
                torch.tensor(counts_one, dtype=torch.float))
    return features


def get_kmer_sequence(original_string, kmer=1):
    if kmer == -1:
        return original_string

    sequence = []
    original_string = original_string.replace("\n", "")
    for i in range(len(original_string)-kmer):
        sequence.append(original_string[i:i+kmer])
    
    sequence.append(original_string[-kmer:])
    return sequence

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_data(args, tokenizer, kmer_dict, mode):
    kmer_len_list_str = args.kmer_len_list.split('_')
    kmer_len_list = []
    print(kmer_len_list_str)
    for item in kmer_len_list_str:
        kmer_len = int(item)
        kmer_len_list.append(kmer_len)
    print(kmer_len_list)

    hdf5_file = h5py.File(args.hdf5_file, 'r')

    if mode == "train":
        seqs = hdf5_file['train_seqs']
        profile = hdf5_file['train_profile']
        counts = hdf5_file['train_counts']
    elif mode == "test":
        seqs = hdf5_file['test_seqs']
        profile = hdf5_file['test_profile']
        counts = hdf5_file['test_counts']
    else:
        seqs = hdf5_file['validate_seqs']
        profile = hdf5_file['validate_profile']
        counts = hdf5_file['validate_counts']

    return TfbsDataset(seqs, profile, counts, args.max_seq_length, tokenizer, kmer_dict,kmer_len_list)


class TfbsDataset(Dataset):
    def __init__(self, seqs, profile, counts, max_seq_length, tokenizer, kmer_dict,kmer_len_list):
        super(TfbsDataset, self).__init__()
        self.seqs = seqs
        self.profile = profile
        self.counts = counts

        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.kmer_dict = kmer_dict
        self.kmer_len_list = kmer_len_list

        assert len(self.seqs) == len(self.profile)
        assert len(self.seqs) == len(self.counts)

        print("The number of data is {}".format(len(self.seqs)))

    
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = str(self.seqs[index], encoding='utf-8')
        profile_one = self.profile[index]
        counts_one = self.counts[index]

        profile_one = profile_one.astype(np.float32)
        counts_one = counts_one.astype(np.float32)

        features = convert_example_to_features(seq, profile_one, counts_one, self.max_seq_length, self.tokenizer, self.kmer_dict,self.kmer_len_list)
        return features


def save_bpbert_model(save_bpbert_model_path, model, tokenizer, kmer_dict, args):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(save_bpbert_model_path, WEIGHTS_NAME)
    output_config_file = os.path.join(save_bpbert_model_path, CONFIG_NAME)
    output_kmer_dict_file = os.path.join(save_bpbert_model_path, NGRAM_DICT_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(save_bpbert_model_path)
    kmer_dict.save(output_kmer_dict_file)
    output_args_file = os.path.join(save_bpbert_model_path, 'training_args.bin')
    torch.save(args, output_args_file)



def evaluate(args, model, tokenizer, kmer_dict):
    # Run prediction for full data
    eval_dataset = load_data(args, tokenizer, kmer_dict, mode="test")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)

    # Eval!
    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)

    n_tasks = args.n_tasks

    import os.path as path
    temp_dir = os.path.join(args.output_dir, "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    filename1 = path.join(temp_dir, 'newfile1.dat')
    filename2 = path.join(temp_dir, 'newfile2.dat')
    filename3 = path.join(temp_dir, 'newfile3.dat')
    filename4 = path.join(temp_dir, 'newfile4.dat')
    fp_profile_pred = np.memmap(filename1, dtype='float32', mode='w+', shape=eval_dataset.profile.shape)
    fp_counts_pred = np.memmap(filename2, dtype='float32', mode='w+', shape=eval_dataset.counts.shape)
    fp_profile_targets = np.memmap(filename3, dtype='float32', mode='w+', shape=eval_dataset.profile.shape)
    fp_counts_targets = np.memmap(filename4, dtype='float32', mode='w+', shape=eval_dataset.counts.shape)


    model.eval()
    si = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, kmer_ids, kmer_positions,kmer_positions_matrix, kmer_lengths, kmer_masks, l_mask, profile, counts = batch
        targets = {"profile": profile, "counts": counts}

        with torch.no_grad():
            logits, _ = model(input_ids=input_ids,
                input_kmer_ids=kmer_ids,
                kmer_position_matrix=kmer_positions_matrix,
                kmer_position_ids=kmer_positions,
                kmer_type_ids=kmer_lengths,
                attention_mask_label=l_mask,
                attention_mask=input_mask,
                kmer_attention_mask=kmer_masks)

        assert isinstance(logits, dict)

        for k, v in targets.items():
            targets[k] = v.detach().cpu().numpy()

        # Delete the paddings
        batch_size, max_len, feat_dim, n_tasks = logits['profile'].shape
        active = l_mask.detach().cpu().numpy().reshape(-1) == 1
        active_logits = logits['profile'].reshape(-1)[active]
        active_labels = targets['profile'].reshape(-1)[active]
        active_logits = active_logits.reshape(batch_size, -1, feat_dim, n_tasks)
        active_labels = active_labels.reshape(batch_size, -1, feat_dim, n_tasks)

        # Delete [CLS] and [SEP]
        indexes = [0, -1]
        logits['profile'] = np.delete(active_logits, indexes, axis=1)
        targets['profile'] = np.delete(active_labels, indexes, axis=1)

        bs = len(logits['profile'])
        fp_profile_pred[si:si+bs, ...] = logits['profile']
        fp_counts_pred[si:si+bs, ...] = logits['counts']
        fp_profile_targets[si:si+bs, ...] = targets['profile']
        fp_counts_targets[si:si+bs, ...] = targets['counts']

        si += bs

    preds = {"profile": [], "counts": []}
    preds["profile"] = fp_profile_pred
    preds["counts"] = fp_counts_pred
    labels = {"profile": [], "counts": []}
    labels["profile"] = fp_profile_targets
    labels["counts"] = fp_counts_targets


    # Compute the metrics
    if args.local_rank != -1:
        model = model.module
    out = defaultdict(dict)
    for i in range(model.n_tasks):
        for j, head in enumerate(model.heads):
            head_name = head.get_name()
            if head.postproc_fn is not None:
                preds[head_name][..., i] = head.postproc_fn(preds[head_name][..., i])
            res = head.metric(labels[head_name][..., i],
                            preds[head_name][..., i])
            if res == None:
                continue
            for k, v in res.items():
                out[i][k] = v
    os.remove(filename1)
    os.remove(filename2)
    os.remove(filename3)
    os.remove(filename4)
    del preds
    del labels

    avg_res = average_dict(out)

    out_all = deepcopy(out)
    out_all['avg'] = avg_res

    for k, v in out_all.items():
        for k_m, v_m in v.items():
            if isinstance(v_m, dict):
                for k_b, v_b in v_m.items():
                    out_all[k][k_m][k_b] = float(v_b)
            else:
                out_all[k][k_m] = float(v_m)

    # write the results
    json_str = json.dumps(out_all, indent=2)
    evaluation_path = f'{args.output_dir}eval_metrics.json'
    with open(evaluation_path, 'w') as json_file:
        json_file.write(json_str)
    logging.info("Saved metrics to {}".format(evaluation_path))

    return out_all


def gather_tensors(args, tensor):
    tensor = tensor.contiguous()
    gathered_tensors = [torch.zeros_like(tensor, dtype=torch.float).to(args.device) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered_tensors, tensor)
    return gathered_tensors


def validate(args, model, tokenizer, kmer_dict, mode="validate"):
    # Run on the validating set
    eval_dataset = load_data(args, tokenizer, kmer_dict, mode=mode)
    
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        eval_sampler = DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)

    # Eval!
    logging.info("***** Running validation *****")
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    val_loss_list = []
    profile_loss_list = []
    counts_loss_list = []
    all_predictions_counts = [[] for _ in range(args.n_tasks)]
    all_targets_counts = [[] for _ in range(args.n_tasks)]
    all_predictions_profile = [[] for _ in range(args.n_tasks)]
    all_targets_profile = [[] for _ in range(args.n_tasks)]

    for batch in tqdm(eval_dataloader, desc="Validating"):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, kmer_ids, kmer_positions,kmer_positions_matrix, kmer_lengths, kmer_masks, l_mask, profile, counts = batch
        targets = {"profile": profile, "counts": counts}

        with torch.no_grad():
            eval_loss, preds, head_loss = model(input_ids=input_ids,
                input_kmer_ids=kmer_ids,
                kmer_position_matrix=kmer_positions_matrix,
                kmer_position_ids=kmer_positions,
                kmer_type_ids=kmer_lengths,
                targets=targets,
                output_logits=True,
                attention_mask_label=l_mask,
                attention_mask=input_mask,
                kmer_attention_mask=kmer_masks)
            profile_loss = head_loss['profile']
            counts_loss = head_loss['counts']

            if args.local_rank != -1:
                # gather the tensors from each gpu if DDP
                gathered_val_loss_tensor_list = [torch.zeros(1, dtype=torch.float).to(args.device) for _ in range(torch.distributed.get_world_size())] # len=2
                torch.distributed.all_gather(gathered_val_loss_tensor_list, eval_loss)
                for t in gathered_val_loss_tensor_list:
                    val_loss_list.extend(t.detach().cpu().numpy().tolist())

                # profile_loss
                gathered_profile_loss_tensor_list = [torch.zeros(1, dtype=torch.float).to(args.device) for _ in range(torch.distributed.get_world_size())] # len=2
                torch.distributed.all_gather(gathered_profile_loss_tensor_list, profile_loss)
                for t in gathered_profile_loss_tensor_list:
                    profile_loss_list.extend(t.detach().cpu().numpy().tolist())
                
                # counts_loss
                gathered_counts_loss_tensor_list = [torch.zeros(1, dtype=torch.float).to(args.device) for _ in range(torch.distributed.get_world_size())] # len=2
                torch.distributed.all_gather(gathered_counts_loss_tensor_list, counts_loss)
                for t in gathered_counts_loss_tensor_list:
                    counts_loss_list.extend(t.detach().cpu().numpy().tolist())

                # gather the tensors from each gpu if DDP
                for i in range(args.n_tasks):
                    gathered_predictions_counts = torch.cat(gather_tensors(args, preds['counts'][i]), dim=0)
                    gathered_predictions_profile = torch.cat(gather_tensors(args, preds['profile'][i]), dim=0)

                    gathered_targets_counts = torch.cat(gather_tensors(args, targets['counts'][:, :, i]), dim=0)
                    gathered_targets_profile = torch.cat(gather_tensors(args, targets['profile'][:, :, :, i]), dim=0)

                    all_predictions_counts[i].append(gathered_predictions_counts)
                    all_predictions_profile[i].append(gathered_predictions_profile)
                    all_targets_counts[i].append(gathered_targets_counts)
                    all_targets_profile[i].append(gathered_targets_profile)
                
                
            else:
                val_loss_list.append(eval_loss.detach().cpu().numpy().item())
                profile_loss_list.append(profile_loss.detach().cpu().numpy().item())
                counts_loss_list.append(counts_loss.detach().cpu().numpy().item())
                for i in range(args.n_tasks):
                    all_predictions_counts[i].append(preds['counts'][i])
                    all_predictions_profile[i].append(preds['profile'][i])
                    all_targets_counts[i].append(targets['counts'][:, :, i])
                    all_targets_profile[i].append(targets['profile'][:, :, :, i])
            
    val_loss_mean = sum(val_loss_list)/len(val_loss_list)
    profile_loss_mean = sum(profile_loss_list)/len(profile_loss_list)
    counts_loss_mean = sum(counts_loss_list)/len(counts_loss_list)

    for i in range(args.n_tasks):
        all_predictions_counts[i] = torch.cat(all_predictions_counts[i], dim=0).detach().cpu().numpy()
        all_predictions_profile[i] = torch.cat(all_predictions_profile[i], dim=0).detach().cpu().numpy()
        all_targets_counts[i] = torch.cat(all_targets_counts[i], dim=0).detach().cpu().numpy()
        all_targets_profile[i] = torch.cat(all_targets_profile[i], dim=0).detach().cpu().numpy()

        indexes = [0, -1]
        all_predictions_profile[i] = np.delete(all_predictions_profile[i], indexes, axis=1)
        all_targets_profile[i] = np.delete(all_targets_profile[i], indexes, axis=1)
    
    preds = {}
    labels = {}
    preds['profile'] = np.stack(all_predictions_profile, axis=-1)
    preds['counts'] = np.stack(all_predictions_counts, axis=-1)
    labels['profile'] = np.stack(all_targets_profile, axis=-1)
    labels['counts'] = np.stack(all_targets_counts, axis=-1)



    # Compute the metrics
    if args.local_rank != -1:
        model = model.module
    out = defaultdict(dict)
    for i in range(model.n_tasks):
        for j, head in enumerate(model.heads):
            head_name = head.get_name()
            if head.postproc_fn is not None:
                preds[head_name][..., i] = head.postproc_fn(preds[head_name][..., i])
            res = head.metric(labels[head_name][..., i],
                            preds[head_name][..., i])
            if res == None:
                continue
            for k, v in res.items():
                out[i][k] = v

    avg_res = average_dict(out)

    out_all = deepcopy(out)
    out_all['avg'] = avg_res

    for k, v in out_all.items():
        for k_m, v_m in v.items():
            if isinstance(v_m, dict):
                for k_b, v_b in v_m.items():
                    out_all[k][k_m][k_b] = float(v_b)
            else:
                out_all[k][k_m] = float(v_m)


    return val_loss_mean, profile_loss_mean, counts_loss_mean, out_all


def train(args, model, tokenizer, kmer_dict):
    train_dataset = load_data(args, tokenizer, kmer_dict, mode="train")

    if args.local_rank in [-1, 0]:
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('runs_tfbs_1003', f"{current_time}_{args.comment}")
        tb_writer = SummaryWriter(log_dir=log_dir)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank], 
             output_device=args.local_rank,find_unused_parameters=True)

    num_train_optimization_steps = int(
        len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    param_optimizer = filter(lambda t: t[1].requires_grad, list(model.named_parameters()))
    param_optimizer = list(param_optimizer)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
   

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=0.01, correct_bias=False)
    warmup_steps = args.warmup_proportion * num_train_optimization_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.continue_train
        and os.path.isfile(os.path.join(args.bert_model, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.bert_model, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.bert_model, "optimizer.pt"), map_location='cpu'))
        scheduler.load_state_dict(torch.load(os.path.join(args.bert_model, "scheduler.pt"), map_location='cpu'))


    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    loss_log = 0
    profile_loss_log = 0
    counts_loss_log = 0
    train_loss_epoch = 0
    train_profile_loss_epoch = 0
    train_counts_loss_epoch = 0

    min_loss = float("inf")
    max_auprc = 0
    early_stop_count = 0

    # Check if continuing training from a checkpoint
    if args.continue_train and os.path.exists(args.bert_model):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            steps_per_epoch = math.ceil(len(train_dataset) / args.train_batch_size / torch.distributed.get_world_size())
            checkpoint_suffix = args.bert_model.split("-")[-1].split("/")[0]

            checkpoint_suffix = args.bert_model.split('/')[-2].split('-')
            global_step = int(checkpoint_suffix[1])
            min_loss = float(checkpoint_suffix[2])
            max_auprc = float(checkpoint_suffix[3])
            early_stop_count = int(checkpoint_suffix[4])

            epochs_trained = global_step // (steps_per_epoch // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (steps_per_epoch // args.gradient_accumulation_steps)

            if args.local_rank in [-1,0]:
                logging.info("  Continuing training from checkpoint, will skip to saved global_step")
                logging.info("  Continuing training from epoch %d", epochs_trained)
                logging.info("  Continuing training from global step %d", global_step)
                logging.info("  Continuing training from min_loss %f", min_loss)
                logging.info("  Continuing training from max_auprc %f", max_auprc)
                logging.info("  Continuing training from early_stop_count %d", early_stop_count)
                logging.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logging.info("  Starting fine-tuning.")

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4, pin_memory=True)

    steps_trained_in_current_epoch = steps_trained_in_current_epoch * args.gradient_accumulation_steps
    for epoch_num in trange(epochs_trained, int(args.num_train_epochs), desc="Epoch"):
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch_num)
        model.train()

        tr_steps = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
             # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, kmer_ids, kmer_positions,kmer_positions_matrix, kmer_lengths, kmer_masks, l_mask, profile, counts = batch
            targets = {"profile": profile, "counts": counts}
            loss, head_loss = model(input_ids=input_ids,
                input_kmer_ids=kmer_ids,
                kmer_position_matrix=kmer_positions_matrix,
                kmer_position_ids=kmer_positions,
                kmer_type_ids=kmer_lengths,
                targets=targets,
                attention_mask_label=l_mask,
                attention_mask=input_mask,
                kmer_attention_mask=kmer_masks)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if args.local_rank != -1:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                loss /= torch.distributed.get_world_size()
                torch.distributed.all_reduce(head_loss['profile'], op=torch.distributed.ReduceOp.SUM)
                head_loss['profile'] /= torch.distributed.get_world_size()
                torch.distributed.all_reduce(head_loss['counts'], op=torch.distributed.ReduceOp.SUM)
                head_loss['counts'] /= torch.distributed.get_world_size()
            
            loss_log += loss.item() * args.gradient_accumulation_steps
            profile_loss_log += head_loss['profile'].item()
            counts_loss_log += head_loss['counts'].item()
            train_loss_epoch += loss.item() * args.gradient_accumulation_steps
            train_profile_loss_epoch += head_loss['profile'].item()
            train_counts_loss_epoch += head_loss['counts'].item()
            tr_steps += 1

            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                max_grad_norm = 1.0
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0]:
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('batch_loss', loss.item(), global_step)
                    
                    if args.log_steps > 0 and global_step % args.log_steps == 0:
                        tb_writer.add_scalar('loss_log', loss_log/args.log_steps, global_step)
                        tb_writer.add_scalar('profile_loss_log', profile_loss_log/args.log_steps, global_step)
                        tb_writer.add_scalar('counts_loss_log', counts_loss_log/args.log_steps, global_step)
                        loss_log  = 0
                        profile_loss_log = 0
                        counts_loss_log = 0
                
                # save checkpoint for continue training
                if args.local_rank in [-1, 0] and args.ckpt_steps > 0 and global_step % args.ckpt_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}-{}-{}-{}".format(checkpoint_prefix, global_step, min_loss, max_auprc, early_stop_count))
                    os.makedirs(output_dir, exist_ok=True)

                    logging.info("Saving model checkpoint to %s", output_dir)
                    save_bpbert_model(output_dir, model, tokenizer, kmer_dict, args)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logging.info("Saving optimizer and scheduler states to %s", output_dir)

        # Validate
        early_stop_count += 1
        val_loss, val_profile_loss, val_counts_loss, metrics = validate(args, model, tokenizer, kmer_dict)
        epoch_val_avg_auprc = metrics["avg"]["auprc"]["binsize=1"]

        if args.local_rank in [-1, 0]:
            tb_writer.add_scalar('train_loss_epoch', train_loss_epoch/(tr_steps), (epoch_num + 1))
            tb_writer.add_scalar('train_profile_loss_epoch', train_profile_loss_epoch/(tr_steps), (epoch_num + 1))
            tb_writer.add_scalar('train_counts_loss_epoch', train_counts_loss_epoch/(tr_steps), (epoch_num + 1))
            tb_writer.add_scalar('val_loss_epoch', float(val_loss), (epoch_num + 1))
            tb_writer.add_scalar('val_profile_loss_epoch', float(val_profile_loss), (epoch_num + 1))
            tb_writer.add_scalar('val_counts_loss_epoch', float(val_counts_loss), (epoch_num + 1))

            tb_writer.add_scalar('val_avg_auprc_epoch', float(metrics["avg"]["auprc"]["binsize=1"]), (epoch_num + 1))
            tb_writer.add_scalar('val_avg_preasonr_epoch', float(metrics["avg"]["pearsonr"]), (epoch_num + 1))
            tb_writer.add_scalar('val_avg_spearmanr_epoch', float(metrics["avg"]["spearmanr"]), (epoch_num + 1))

            train_loss_epoch = 0
            train_profile_loss_epoch = 0
            train_counts_loss_epoch = 0

        if val_loss < min_loss:
            min_loss = val_loss
            if args.local_rank in [-1, 0]:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "lowest_loss_checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                save_bpbert_model(output_dir, model, tokenizer, kmer_dict, args)
            early_stop_count = 0

        if epoch_val_avg_auprc > max_auprc:
            max_auprc = epoch_val_avg_auprc
            if args.local_rank in [-1, 0]:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "max_auprc_checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                save_bpbert_model(output_dir, model, tokenizer, kmer_dict, args)
            early_stop_count = 0
            

        if args.local_rank in [-1, 0]:
            logger.info('Early stop count '+str(early_stop_count)+'.'+'Global step '+str(global_step + 1)+'.')
            
        if early_stop_count >= args.early_stop_patience:
            if args.local_rank in [-1, 0]:
                logger.info('\#Model is not improving, so we halt the training session.')
            return

            



def main():
    parser = argparse.ArgumentParser()

    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    ## Required parameters
    parser.add_argument("--hdf5_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data path. Should be a .hdf5 file for tasks.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")

    parser.add_argument("--output_dir",
                       default='./results/result-tokenlevel-{}'.format(now_time),
                       type=str,
                       help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--n_tasks", type=int, required=True, help="Number of tasks.")

    ## Other parameters
    parser.add_argument("--multift",
                        action='store_true',
                        help="True for multi-task fine tune")

    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--log_steps", type=int, default=200,
                        help="log loss every X updates steps.")

    parser.add_argument('--comment',
                        type=str,
                        default="",
                        help="The comment for log dir")
    parser.add_argument("--kmer_list", type=str, default="kmer.txt")
    parser.add_argument("--kmer_len_list", type=str, default='4_5_6_7_8')
    parser.add_argument("--max_kmer_in_sequence", type=int, default=900)
    parser.add_argument("--from_pretrained", action='store_true')

    parser.add_argument("--early_stop_patience", type=int, default=3,
                        help="Early stop patience.")
    parser.add_argument("--freeze_params", action='store_true')
    parser.add_argument("--ckpt_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--continue_train", action='store_true')

    args = parser.parse_args()
    import os


    if args.multi_gpu:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        MASTER_ADDR = os.environ["MASTER_ADDR"]
        MASTER_PORT = os.environ["MASTER_PORT"]
        LOCAL_RANK = os.environ["LOCAL_RANK"]
        RANK = os.environ["RANK"]
        WORLD_SIZE = os.environ["WORLD_SIZE"]
        
        print("MASTER_ADDR: {}\tMASTER_PORT: {}".format(MASTER_ADDR, MASTER_PORT))
        print("LOCAL_RANK: {}\tRANK: {}\tWORLD_SIZE: {}".format(LOCAL_RANK, RANK, WORLD_SIZE))
    else:
        args.local_rank = -1
    
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        filemode='w',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    
    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda",args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        

    logging.info("device: {} n_gpu: {}, distributed training: {}".format(
        args.device, args.n_gpu, bool(args.local_rank != -1)))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))


    # Set seed
    set_seed(args)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        print("Output directory already exists and is not empty.")
    if args.local_rank == 0:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    # 等待主进程完成目录创建
    if args.local_rank != -1:
        torch.distributed.barrier()


    # Prepare model tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    kmer_dict = bpBERTKmerDict(args.kmer_list, tokenizer=tokenizer,max_kmer_in_seq=args.max_kmer_in_sequence)
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))
    head_names = ['profile', 'counts']

    if args.from_pretrained or args.continue_train:
        model = MutitaskModel.from_pretrained(args.bert_model, args.n_tasks, head_names,
                                cache_dir=cache_dir,
                                multift=args.multift)
    else:
        # 随机初始化模型参数
        config = bpBERTConfig(10, 8735, num_hidden_layers=12, num_hidden_word_layers=8)
        model = MutitaskModel(config, args.n_tasks, head_names)
    
    
    if args.freeze_params:
        unfreeze_layers = ['layer.11', 'layer.10', 'layer.9', 'layer.8','layer.7', 'layer.6', 'word_layers.7', 'word_layers.6', 'pooler', 'heads_layer']

        for name ,param in model.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
       
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("total_num: ", total_num)
        print("trainable_num: ", trainable_num)

    model.to(args.device)

    if args.do_train:
        train(args, model, tokenizer, kmer_dict)
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        final_metrics = evaluate(args, model, tokenizer, kmer_dict)
        logging.info("Done!")
        print("-" * 40)
        print("Final metrics: ")
        print(json.dumps(final_metrics, indent=2))

    print("Done!")

if __name__ == "__main__":
    main()