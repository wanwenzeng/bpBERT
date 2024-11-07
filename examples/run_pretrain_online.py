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
"""PyTorch pretrain for bpBERT model."""
import sys

sys.path.append("..")
from argparse import ArgumentParser
from pathlib import Path
import os
import torch
import logging
import json
import random
import numpy as np
from collections import namedtuple
import time
import datetime
import shelve
from tempfile import TemporaryDirectory

from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from bpBERT.file_utils import WEIGHTS_NAME, CONFIG_NAME
from bpBERT.modeling import bpBERTConfig, bpBERTForPreTraining
from bpBERT.tokenization import BertTokenizer
from bpBERT.optimization import BertAdam, WarmupLinearSchedule
from bpBERT.kmer_utils import bpBERTKmerDict, KMER_DICT_NAME



from tensorboardX import SummaryWriter
InputFeatures = namedtuple(
    "InputFeatures",
    "input_ids input_mask lm_label_ids msk_kmer_id_arr kmer_ids kmer_masks kmer_positions kmer_position_ids kmer_type_ids kmer_lengths")

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

def truncate_seq_pair_a(tokens_a, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a)
        if total_length <= max_num_tokens:
            return
        trunc_tokens = tokens_a 
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def create_masked_kmer_predictions(tokens, masked_lm_count, vocab_list, kmer_dict,kmer_len_list):
    """Creates the predictions for the masked LM objective. This is modified from the Google BERT repo, mask 5(masked_lm_count=5) bases in each 200bp pretrain sequence."""
    kmer_list = list(kmer_dict.kmer_to_id_dict)

    token_non = list(filter(lambda x : x != 'n', tokens))
    masked_lms = []
    covered_indexes = set()
    msk_token_arr = ['0' for i in range(len(tokens))]
    tokens_masked = tokens[:]
    tokens_masked_len = tokens[:]
    mask_center = []
    loop = 0

    while len(covered_indexes) < masked_lm_count and loop < 10:
        loop += 1
        index_center = random.randint(1,len(tokens)-1)
        covered_indexes.add(index_center)
        tokens_masked[index_center] = "[MASK]"
        tokens_masked_len[index_center] = random.randint(0,len(kmer_len_list)-1)

    mask_indices =[]
    masked_token_labels=[]

    for i in range(len(tokens)):
        if tokens_masked[i] == '[MASK]':
            mask_indices.append(i)
            masked_token_labels.append(tokens[i])

    return tokens_masked,tokens_masked_len, mask_indices, masked_token_labels

def convert_example_to_features(example, tokenizer, max_seq_length, max_kmer_in_sequence):
    # convert example to features, including padding and truncating
    tokens = example["tokens"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_token_label_ids = example["masked_lm_labels"]
    masked_token_label_ids = [int(x) for x in list(masked_token_label_ids)]
    masked_kmer_positions = example["masked_kmer_positions"]
    masked_kmer_labels = example["masked_kmer_labels"]
    msk_kmer_id_arr_array = np.full(max_kmer_in_sequence, dtype=int, fill_value=-1)
    msk_kmer_id_arr_array[masked_kmer_positions] = masked_kmer_labels

    kmer_ids = example["kmer_ids"]
    kmer_positions = example["kmer_positions"]
    kmer_lengths = example["kmer_lengths"]
    kmer_lengths = kmer_lengths[0]*[4]+kmer_lengths[1]*[5]+kmer_lengths[2]*[6]+kmer_lengths[3]*[7]+kmer_lengths[4]*[8]
    input_ids = np.array([int(x) for x in list(tokens)], dtype=int)
    
    assert len(input_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_array = np.zeros(max_seq_length, dtype=int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=bool)
    mask_array[:len(input_ids)] = 1

    lm_label_token_array = np.full(max_seq_length, dtype=int, fill_value=-1)
    lm_label_token_array[masked_lm_positions] = masked_token_label_ids
    if len(kmer_ids) > max_kmer_in_sequence:
        kmer_ids = kmer_ids[:max_kmer_in_sequence]
        kmer_positions = kmer_positions[:max_kmer_in_sequence]
        kmer_lengths = kmer_lengths[:max_kmer_in_sequence]

    # add kmer pads
    kmer_id_array = np.zeros(max_kmer_in_sequence, dtype=int)
    kmer_id_array[:len(kmer_ids)] = kmer_ids
    kmer_position_ids = np.zeros(max_kmer_in_sequence, dtype=int)
    kmer_position_ids[:len(kmer_ids)] = kmer_positions

    kmer_type_ids = np.zeros(max_kmer_in_sequence, dtype=int)
    kmer_type_ids[:len(kmer_ids)] = kmer_lengths
    # record the masked positions
    kmer_positions_matrix = np.zeros(shape=(max_seq_length, max_kmer_in_sequence), dtype=bool)

    for i in range(len(kmer_ids)):
        for index,item in enumerate([kmer_positions[i]]):
            kmer_positions_matrix[item:item+kmer_lengths[i], i] = 1

    kmer_length_array = np.zeros(max_kmer_in_sequence, dtype=np.int32)
    kmer_length_array[:len(kmer_ids)] = kmer_lengths
    kmer_mask_array = np.zeros(max_kmer_in_sequence, dtype=bool)
    kmer_mask_array[:len(kmer_ids)] = 1

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             lm_label_ids=lm_label_token_array,
                             msk_kmer_id_arr=msk_kmer_id_arr_array,
                             kmer_ids=kmer_id_array,
                             kmer_masks=kmer_mask_array,
                             kmer_positions=kmer_positions_matrix,
                             kmer_position_ids=kmer_position_ids,
                             kmer_type_ids=kmer_type_ids,
                             kmer_lengths=kmer_length_array,
                             )
    return features

class PretrainDataset(Dataset):
    # Pretrain Dataset, handling DNA sequences to input features for base encoder and matching k-mers to k-mer features for k-mer encoder.
    def __init__(self,local_rank,seq_len,kmer_dict, training_path, epoch, tokenizer,max_kmer_in_sequence, num_data_epochs,reduce_memory=False, fp16=False):
        self.vocab = tokenizer.vocab
        self.kmer_dict = kmer_dict
        self.tokenizer = tokenizer
        self.epoch = epoch
        with open(training_path, encoding='utf-8') as f2:
            contents = f2.readlines()

        kmer_len_list_str = '4_5_6_7_8'.split('_')
        kmer_len_list = []
        for item in kmer_len_list_str:
            kmer_len = int(item)
            kmer_len_list.append(kmer_len)
        self.kmer_len_list = kmer_len_list
        self.examples = contents
        self.data_epoch = epoch % num_data_epochs
        num_samples = len(self.examples)
        self.temp_dir = None
        self.working_dir = None
        self.fp16 = fp16
        self.reduce_memory = reduce_memory
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.max_kmer_in_sequence = max_kmer_in_sequence


    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, item):
        # Random offset is added to the input DNA sequence to make the training data of each epoch diverse
        offset = random.randint(-100,99)
        add_seq = ''
        seq_dict = self.examples
        idx= item
        idx_add_1= item+1
        idx_reduce_1= item-1
        selected_seq = self.examples[idx].strip()
        if offset>=0:
            if item+1<=len(seq_dict)-1:
                extra_seq = seq_dict[idx_add_1].strip()
                # print("extra_seq","".join(extra_seq))
                add_seq = extra_seq[:offset]
                
                selected_seq = selected_seq[offset:]+add_seq
            else:
                selected_seq = selected_seq
        else:
            if item-1>=0:
                
                extra_seq = seq_dict[idx_reduce_1].strip()
                # print("extra_seq","".join(extra_seq))
                if len(extra_seq)<200:
                    selected_seq = selected_seq
                else:
                    add_seq = extra_seq[offset:]
                
                    selected_seq = add_seq+selected_seq[:offset]
            else:
                selected_seq = selected_seq
        max_seq_length =self.seq_len
        # adding [CLS]
        selected_seq_token = ["[CLS]"] + list(selected_seq)
        if len(selected_seq_token)> max_seq_length:
            print("seq too long",len(selected_seq_token))
            selected_seq_token = selected_seq_token[:max_seq_length]
            print("seq cut to",len(selected_seq_token))

        masked_lm_count = 5
        tokens = selected_seq_token
        tokens_masked,tokens_masked_len, masked_lm_positions, masked_lm_labels = create_masked_kmer_predictions(
                    selected_seq_token, masked_lm_count, self.vocab,self.kmer_dict,self.kmer_len_list)
        kmer_ids = []
        kmer_positions = []
        kmer_lengths = []
        kmer_matches_all = []
        msk_kmer_id_arr = [-1 for i in range(self.kmer_dict.max_kmer_in_seq)]
        is_mask = False
        # Sliding window matching of sequences was performed using a window size of length 4 to 8, and the corresponding k-mer was matched in the kmer vocabulary
        for p in self.kmer_len_list:
            for q in range(0, len(tokens_masked) - p + 1):
                character_segment = tokens_masked[q:q + p]
                # j is the starting position of the kmer
                # i is the length of the current kmer
                character_segment = tuple(character_segment)
                if 'n' in character_segment or '[CLS]' in character_segment :
                        continue
                if character_segment in self.kmer_dict.kmer_to_id_dict:
                    is_mask = False
                    kmer_id = self.kmer_dict.kmer_to_id_dict[character_segment]
                    kmer_matches_all.append([kmer_id,q,p,is_mask,character_segment])

                elif '[MASK]' in character_segment:
                    masked_character_segment = tokens[q:q + p]
                    masked_character_segment = tuple(masked_character_segment)
                    is_mask = True
                    if masked_character_segment not in self.kmer_dict.kmer_to_id_dict:
                        masked_kmer_id = self.kmer_dict.kmer_to_id_dict["[unk]"]
                    else:
                        masked_kmer_id = self.kmer_dict.kmer_to_id_dict[masked_character_segment]
                    kmer_matches_all.append([masked_kmer_id,q,p,is_mask,masked_character_segment])
        
        # Some kmers are randomly removed so that the total number of kmers is less than or equal to max_kmer_in_seq
        idx_range = list(range(0,len(kmer_matches_all)))
        random.shuffle(idx_range)
        ignore_kmer = []
        if len(kmer_matches_all) > self.kmer_dict.max_kmer_in_seq:
            ignore_kmer=idx_range[self.kmer_dict.max_kmer_in_seq:]

        kmer_ids = []
        kmer_positions = []
        kmer_pos = ['0' for i in range((max_seq_length-1)*len(self.kmer_len_list))]
        kmer_lengths = [0 for i in range(len(self.kmer_len_list))]
        masked_kmer_positions = []
        masked_kmer_labels = []
        for index,item in enumerate(kmer_matches_all):
            if index not in ignore_kmer:
                item = list(item)
                if item[3]==True:
                    # masking k-mers that overlapped with masked bases
                    kmer_ids.append(self.kmer_dict.kmer_to_id_dict["[mask]"])
                    masked_kmer_positions.append(len(kmer_ids)-1)
                    masked_kmer_labels.append(item[0])
                else:
                    kmer_ids.append(item[0])
                kmer_positions.append(item[1])
                # kmer_lengths.append(item[2])
                kmer_len_list_idx = 0
                for i,k in enumerate(self.kmer_len_list):
                    if k == item[2]:
                        kmer_len_list_idx = i
                        kmer_lengths[i] = kmer_lengths[i] +1


        idx = [i%(max_seq_length-1) for i,x in enumerate(kmer_pos) if x == '1']
        kmer_pos = ''.join(kmer_pos)
        tokens_masked = self.tokenizer.convert_tokens_to_ids(tokens_masked)
        tokens_masked = ''.join([str(x) for x in tokens_masked])
        masked_lm_labels = self.tokenizer.convert_tokens_to_ids(masked_lm_labels)
        masked_lm_labels = ''.join([str(x) for x in masked_lm_labels])

        instance = {
            "tokens": tokens_masked,
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_labels": masked_lm_labels,
            "masked_kmer_positions": masked_kmer_positions,
            "masked_kmer_labels": masked_kmer_labels,
            "kmer_ids": kmer_ids,
            "kmer_positions": kmer_positions,
            "kmer_lengths": kmer_lengths,
        }


        features = convert_example_to_features(instance, self.tokenizer, self.seq_len, self.max_kmer_in_sequence)
        position = torch.tensor(features.kmer_positions.astype(np.double))
        if self.fp16:
            position = position.half()
        else:
            position = position.float()
        
        

        return (torch.tensor(features.input_ids.astype(np.int64)),
                torch.tensor(features.input_mask.astype(np.int64)),
                torch.tensor(features.lm_label_ids.astype(np.int64)),
                torch.tensor(features.msk_kmer_id_arr.astype(np.int64)),
                torch.tensor(features.kmer_ids.astype(np.int64)),
                torch.tensor(features.kmer_masks.astype(np.int64)),
                position,
                torch.tensor(features.kmer_position_ids.astype(np.int64)),
                torch.tensor(features.kmer_type_ids.astype(np.int64)),
                torch.tensor(features.kmer_lengths.astype(np.int64)),
                )
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        print("end")
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            deldir(self.temp_dir)
            print("To clean up")

def deldir(dir):
    # Recursively deletes empty folders under directories
    print(dir)
    import os
    if not os.path.exists(dir):
        return False
    if os.path.isfile(dir):
        os.remove(dir)
        return
    for i in os.listdir(dir):
        t = os.path.join(dir, i)
        if os.path.isdir(t):
            deldir(t)
        else:
            os.unlink(t)
    os.removedirs(dir)


def main():
    parser = ArgumentParser()
    parser.add_argument('--training_data', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True,
                        help="Bert pre-trained model selected in the list: bpBERT-base.")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")

    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    # parser.add_argument("--local_rank",
    #                     type=int,
    #                     default=-1,
    #                     help="local_rank for distributed training on gpus")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--scratch',
                        action='store_true',
                        help="Whether to train from scratch")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--learning_rate",
                        default=3.5e-5,
                        type=float,
                        help="The highest learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=16,
                        help="random seed for initialization")
    parser.add_argument('--save_name',
                        type=str,
                        default="bpBERT",
                        help="The prefix used for saving the remote model")
    parser.add_argument('--comment',
                        type=str,
                        default="",
                        help="The comment for log dir")
    parser.add_argument("--already_trained_epoch",
                        default=0,
                        type=int)
    parser.add_argument("--kmer_dict_size",
                        default=0,
                        type=int)
    parser.add_argument("--kmer_list", type=str, default="kmer.txt")
    parser.add_argument("--max_kmer_in_sequence", type=int, default=750)
    parser.add_argument("--max_seq_len", type=int, default=250)

    args = parser.parse_args()

    samples_per_epoch = [0]
    with open(args.training_data, 'r') as infile:
        infile_list = infile.readlines()
        samples_per_epoch[0] = len(infile_list)
    mask_lm_count_per_epoch = [5]
    num_data_epochs = args.epochs

    if args.multi_gpu:
        local_rank = int(os.environ["LOCAL_RANK"])
        MASTER_ADDR = os.environ["MASTER_ADDR"]
        MASTER_PORT = os.environ["MASTER_PORT"]
        LOCAL_RANK = os.environ["LOCAL_RANK"]
        RANK = os.environ["RANK"]
        WORLD_SIZE = os.environ["WORLD_SIZE"]
        print("MASTER_ADDR: {}\tMASTER_PORT: {}".format(MASTER_ADDR, MASTER_PORT))
        print("LOCAL_RANK: {}\tRANK: {}\tWORLD_SIZE: {}".format(LOCAL_RANK, RANK, WORLD_SIZE))
    else:
        local_rank = -1
    
    if local_rank == -1 or args.no_cuda:
        tb_writer = SummaryWriter(comment=args.comment)
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda",local_rank)
        n_gpu = 8
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        
    logging.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    total_train_examples = 0
    for i in range(args.epochs):
        # The module takes into account the fact that we may loop over limited epochs of data
        total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

    num_train_optimization_steps = int(
        total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)

    if local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.scratch:
        config = bpBERTConfig(vocab_size_or_config_json_file=10, word_vocab_size=args.kmer_dict_size,num_hidden_layers=2,num_hidden_word_layers=1)
        model = bpBERTForPreTraining(config,output_attentions = True,keep_multihead_output = True)
    else:
        print("Load from checkpoints")
        model = bpBERTForPreTraining.from_pretrained(args.bert_model)

    if args.fp16:
        model.half()
    model.to(device)

    if local_rank != -1:
        model = model.cuda(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank], 
             output_device=local_rank,find_unused_parameters=True)
        
        
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                            schedule='warmup_linear',
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    if local_rank in [-1,0]:
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", total_train_examples)
        logging.info("  Batch size = %d", args.train_batch_size)
        logging.info("  Num steps = %d", num_train_optimization_steps)
    model.train()
    log_loss = 0
    log_acc = 0
    log_acc_kmer = 0
    log_loss_lowest = 100000
    log_loss_list = []
    is_lower_14 = False
    decrease_lr_loss_list = []
    vocab_list = list(tokenizer.vocab.keys())
    kmer_dict = bpBERTKmerDict(args.kmer_list, tokenizer=tokenizer,max_kmer_in_seq=args.max_kmer_in_sequence)
    total = sum(p.numel() for p in model.parameters())
    if local_rank in [-1,0]:
        print("total param:",total)
        tb_writer = SummaryWriter(comment=args.comment)
    for epoch in range(args.epochs):
        if epoch < len(mask_lm_count_per_epoch):
            mask_lm_count = mask_lm_count_per_epoch[epoch]

        epoch_dataset = PretrainDataset(local_rank=local_rank,
                                            seq_len = args.max_seq_len,
                                            kmer_dict = kmer_dict,
                                            epoch=epoch,
                                            training_path=args.training_data,
                                            tokenizer=tokenizer,
                                            max_kmer_in_sequence =args.max_kmer_in_sequence,
                                            num_data_epochs=num_data_epochs,
                                            reduce_memory=args.reduce_memory,
                                            fp16=args.fp16
                                            )
        if local_rank == -1:
            train_sampler = RandomSampler(epoch_dataset)
        else:
            train_sampler = DistributedSampler(epoch_dataset)
        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, num_workers=8, pin_memory=True, batch_size=args.train_batch_size)

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
            log_loss_all = 0
            log_loss_token_all = 0
            log_loss_kmer_all = 0
            log_acc_all = 0
            log_acc_kmer_all = 0
            lr_pre = args.learning_rate
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, lm_label_ids,msk_kmer_id_arr, kmer_ids, kmer_masks,  kmer_positions, kmer_position_ids,kmer_type_ids, \
                kmer_lengths = batch

                loss,loss_token,loss_kmer,acc_mlm,acc_kmer, = model(
                             input_ids=input_ids,
                             input_kmer_ids=kmer_ids,
                             kmer_position_matrix=kmer_positions,
                             kmer_position_ids=kmer_position_ids,
                             kmer_type_ids=kmer_type_ids,
                             attention_mask=input_mask,
                             kmer_attention_mask=kmer_masks,
                             masked_lm_labels=lm_label_ids,
                             msk_kmer_id_arr=msk_kmer_id_arr,
                             kmer_dict=kmer_dict,
                             output_dir=args.output_dir)

                if n_gpu > 1:
                    loss = loss.mean().view(-1)  # mean() to average on multi-gpu.
                    loss_token = loss_token.mean()
                    loss_kmer = loss_kmer.mean()
                    acc_mlm = acc_mlm.mean()
                    acc_kmer = acc_kmer.mean()

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()



                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                pbar.update(1)
                mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
                if local_rank in [-1,0]:
                    tb_writer.add_scalar('mean_loss', mean_loss, global_step)
                    tb_writer.add_scalar('batch_loss', loss.item(), global_step)
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 and local_rank in [-1,0]:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        tb_writer.add_scalar('lr', lr_this_step, global_step)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    l = optimizer.step()
                    optimizer.zero_grad()

                    log_step_len = 100
                    if (step+1)%log_step_len == 0:
                        log_loss_all += loss.item()
                        log_loss_token_all += loss_token.item()
                        log_loss_kmer_all += loss_kmer.item()
                        log_acc_all += acc_mlm
                        log_acc_kmer_all += acc_kmer
                        log_kmer_loss = log_loss_kmer_all/log_step_len
                        log_token_loss = log_loss_token_all/log_step_len
                        log_loss = log_loss_all/log_step_len
                        log_acc = log_acc_all/log_step_len
                        log_acc_kmer = log_acc_kmer_all/log_step_len
                        log_loss_list.append(log_loss)
                        log_loss_all = 0
                        log_loss_kmer_all = 0
                        log_loss_token_all = 0
                        log_acc_all = 0
                        log_acc_kmer_all = 0
                        log_step = (global_step+1)/log_step_len
                        if local_rank in [-1,0]:
                            tb_writer.add_scalar('log_loss', log_loss, log_step)
                            tb_writer.add_scalar('log_token_loss', log_token_loss, log_step)
                            tb_writer.add_scalar('log_kmer_loss', log_kmer_loss, log_step)
                            tb_writer.add_scalar('log_acc_mlm', log_acc, log_step)
                            tb_writer.add_scalar('log_acc_kmer', log_acc_kmer, log_step)

                            logging.info("** ** * log_loss***   "+str(log_loss)+"   log_step:"+str(log_step)+"  ** ** * ") 
                            logging.info("** ** * log_acc_mlm***    "+str(log_acc)+"   log_step:"+str(log_step)+"  ** ** * ") 
                            logging.info("** ** * log_acc_kmer_mlm***  "+str(log_acc_kmer)+"   log_step:"+str(log_step)+"  ** ** * ") 

                    else:
                        log_loss_all += loss.item()
                        log_loss_token_all += loss_token.item()
                        log_loss_kmer_all += loss_kmer.item()
                        log_acc_all += acc_mlm
                        log_acc_kmer_all += acc_kmer


                    if local_rank in [-1,0]:

                        if (global_step+1)%2000 == 0:
                            # Save a trained model
                            ts = time.time()
                            st = datetime.datetime.fromtimestamp(ts).strftime('%m%d%H%M%S')
                            saving_path_i = args.output_dir
                            saving_path_i = Path(os.path.join(saving_path_i, 'checkpoint' +str(global_step+1) + "_lr_" + str(lr_pre)+ st + "_loss_" + str(log_loss)+ "_" + str(st)))
                            if saving_path_i.is_dir() and list(saving_path_i.iterdir()):
                                logging.warning(f"Output directory ({ saving_path_i }) already exists and is not empty!")
                            saving_path_i.mkdir(parents=True, exist_ok=True)

                            logging.info("** ** * Saving pretrained model ** ** * ")
                            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

                            output_model_file = os.path.join(saving_path_i, WEIGHTS_NAME)
                            output_config_file = os.path.join(saving_path_i, CONFIG_NAME)

                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)
                            tokenizer.save_vocabulary(saving_path_i)

                    global_step += 1
                    if local_rank in [-1,0]:
                        lr_pre = optimizer.get_lr()[0]
                        tb_writer.add_scalar('lr_pre', lr_pre, global_step)

        # Save a trained model
        if local_rank in [-1,0]:
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%m%d%H%M%S')
            saving_path = args.output_dir
            saving_path = Path(os.path.join(saving_path, args.save_name + st + "_epoch_" + str(epoch + args.already_trained_epoch)+'_checkpoint' +str(global_step+1) + "_lr_" + str(lr_pre)+ st + "_loss_" + str(log_loss)))
            if saving_path.is_dir() and list(saving_path.iterdir()):
                logging.warning(f"Output directory ({ saving_path }) already exists and is not empty!")
            saving_path.mkdir(parents=True, exist_ok=True)
            logging.info("** ** * Saving pretrained model ** ** * ")
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

            output_model_file = os.path.join(saving_path, WEIGHTS_NAME)
            output_config_file = os.path.join(saving_path, CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(saving_path)

if __name__ == '__main__':
    main()