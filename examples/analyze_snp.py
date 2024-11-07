
import sys
sys.path.append("..")

import logomaker
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import pandas as pd
import logging
import math
from random import shuffle
import torch
from torch.distributions.kl import kl_divergence
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from torch.utils.data import Dataset, RandomSampler
from tqdm import tqdm, trange
from copy import deepcopy
from kipoi_utils.data_utils import numpy_collate_concat

from bpBERT.tokenization import BertTokenizer
from bpBERT.optimization import BertAdam, WarmupLinearSchedule
from bpBERT.kmer_utils import bpBERTKmerDict, NGRAM_DICT_NAME
from bpBERT.modeling import MutitaskModel, bpBERTConfig
from bpBERT.file_utils import WEIGHTS_NAME, CONFIG_NAME, PYTORCH_PRETRAINED_BERT_CACHE
from bpBERT.functions import softmax, mean

logger = logging.getLogger(__name__)


class SNPDataset(Dataset):
    def __init__(self,examples, max_seq_length, tokenizer, kmer_dict):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.kmer_dict = kmer_dict
        self.num_samples = len(examples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        seq = self.examples[item]
        features = convert_seq_to_features(seq=seq, max_seq_length=self.max_seq_length, 
                                         tokenizer=self.tokenizer, kmer_dict=self.kmer_dict, kmer_len_list=[4,5,6,7,8])
        return features
    
class SNPFeaturesDataset(Dataset):
    def __init__(self,features):
        self.features = features
        self.num_samples = len(features)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return self.features[item]
    
def convert_seq_to_features_cancel_match(cancel_pos, seq, max_seq_length, tokenizer, kmer_dict,kmer_len_list):
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

    # padding
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
            
    # ----------- code for kmer BEGIN-----------
    kmer_matches = []
    #  Filter the kmer segment from 4 to 8 to check whether there is a kmer
    for p in kmer_len_list:
        for q in range(0, len(ntokens) - p + 1):
            character_segment = ntokens[q:q + p]
            kmer_covered = range(q, q+p)
            if cancel_pos + 1 in kmer_covered:
                continue
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
    # kmer_seg_ids = []

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
        # print(kmer_positions[i],kmer_positions[i] + kmer_lengths[i])
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



def mutate(seq, start, end, target=None):
    """
    Mutate input sequence at specified position.
    
    If target is not None, returns the mutated seq. Otherwise, returns a numpy array with shape (4,1)
    with all four mutated possibilities.
    
    Arguments:
    seq -- str, original sequence.
    start -- int, starting index where nucleotide needs to be changed. Counting starts at zero.
    end -- int, ending index where nucleotide needs to be changed. Counting starts at zero.
    
    Keyword arguments:
    target -- str, the target nucleotide(s) to be changed to (default: None).
    
    Returns:
    mutated_seq -- str, mutated sequence.

    """
    assert end >= start and start >= 0 and end <= len(seq), "Wrong start and end index input."
    
    if target is not None:
        mutated_seq = seq[:start] + str(target) + seq[end:]
    else:
        mutated_seq = []
        for n in ['a','c','g','t']:
            m_seq = seq[:start] + str(n) + seq[end:]
            mutated_seq.append(m_seq)
    return mutated_seq

def log_fold_change(logits1, logits2):
    max_logits2 = np.max(logits2, axis=1)
    max_logits1 = np.max(logits1, axis=1)
    
    log_fold_change = np.log(max_logits2 / max_logits1)

    return log_fold_change

def convert_seq_to_features(seq, max_seq_length, tokenizer, kmer_dict,kmer_len_list):
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

    # padding
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
        # print(kmer_positions[i],kmer_positions[i] + kmer_lengths[i])
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

def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        # print(layer_attention.size())
        squeezed.append(layer_attention.squeeze(0))

    return torch.stack(squeezed)

def format_seqlogo_data(seq, score):
    seq = seq.upper()
    seq_data = pd.DataFrame(0, index=range(len(seq)), columns=['A', 'C', 'G', 'T'])

    for i, base in enumerate(seq):
        seq_data.at[i, base] = score[i]
    return seq_data

def predict(args, seq_list, model, tokenizer, kmer_dict):
    dataset = SNPDataset(seq_list, max_seq_length=args.max_seq_length,
                                   tokenizer=tokenizer, kmer_dict=kmer_dict)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

    # Eval!
    logger.info("***** Running predict *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.batch_size)

    model.eval()
    lpreds = []
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_batch = tuple(t.to('cuda') for t in batch)
        input_ids, input_mask, kmer_ids, kmer_positions, \
        kmer_positions_matrix, kmer_lengths, kmer_masks = input_batch


        with torch.no_grad():
            logits, all_attentions, _, _ = model(input_ids=input_ids,
                input_kmer_ids=kmer_ids,
                kmer_position_matrix=kmer_positions_matrix,
                kmer_position_ids=kmer_positions,
                kmer_type_ids=kmer_lengths,
                attention_mask=input_mask,
                kmer_attention_mask=kmer_masks)
        assert isinstance(logits, dict)


        # Delete [CLS] and [SEP]
        indexes = [0, -1]
        logits['profile'] = np.delete(logits['profile'], indexes, axis=1)

        lpreds.append(logits)

    preds = numpy_collate_concat(lpreds)


    return preds, all_attentions


def predict_cancel(args, features, model, tokenizer, kmer_dict):
    dataset = SNPFeaturesDataset(features)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

    # Eval!
    logger.info("***** Running predict *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.batch_size)

    model.eval()
    lpreds = []
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_batch = tuple(t.to('cuda') for t in batch)
        input_ids, input_mask, kmer_ids, kmer_positions, \
        kmer_positions_matrix, kmer_lengths, kmer_masks = input_batch


        with torch.no_grad():
            logits, all_attentions, _, _ = model(input_ids=input_ids,
                input_kmer_ids=kmer_ids,
                kmer_position_matrix=kmer_positions_matrix,
                kmer_position_ids=kmer_positions,
                kmer_type_ids=kmer_lengths,
                attention_mask=input_mask,
                kmer_attention_mask=kmer_masks)
        assert isinstance(logits, dict)

        indexes = [0, -1]
        logits['profile'] = np.delete(logits['profile'], indexes, axis=1)

        lpreds.append(logits)
    preds = numpy_collate_concat(lpreds)


    return preds, all_attentions




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--data_file", type=str, required=True, help="Input data file.")
    parser.add_argument("--output_dir",
                       default='./out',
                       type=str,
                       help="The output directory.")
    parser.add_argument("--n_tasks", type=int, required=True, help="Number of tasks.")
    parser.add_argument("--rsid", type=str, required=True, help="SNP's rsid")
    parser.add_argument("--alt", type=str, required=True, help="SNP's alt")
    parser.add_argument("--task_id", type=int, required=True, help="Task's id")
    parser.add_argument("--kmer_list", type=str, default="kmer.txt")
    parser.add_argument("--kmer_len_list", type=str, default='4_5_6_7_8')
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for predict.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_kmer_in_sequence", type=int, default=900)


    args = parser.parse_args()

    head_names = ['profile', 'counts']
    model = MutitaskModel.from_pretrained(args.bert_model, args.n_tasks, head_names, output_attentions=True)
    model.to('cuda')

    # Prepare model tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    kmer_dict = bpBERTKmerDict(args.kmer_list, tokenizer=tokenizer,max_kmer_in_seq=args.max_kmer_in_sequence)
    
    df = pd.read_csv(args.data_file, sep='\t')
    row = df[(df['rsid'] == args.rsid) & (df['alt'] == args.alt)]

    seq_orig = row['seq_orig'].values[0]
    preds_orig, all_attentions = predict(args, [seq_orig], model, tokenizer, kmer_dict)

    # Cancel k-mers matching to each base
    features_cancel = []
    for i in range(len(seq_orig)):
        f = convert_seq_to_features_cancel_match(cancel_pos=i, seq=seq_orig, max_seq_length=args.max_seq_length, 
                                         tokenizer=tokenizer, kmer_dict=kmer_dict, kmer_len_list=[4,5,6,7,8])
        features_cancel.append(f)
   
    preds_cancel, _ = predict_cancel(args, features_cancel, model, tokenizer, kmer_dict)
    # Compute fold change
    logits_orig_ = preds_orig['profile'][:, :, 0, args.task_id]
    logits_cancel = preds_cancel['profile'][:, :, 0, args.task_id]
    logits_orig_ = np.repeat(logits_orig_, logits_cancel.shape[0], axis=0)
   
    diff_cancel = log_fold_change(logits_orig_, logits_cancel)
    diff_cancel = np.absolute(diff_cancel)


    # Compute attention scores for the original sequence
    attn = format_attention(all_attentions)
    start = 0
    end = 11
    all_score = []
    for i in range(1, 1001):
        all_score.append(float(attn[start:end+1,:,0,i].sum()))
    all_score = np.array(all_score)
    diff_cancel_plus_attn = diff_cancel + all_score
    
    # Mutate each base in the original sequence
    seq_mut_list = []
    for j in range(len(seq_orig)):
        mut_seqs = mutate(seq_orig, j, j+1)
        seq_mut_list.extend(mut_seqs)

    # Predict mutant
    preds_mut, _ = predict(args, seq_mut_list, model, tokenizer, kmer_dict)

    # Compute fold change
    logits_orig = preds_orig['profile'][:, :, 0, args.task_id]
    logits_mut = preds_mut['profile'][:, :, 0, args.task_id]
    logits_orig = np.repeat(logits_orig, logits_mut.shape[0], axis=0)
    diff = log_fold_change(logits_orig, logits_mut)
    diff = diff.reshape((logits_mut.shape[1], 4)).T

    # Save the results
    np.save(f'{args.output_dir}/diff_{args.rsid}-{args.alt}-task{args.task_id}.npy', diff)
    np.save(f'{args.output_dir}/attn_{args.rsid}-{args.alt}-task{args.task_id}.npy', all_score)
    np.save(f'{args.output_dir}/diff_cancel_{args.rsid}-{args.alt}-task{args.task_id}.npy', diff_cancel)
    np.save(f'{args.output_dir}/diff_cancel_plus_attn_{args.rsid}-{args.alt}-task{args.task_id}.npy', diff_cancel_plus_attn)
    with open(f'{args.output_dir}/seq_orig_{args.rsid}-{args.alt}-task{args.task_id}.txt', 'w') as f:
        f.write(seq_orig + '\n')

    

if __name__ == "__main__":
    main()
