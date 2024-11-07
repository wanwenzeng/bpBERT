import sys
sys.path.append("..")

import argparse
import numpy as np
import multiprocessing
from collections import OrderedDict
import random
import datetime
import h5py

from bpBERT.intervals import load_intervals
from bpBERT.extract import extract_fasta, extract_bigwig, extract_bigwig_stranded

from pybedtools import BedTool, Interval

import warnings
warnings.filterwarnings("ignore")

def extract_batch(args, target_wigs, intervals):
    # extract seqs and profile
    pool = multiprocessing.Pool(args.n_processes)
    bstart = 0
    seqs = []
    profile = []
    final_intervals = []
    while bstart < len(intervals):
        bend = bstart + args.extract_batch_size

        # dropout
        if bend > len(intervals):
            break

        final_intervals.extend(intervals[bstart:bend])

        # extract seqs
        seqs_batch = extract_fasta(args.fasta_file, intervals[bstart:bend])
        seqs.append(seqs_batch)

        # bigwig_extract parameters
        ex_params = [(wig_files, intervals[bstart:bend]) for task, wig_files in target_wigs.items() if task != "bias_input"]

        # extract bigwigs
        profile_batch = np.array(pool.starmap(extract_bigwig_stranded, ex_params))
        profile_batch = np.transpose(profile_batch, axes=(1,2,3,0))
        profile.append(profile_batch)

        # update batch
        bstart = bend

    pool.close()

    # stack arrays
    if len(seqs) > 0:
        seqs = np.hstack(seqs) # (num_seqs,)
    else:
        seqs = seqs[0]
    
    if len(profile) > 0:
        profile = np.vstack(profile) # (num_seqs, seq_length, 2, num_tfs)
    else:
        profile = profile[0]

    # clean DNA seqs
    seqs_clean = []
    profile_clean = []
    final_intervals_clean = []
    for i, seq in enumerate(seqs):
        if set(seq) <= set('atgc'):
            seqs_clean.append(seq)
            profile_clean.append(profile[i])
            final_intervals_clean.append(final_intervals[i])
    
    seqs_clean = np.array(seqs_clean, dtype=object)
    profile_clean = np.array(profile_clean)

    # Add total number of counts
    total_count_transform = lambda x: np.log(1 + x)
    counts_clean = total_count_transform(profile_clean.sum(1)) # (num_seqs, 2, num_tfs)

    assert(seqs_clean.shape[0] == profile_clean.shape[0])
    assert(seqs_clean.shape[0] == len(final_intervals_clean))
    assert(seqs_clean.shape[0] == counts_clean.shape[0])

    return seqs_clean, profile_clean, counts_clean, final_intervals_clean


def append_hdf5_dataset(dataset, new_data):
    existing_data_length = len(dataset)
    dataset.resize(existing_data_length + len(new_data), axis=0)
    dataset[existing_data_length:] = new_data


def append_hdf5(args, target_wigs, intervals):
    seqs_clean, profile_clean, counts_clean, final_intervals_clean = extract_batch(args, target_wigs, intervals)

    val_indexes = [i for i in range(len(final_intervals_clean)) if final_intervals_clean[i][0] in args.val_chr]
    test_indexes = [i for i in range(len(final_intervals_clean)) if final_intervals_clean[i][0] in args.test_chr]
    exclude_indexes = [i for i in range(len(final_intervals_clean)) if final_intervals_clean[i][0] in args.exclude_chr]
    train_indexes = list(set(range(len(final_intervals_clean))) - set(val_indexes) - set(test_indexes) - set(exclude_indexes))

    assert(len(val_indexes) + len(test_indexes) + len(train_indexes) +len(exclude_indexes) == len(final_intervals_clean))

    random.shuffle(train_indexes)
    random.shuffle(val_indexes)
    random.shuffle(test_indexes)

    with h5py.File(args.hdf5_file, 'a') as f:
        train_seqs = f['train_seqs']
        train_profile = f['train_profile']
        train_counts = f['train_counts']

        validate_seqs = f['validate_seqs']
        validate_profile = f['validate_profile']
        validate_counts = f['validate_counts']

        test_seqs = f['test_seqs']
        test_profile = f['test_profile']
        test_counts = f['test_counts']

        append_hdf5_dataset(train_seqs, seqs_clean[train_indexes])
        append_hdf5_dataset(train_profile, profile_clean[train_indexes])
        append_hdf5_dataset(train_counts, counts_clean[train_indexes])

        append_hdf5_dataset(validate_seqs, seqs_clean[val_indexes])
        append_hdf5_dataset(validate_profile, profile_clean[val_indexes])
        append_hdf5_dataset(validate_counts, counts_clean[val_indexes])

        append_hdf5_dataset(test_seqs, seqs_clean[test_indexes])
        append_hdf5_dataset(test_profile, profile_clean[test_indexes])
        append_hdf5_dataset(test_counts, counts_clean[test_indexes])

        f.close()


def main():
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    parser = argparse.ArgumentParser()
    parser.add_argument("--beds_file",type=str,required=True,help="The list of peaks .bed files")
    parser.add_argument("--bws_file",type=str,required=True,help="The list of profile .bw files")
    parser.add_argument("--fasta_file",type=str,required=True,help="The reference genome .fasta file")
    parser.add_argument("--hdf5_file", type=str,required=True,help="The output file")
    parser.add_argument("--max_shift",type=int,default=0,help="The max shift of the peaks summit")
    parser.add_argument("--bin_length",type=int,default=200,help="The length of bins")
    parser.add_argument("--seq_length",type=int,default=1000,help="The length of input sequences")
    parser.add_argument("--n_processes",type=int,default=4,help="Number parallel processes to load data")
    parser.add_argument("--hdf5_batch_size",type=int,default=5000,help="Batch size to write to .hdf5 file")
    parser.add_argument("--extract_batch_size",type=int,default=100,help="Batch size to extract profile")
    parser.add_argument("--val_chr", nargs='+',default=['chr2', 'chr3', 'chr4'],help="Chromosomes for validation")
    parser.add_argument("--test_chr", nargs='+',default=['chr1', 'chr8', 'chr9'],help="Chromosomes for test")
    parser.add_argument("--exclude_chr", nargs='+',default=[],help="Chromosomes to exclude")
    args = parser.parse_args()

    
	# prepare intervals
    intervals = load_intervals(args.beds_file, max_shift=args.max_shift, width=args.seq_length, exclude_chr=args.exclude_chr)

    print("load intervals done!")


    # prepare bigwigs
    target_wigs = OrderedDict()
    for line in open(args.bws_file, encoding='UTF-8'):
        line = line.rstrip().split('\t')
        target_wigs[line[0]] = [line[1], line[2]]
    n_tasks = len(target_wigs)
    

    # Create hdf5 dataset and write the first batch
    bstart_0 = 0
    if len(intervals) < args.hdf5_batch_size:
        bend_0 = len(intervals)
    else:
        bend_0 = args.hdf5_batch_size
    intervals_batch_0 = intervals[bstart_0:bend_0]
    seqs_clean_0, profile_clean_0, counts_clean_0, final_intervals_clean_0 = extract_batch(args, target_wigs, intervals_batch_0)

    val_indexes_0 = [i for i in range(len(final_intervals_clean_0)) if final_intervals_clean_0[i][0] in args.val_chr]
    test_indexes_0 = [i for i in range(len(final_intervals_clean_0)) if final_intervals_clean_0[i][0] in args.test_chr]
    exclude_indexes_0 = [i for i in range(len(final_intervals_clean_0)) if final_intervals_clean_0[i][0] in args.exclude_chr]
    train_indexes_0 = list(set(range(len(final_intervals_clean_0))) - set(val_indexes_0) - set(test_indexes_0) - set(exclude_indexes_0))

    assert(len(val_indexes_0) + len(test_indexes_0) + len(train_indexes_0) +len(exclude_indexes_0) == len(final_intervals_clean_0))

    random.shuffle(train_indexes_0)
    random.shuffle(val_indexes_0)
    random.shuffle(test_indexes_0)

    hdf5_out = h5py.File(args.hdf5_file, 'w')
    dt = h5py.special_dtype(vlen=str)
    hdf5_out.create_dataset("train_seqs", data=seqs_clean_0[train_indexes_0], maxshape=[None,],chunks=True,compression='gzip', dtype=dt)
    hdf5_out.create_dataset("train_profile", data=profile_clean_0[train_indexes_0], maxshape=[None, args.seq_length, 2, n_tasks],chunks=True,compression='gzip')
    hdf5_out.create_dataset("train_counts", data=counts_clean_0[train_indexes_0], maxshape=[None, 2, n_tasks],chunks=True,compression='gzip')

    hdf5_out.create_dataset("validate_seqs", data=seqs_clean_0[val_indexes_0], maxshape=[None,],chunks=True,compression='gzip', dtype=dt)
    hdf5_out.create_dataset("validate_profile", data=profile_clean_0[val_indexes_0], maxshape=[None, args.seq_length, 2, n_tasks],chunks=True,compression='gzip')
    hdf5_out.create_dataset("validate_counts", data=counts_clean_0[val_indexes_0], maxshape=[None, 2, n_tasks],chunks=True,compression='gzip')

    hdf5_out.create_dataset("test_seqs", data=seqs_clean_0[test_indexes_0], maxshape=[None,],chunks=True,compression='gzip', dtype=dt)
    hdf5_out.create_dataset("test_profile", data=profile_clean_0[test_indexes_0], maxshape=[None, args.seq_length, 2, n_tasks],chunks=True,compression='gzip')
    hdf5_out.create_dataset("test_counts", data=counts_clean_0[test_indexes_0], maxshape=[None, 2, n_tasks],chunks=True,compression='gzip')

    hdf5_out.close()

    del seqs_clean_0
    del profile_clean_0
    del counts_clean_0

    # Loop to generate data
    bstart = bend_0
    lstart = 0
    while bstart < len(intervals):
        bend = bstart + args.hdf5_batch_size

        if bend > len(intervals):
            lstart = bstart
            break
        
        intervals_batch = intervals[bstart:bend]

        append_hdf5(args, target_wigs, intervals_batch)

        # update batch
        bstart = bend
    
    # Write the remaining data
    lend = len(intervals)
    if lend - lstart >= args.extract_batch_size and lstart > 0:
        append_hdf5(args, target_wigs, intervals[lstart:lend])

    with h5py.File(args.hdf5_file, 'r') as f:
        train_seqs = f['train_seqs']
        validate_seqs = f['validate_seqs']
        test_seqs = f['test_seqs']

        print("Train data: {}".format(len(train_seqs)))
        print("Validate data: {}".format(len(validate_seqs)))
        print("Test data: {}".format(len(test_seqs)))


    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    print(now_time)


if __name__ == '__main__':
    main()


    
