# motif finding code, modified from DNABERT-viz
import os
import pandas as pd
import numpy as np
import argparse
import motif_utils as utils
from collections import OrderedDict



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the sequence+label .tsv files (or other data files) for the task.",
    )

    parser.add_argument(
        "--predict_dir",
        default=None,
        type=str,
        help="Path where the attention scores were saved. Should contain both pred_results.npy and atten.npy",
    )

    parser.add_argument(
        "--window_size",
        default=24,
        type=int,
        help="Specified window size to be final motif length",
    )

    parser.add_argument(
        "--min_len",
        default=5,
        type=int,
        help="Specified minimum length threshold for contiguous region",
    )


    parser.add_argument(
        "--min_n_motif",
        default=3,
        type=int,
        help="Minimum instance inside motif to be filtered",
    )

    parser.add_argument(
        "--first_bed",
        type=int,
        help="Minimum instance inside motif to be filtered",
    )

    parser.add_argument(
        "--select_first",
        action='store_true'
    )

    parser.add_argument(
        "--align_all_ties",
        action='store_true',
        help="Whether to keep all best alignments when ties encountered",
    )

    parser.add_argument(
        "--save_file_dir",
        default='.',
        type=str,
        help="Path to save outputs",
    )

    parser.add_argument(
        "--tf",
        default='',
        type=str
    )

    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Verbosity controller",
    )

    parser.add_argument(
        "--return_idx",
        action='store_true',
        help="Whether the indices of the motifs are only returned",
    )

    # TODO: add the conditions
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
    tf_sign = tf_dict[args.tf][0]
    print(tf_sign)
    args.predict_dir = f'{args.data_dir}/{tf_dict[args.tf][-1]}/{tf_dict[args.tf][-2]}_{args.tf}/attention'
    args.save_file_dir = f'./high_peak_attn_motifs/motif_{args.save_file_dir}/{tf_dict[args.tf][-1]}_{tf_dict[args.tf][-2]}_{args.tf}/'
    print('predict_dir',args.predict_dir)
    print('save_file_dir',args.save_file_dir)

    # atten_scores = np.load(os.path.join(args.predict_dir,"avg_cls_attention.npy"))
    atten_scores = np.load(os.path.join(args.predict_dir,"avg_cls_attention_peak.npy"))
    # atten_scores = np.load(os.path.join(args.predict_dir,"avg_profile_attention.npy"))
    # atten_scores = atten_scores[:,:,tf_sign]
    # pred_profile = np.load(os.path.join(args.predict_dir,"all_pred_profile.npy"))
    # pred_counts = np.load(os.path.join(args.predict_dir,"all_pred_counts.npy"))
    # gt_profile = np.load(os.path.join(args.predict_dir,"all_gt_profile.npy"))
    # gt_counts = np.load(os.path.join(args.predict_dir,"all_gt_counts.npy"))
    all_input_ids = np.load(os.path.join(args.predict_dir,"all_input_ids.npy"))
    # atten_scores = np.array([a+b for a,b in zip(atten_scores,atten_scores_peak)])
    print('atten_scores',atten_scores.shape)

    if args.select_first:
        atten_scores = atten_scores[:args.first_bed]
        # pred_profile = pred_profile[:args.first_bed]
        # gt_profile = gt_profile[:args.first_bed]
        all_input_ids = all_input_ids[:args.first_bed]


    # # train_path = os.path.join(args.predict_dir, 'train')
    # # test_path = os.path.join(args.predict_dir, 'test')
    # # validate_path = os.path.join(args.predict_dir, 'validate')

    # # preparing avg_cls_attention.npy
    # train_atten_scores = np.load(os.path.join(train_path, "avg_cls_attention.npy"))
    # test_atten_scores = np.load(os.path.join(test_path, "avg_cls_attention.npy"))
    # validate_atten_scores = np.load(os.path.join(validate_path, "avg_cls_attention.npy"))
    # atten_scores = np.concatenate((train_atten_scores, test_atten_scores, validate_atten_scores), axis=0)
    # # atten_scores = np.concatenate((test_atten_scores, validate_atten_scores), axis=0)

    # # preparing all_pred_profile.npy
    # train_pred_profile = np.load(os.path.join(train_path, "all_pred_profile.npy"))
    # test_pred_profile = np.load(os.path.join(test_path, "all_pred_profile.npy"))
    # validate_pred_profile = np.load(os.path.join(validate_path, "all_pred_profile.npy"))
    # pred_profile = np.concatenate((train_pred_profile, test_pred_profile, validate_pred_profile), axis=0)
    # # pred_profile = np.concatenate((test_pred_profile, validate_pred_profile), axis=0)

    # # # preparing all_pred_counts.npy
    # # train_pred_counts = np.load(os.path.join(train_path, "all_pred_counts.npy"))
    # # test_pred_counts = np.load(os.path.join(test_path, "all_pred_counts.npy"))
    # # validate_pred_counts = np.load(os.path.join(validate_path, "all_pred_counts.npy"))
    # # pred_counts = np.concatenate((train_pred_counts, test_pred_counts, validate_pred_counts), axis=0)
    # # # pred_counts = np.concatenate((test_pred_counts, validate_pred_counts), axis=0)

    # # preparing all_gt_profile.npy
    # train_gt_profile = np.load(os.path.join(train_path, "all_gt_profile.npy"))
    # test_gt_profile = np.load(os.path.join(test_path, "all_gt_profile.npy"))
    # validate_gt_profile = np.load(os.path.join(validate_path, "all_gt_profile.npy"))
    # gt_profile = np.concatenate((train_gt_profile, test_gt_profile, validate_gt_profile), axis=0)
    # # gt_profile = np.concatenate((test_gt_profile, validate_gt_profile), axis=0)

    # # # preparing all_gt_counts.npy
    # # train_gt_counts = np.load(os.path.join(train_path, "all_gt_counts.npy"))
    # # test_gt_counts = np.load(os.path.join(test_path, "all_gt_counts.npy"))
    # # validate_gt_counts = np.load(os.path.join(validate_path, "all_gt_counts.npy"))
    # # gt_counts = np.concatenate((train_gt_counts, test_gt_counts, validate_gt_counts), axis=0)
    # # # gt_counts = np.concatenate((test_gt_counts, validate_gt_counts), axis=0)

    # # preparing all_input_ids.npy
    # train_input_ids = np.load(os.path.join(train_path, "all_input_ids.npy"))
    # test_input_ids = np.load(os.path.join(test_path, "all_input_ids.npy"))
    # validate_input_ids = np.load(os.path.join(validate_path, "all_input_ids.npy"))
    # all_input_ids = np.concatenate((train_input_ids, test_input_ids, validate_input_ids), axis=0)
    # # all_input_ids = np.concatenate((test_input_ids, validate_input_ids), axis=0)




    vocab = OrderedDict([
        ('[PAD]', 0),
        ('[UNK]', 1),
        ('[CLS]', 2),
        ('[SEP]', 3),
        ('[MASK]', 4),
        ('a', 5),
        ('t', 6),
        ('c', 7),
        ('g', 8),
        ('n', 9)
    ])
    id_to_vocab = [key for key, value in vocab.items()]
    print(id_to_vocab)
    print('all_input_ids',all_input_ids.shape)
    print('atten_scores',atten_scores.shape)
    sequences = [(''.join([id_to_vocab[int(i)] for i in input_ids][1:-1]).upper()) for input_ids in all_input_ids]
    atten_scores = atten_scores[:, 1:-1]
    # pred_profile = pred_profile[:, 1:-1,:,:]
    # gt_profile = gt_profile[:, 1:-1,:,:]
    print('all_input_ids',all_input_ids.shape)
    print('atten_scores',atten_scores.shape)
    # print('pred_profile',pred_profile.shape)
    print(len(sequences),len(sequences[0]),sequences[0])

    # run motif analysis
    merged_motif_seqs = utils.motif_analysis(sequences,
                                        sequences,
                                        atten_scores,
                                        window_size = args.window_size,
                                        min_len = args.min_len,
                                        min_n_motif = args.min_n_motif,
                                        align_all_ties = args.align_all_ties,
                                        save_file_dir = args.save_file_dir,
                                        verbose = args.verbose,
                                        return_idx  = args.return_idx
                                    )
    print('predict_dir',args.predict_dir)
    print('save_file_dir',args.save_file_dir)

if __name__ == "__main__":
    main()


