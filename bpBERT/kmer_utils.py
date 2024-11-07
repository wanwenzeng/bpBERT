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
"""utils for kmer for bpBERT model."""

import os
import logging

KMER_DICT_NAME = 'kmer.txt'

logger = logging.getLogger(__name__)

class bpBERTKmerDict(object):
    """
    Dict class to store the kmer
    """
    def __init__(self, kmer_freq_path, tokenizer, max_kmer_in_seq=128):
        """Constructs bpBERTKmerDict

        :param kmer_freq_path: kmers with frequency
        """
        if os.path.isdir(kmer_freq_path):
            kmer_freq_path = os.path.join(kmer_freq_path, KMER_DICT_NAME)
        self.kmer_freq_path = kmer_freq_path
        self.max_kmer_in_seq = max_kmer_in_seq
        self.id_to_kmer_list = ["[pad]","[mask]","[unk]"]
        self.kmer_to_id_dict = {"[pad]": 0,"[mask]": 1,"[unk]":2}
        self.kmer_to_freq_dict = {}
        logger.info("loading kmer frequency file {}".format(kmer_freq_path))
        with open(kmer_freq_path, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                if " " in line:
                    line =line.strip()
                    kmer,freq = line.split(" ")
                else:
                    kmer = line.strip()
                    freq = 1111

                tokens = tuple(tokenizer.tokenize(kmer))
                self.kmer_to_freq_dict[kmer] = freq
                self.id_to_kmer_list.append(tokens)
                self.kmer_to_id_dict[tokens] = i + 3

    def save(self, kmer_freq_path):
        with open(kmer_freq_path, "w", encoding="utf-8") as fout:
            for kmer,freq in self.kmer_to_freq_dict.items():
                fout.write("{}\n".format(kmer))