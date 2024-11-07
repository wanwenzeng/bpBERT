from pysam import FastaFile
import pyBigWig
import numpy as np

def extract_bigwig(bigwig_file, intervals):
    targets = []

    bw = pyBigWig.open(bigwig_file)
    for interval in intervals:
        chrom, start, end = interval.chrom, interval.start, interval.end
        # read values
        try:
            values = bw.values(chrom, start, end)
        except:
            # print("WARNING: %s doesn't see %s:%d-%d. Setting to all zeros." % (bigwig_file,chrom,start,end))
            values = np.zeros(end-start)

        # set NaNs to zero
        values = np.nan_to_num(values)
        targets.append(values)

    targets = np.array(targets)

    return targets


def extract_bigwig_stranded(bigwigs, intervals):
    out = np.stack([np.abs(extract_bigwig(bw, intervals))
                        for bw in bigwigs], axis=-1)
    return out


def extract_fasta(fasta_file, intervals):
    seqs = []
    fasta_file = FastaFile(fasta_file)

    for interval in intervals:
        chrom, start, end = interval.chrom, interval.start, interval.end
        seq = fasta_file.fetch(chrom, start, end)
        seq = seq.lower()
        seqs.append(seq)
    
    return np.array(seqs)


