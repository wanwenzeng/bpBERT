import pybedtools
from pybedtools import BedTool, Interval
import random
from copy import deepcopy


def shift_interval(interval, max_shift):
    shift = random.randint(-max_shift, max_shift)
    interval.start = interval.start + shift
    interval.end = interval.end + shift
    return interval

def resize_interval(interval, width):
    center = (interval.end + interval.start) // 2
    start = center - width // 2
    end = center + width // 2
    interval.start = start
    interval.end = end
    return interval

def concatenate_peaks_region(beds_list):
    bt_list = []
    for bed in beds_list:
        bt = BedTool(bed)
        bt_list.append(bt)

    a = bt_list.pop(0)
    for i in range(len(bt_list)):
        b = bt_list.pop(0)
        a = a.cat(b)
    return a

def load_intervals(beds_file, max_shift=0, width=1000, exclude_chr=['chrX','chrY']):
    beds_list = []
    for line in open(beds_file, 'r'):
        bed = line.rstrip()
        beds_list.append(bed)
    bed_concat = concatenate_peaks_region(beds_list)
    bed_merge = bed_concat.merge()

    intervals = []
    for feature in bed_merge:
        if feature[0] not in exclude_chr:
            if int(feature[2]) - int(feature[1]) > width:
                continue
            interval = Interval(feature[0], int(feature[1]), int(feature[2]))
            interval = shift_interval(interval, max_shift)
            target_interval = resize_interval(interval, width)
            intervals.append(target_interval)
    
    return intervals




    







