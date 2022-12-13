#!/usr/bin/env python3
#
# Given a trials and scores file, this script
# prepares input for the binary compute-eer.

import sys

spkutt2target = {}
with open(sys.argv[1], 'r') as trials:
    for line in trials:
        spk, utt, target = line.strip().split()
        spkutt2target[spk + utt] = target

with open(sys.argv[2], 'r') as scores:
    for line in scores:
        spk, utt, score = line.strip().split()
        print(f"{score} {spkutt2target[spk + utt]}")
