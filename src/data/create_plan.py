#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
__author __ = 'chao'
__date__ = 7/10/22 3:37 PM

Copyright 2022 Chao Zhao
SPDX-License-Identifier: Apache-2.0
"""


import argparse

from pathlib import Path
from tqdm import tqdm

from aligner import Aligner

aligner = Aligner()


def get_file_len(file):
    with open(file, "r") as f:
        return len(f.readlines())


def convert_realization_to_plan(src_file, tgt_file, plan_file):
    with open(src_file, 'r') as fr1, open(tgt_file, 'r') as fr2, open(plan_file, 'w') as fw:
        for line_src, line_tgt in tqdm(zip(fr1.readlines(), fr2.readlines()), total=get_file_len(src_file)):
            concepts = line_src.strip().split()
            sentence = line_tgt.strip()
            plan, sentence, plan_idx, _ = aligner.align(concepts, sentence, multi=False, distance=1)
            plan = plan.split()
            fw.write(' '.join(plan) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--src_file", default=None, type=str, help="The input src data file name.")
    parser.add_argument("--tgt_file", default=None, type=str, help="The input tgt data file name.")
    parser.add_argument("--plan_file", default=None, type=str, help="The output plan data file name.")
    args = parser.parse_args()

    fast_match = False

    src_file = args.src_file
    tgt_file = args.tgt_file
    plan_file = args.plan_file
    Path(plan_file).parents[0].mkdir(exist_ok=True, parents=True)

    convert_realization_to_plan(src_file, tgt_file, plan_file)
