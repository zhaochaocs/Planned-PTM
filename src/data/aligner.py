#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
__author __ = 'chao'
__date__ = 2/12/21 2:25 PM

Copyright 2021 Chao Zhao
SPDX-License-Identifier: Apache-2.0
"""


import itertools
import re
from collections import defaultdict

import spacy
from nltk import SnowballStemmer
from strsimpy import MetricLCS

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
stemmer = SnowballStemmer("english")
metric_lcs = MetricLCS()


class Aligner:

    def __init__(self):
        self.src_doc, self.tgt_doc = [], []
        self.tgt_matched = set()

    def preprocess(self, text):
        text = re.sub(' +', ' ', text) if isinstance(text, str) else \
            ' '.join([token for token in text if len(token.strip())])
        doc = nlp(text.lower())
        stem = [stemmer.stem(x.text) for x in doc]
        return doc, stem

    def align(self, src, tgt, multi=False, distance=0.0, mute=True):
        self.src_doc, self.src_stem = self.preprocess(src)
        self.tgt_doc, self.tgt_stem = self.preprocess(tgt)
        self.alignment = {} if not multi else defaultdict(list)
        self.tgt_matched.clear()
        self.multi, self.distance = multi, distance

        self.match_word()
        if len(self.src_doc) != len(self.alignment) or self.multi:
            self.match_lemma()
        if len(self.src_doc) != len(self.alignment) or self.multi:
            self.match_stem()
        if self.distance > 0 and (len(self.src_doc) != len(self.alignment) or self.multi):
            self.match_dist(mute)

        if self.multi:
            self.alignment = {src_i: list(set(tgt_idx)) for src_i, tgt_idx in self.alignment.items()}
            self.alignment = sorted(self.alignment.items(), key=lambda kv: min(kv[1]))
            src_list, alignment = [kv[0] for kv in self.alignment], [kv[1] for kv in self.alignment]

            src_lists = []
            for tgt_idxes in itertools.product(*alignment):
                src_lists.append([src for _, src in sorted(zip(tgt_idxes, src_list))])
            src_lists = [" ".join([self.src_doc[i].text for i in src_list])
                         for src_list in src_lists]
            return src_lists, " ".join([tk.text for tk in self.tgt_doc]), \
                   alignment, len(self.src_doc) == len(self.alignment)

        else:
            self.alignment = {src_i: tgt_idx for src_i, tgt_idx in self.alignment.items()}
            # src_list, alignment = list(zip(*sorted(self.alignment.items(), key=lambda kv: kv[1]))) if len(self.alignment) else [], []
            self.alignment = sorted(self.alignment.items(), key=lambda kv: kv[1])
            src_list, alignment = [kv[0] for kv in self.alignment], [kv[1] for kv in self.alignment]
            src_list = " ".join([self.src_doc[i].text for i in src_list])
            return src_list, " ".join([tk.text for tk in self.tgt_doc]), \
                   alignment, len(self.src_doc) == len(self.alignment)

    def add_matched(self, src_i, matched):
        matched = list(matched)
        if len(matched):
            if not self.multi:
                self.alignment[src_i] = matched[0].i
                self.tgt_matched.add(matched[0].i)
            else:
                self.alignment[src_i] += [m.i for m in matched]

    def match_word(self):
        for src_i, src_token in enumerate(self.src_doc):
            matched = filter(lambda x: x.text == src_token.text and (x.i not in self.tgt_matched or self.multi),
                             self.tgt_doc)
            self.add_matched(src_i, matched)

    def match_lemma(self):
        for src_i, src_token in enumerate(self.src_doc):
            if src_i in self.alignment and not self.multi:
                continue
            matched = filter(lambda x: len({src_token.text, src_token.lemma_} & {x.text, x.lemma_}) and \
                                       (x.i not in self.tgt_matched or self.multi), self.tgt_doc)
            self.add_matched(src_i, matched)

    def match_stem(self):
        for src_i, src_token in enumerate(self.src_doc):
            if src_i in self.alignment and not self.multi:
                continue
            matched = filter(lambda x: len({src_token.text, src_token.lemma_, self.src_stem[src_token.i]} &
                                           {x.text, x.lemma_, self.tgt_stem[x.i]}) and \
                                       (x.i not in self.tgt_matched or self.multi), self.tgt_doc)
            self.add_matched(src_i, matched)

    def token_distance(self, tokens1, tokens2):
        distances = []
        for t1, t2 in itertools.product(*[set(tokens1), set(tokens2)]):
            distances.append(metric_lcs.distance(t1, t2))
        return min(distances)

    def match_dist(self, mute=True):
        for src_i, src_tk in enumerate(self.src_doc):
            if src_i in self.alignment and not self.multi:
                continue
            candidates = filter(lambda x: x.i not in self.tgt_matched, self.tgt_doc)
            candidates = map(lambda x: (x, self.token_distance([x.text, x.lemma_, self.tgt_stem[x.i]],
                                                               [src_tk.text, src_tk.lemma_, self.src_stem[src_tk.i]])),
                             candidates)
            candidates = sorted(candidates, key=lambda x: x[1])
            if len(candidates) and candidates[0][1] <= self.distance:
                candidate_idx = [candidate[0].i for candidate in candidates if candidate[1] <= self.distance]
                if not self.multi:
                    candidate_idx = [i for i in candidate_idx if i not in self.tgt_matched]
                    if not mute:
                        print(f"Map {src_tk.text} to {self.tgt_doc[candidate_idx[0]].text} via lcs distance")
                    if len(candidate_idx):
                        self.alignment[src_i] = candidate_idx[0]
                        self.tgt_matched.add(candidate_idx[0])
                else:
                    self.alignment[src_i] += candidate_idx
                    candidates_token = ', '.join([self.tgt_doc[i].text for i in candidate_idx])
                    if not mute:
                        print(f"Map {src_tk.text} to {candidates_token} via lcs distance")

