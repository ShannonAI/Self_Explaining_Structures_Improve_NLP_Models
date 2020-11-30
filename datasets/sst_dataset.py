#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : sst_dataset.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/11/17 11:45
@version: 1.0
@desc  : sst5 and imdb task use the same dataset
"""
import os
from functools import partial

import torch
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader

from datasets.collate_functions import collate_to_max_length


class SSTDataset(Dataset):

    def __init__(self, directory, prefix, bert_path, max_length: int = 512):
        super().__init__()
        self.max_length = max_length
        with open(os.path.join(directory, prefix + '.txt'), 'r', encoding='utf8') as f:
            lines = f.readlines()
        self.lines = lines
        self.tokenizer = RobertaTokenizer.from_pretrained(bert_path)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        label, sentence = line.split('\t', 1)
        # delete .
        sentence = sentence.strip()
        if sentence.endswith("."):
            sentence = sentence[:-1]
        input_ids = self.tokenizer.encode(sentence, add_special_tokens=False)
        if len(input_ids) > self.max_length - 2:
            input_ids = input_ids[:self.max_length - 2]
        # convert list to tensor
        length = torch.LongTensor([len(input_ids) + 2])
        input_ids = torch.LongTensor([0] + input_ids + [2])
        label = torch.LongTensor([int(label)])
        return input_ids, label, length


def unit_test():
    root_path = "/data/nfsdata2/sunzijun/sstc/imdb_data"
    bert_path = "/data/nfsdata2/sunzijun/loop/roberta-base"
    prefix = "train"
    dataset = SSTDataset(directory=root_path, prefix=prefix, bert_path=bert_path)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=10,
        num_workers=0,
        shuffle=False,
        collate_fn=partial(collate_to_max_length, fill_values=[1, 0, 0])
    )
    for input_ids, label, length, start_index, end_index, span_mask in dataloader:
        print(input_ids.shape)
        print(start_index.shape)
        print(end_index.shape)
        print(span_mask.shape)
        print(label.view(-1).shape)
        print()


if __name__ == '__main__':
    unit_test()
