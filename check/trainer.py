#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : trainer.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/11/16 21:55
@version: 1.0
@desc  :
"""

import argparse
import json
import os
from functools import partial

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaTokenizer
from transformers.modeling_roberta import RobertaForSequenceClassification

from datasets.collate_functions import collate_to_max_length
from datasets.sst_dataset import SSTDataset
from datasets.snli_dataset import SNLIDataset
from utils.radom_seed import set_random_seed

set_random_seed(0)


class CheckExplainNLP(pl.LightningModule):

    def __init__(
        self,
        args: argparse.Namespace
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        self.bert_dir = args.bert_path
        self.model = RobertaForSequenceClassification.from_pretrained(self.bert_dir)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.bert_dir)
        self.loss_fn = CrossEntropyLoss()
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          betas=(0.9, 0.98),  # according to RoBERTa paper
                          lr=self.args.lr,
                          eps=self.args.adam_epsilon)
        t_total = len(self.train_dataloader()) // self.args.accumulate_grad_batches * self.args.max_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=t_total)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids):
        attention_mask = (input_ids != 1).long()
        return self.model(input_ids, attention_mask=attention_mask)

    def compute_loss_and_acc(self, batch, mode='train'):
        input_ids, labels, length, start_indexs, end_indexs, span_masks = batch
        y = labels.view(-1)
        y_hat = self.forward(input_ids)[0]
        # compute loss
        loss = self.loss_fn(y_hat, y)
        # compute acc
        predict_scores = F.softmax(y_hat, dim=1)
        predict_labels = torch.argmax(predict_scores, dim=-1)
        if mode == 'train':
            acc = self.train_acc(predict_labels, y)
        else:
            acc = self.valid_acc(predict_labels, y)
        return loss, acc

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    def training_step(self, batch, batch_idx):
        loss, acc = self.compute_loss_and_acc(batch)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log('train_acc', acc, on_step=True, on_epoch=False)
        self.log('train_loss', loss)
        return loss

    def val_dataloader(self):
        return self.get_dataloader("test")

    def validation_step(self, batch, batch_idx):
        loss, acc = self.compute_loss_and_acc(batch, mode='dev')
        self.log('valid_acc', acc, on_step=False, on_epoch=True)
        self.log('valid_loss', loss)
        return loss

    def validation_epoch_end(self, outs):
        # log epoch metric
        self.valid_acc.compute()
        self.log('valid_acc_end', self.valid_acc.compute())

    def get_dataloader(self, prefix="train") -> DataLoader:
        """get training dataloader"""
        if self.args.task == 'snli':
            dataset = SNLIDataset(directory=self.args.data_dir, prefix=prefix,
                                  bert_path=self.bert_dir,
                                  max_length=self.args.max_length)
        else:
            dataset = SSTDataset(directory=self.args.data_dir, prefix=prefix,
                                 bert_path=self.bert_dir,
                                 max_length=self.args.max_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length, fill_values=[1, 0, 0]),
            drop_last=False
        )
        return dataloader


def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", required=True, type=str, help="bert config file")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-9, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--use_memory", action="store_true", help="load dataset to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of dataset")
    parser.add_argument("--data_dir", required=True, type=str, help="train data path")
    parser.add_argument("--save_path", required=True, type=str, help="path to save checkpoints")
    parser.add_argument("--save_topk", default=5, type=int, help="save topk checkpoint")
    parser.add_argument("--checkpoint_path", type=str, help="checkpoint path on test step")
    parser.add_argument("--span_topk", type=int, default=5, help="save topk spans on test step")
    parser.add_argument("--lamb", default=1.0, type=float, help="regularizer lambda")
    parser.add_argument("--task", default='sst5', type=str, help="nlp tasks")

    return parser


def main():
    """main"""
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    # 如果save path不存在，则创建
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model = CheckExplainNLP(args)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_path, '{epoch}-{valid_loss:.4f}-{valid_acc_end:.4f}'),
        save_top_k=args.save_topk,
        save_last=True,
        monitor="valid_acc_end",
        mode="max",
    )
    logger = TensorBoardLogger(
        save_dir=args.save_path,
        name='log'
    )

    # save args
    with open(os.path.join(args.save_path, "args.json"), 'w') as f:
        args_dict = args.__dict__
        del args_dict['tpu_cores']
        json.dump(args_dict, f, indent=4)

    trainer = Trainer.from_argparse_args(args,
                                         checkpoint_callback=checkpoint_callback,
                                         distributed_backend="ddp",
                                         logger=logger)

    trainer.fit(model)


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()
