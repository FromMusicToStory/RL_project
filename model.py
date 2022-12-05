import os
import argparse
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import pytorch_lightning as pl

from dataset import KLAID_dataset
from network import Classifier
from environment import ClassifyEnv
from memory import SequentialMemory


class RLClassification(pl.LightningModule):
    def __init__(self, hparams: Dict, run_mode: str):
        super(RLClassification, self).__init__()
        self.hparams = hparams
        self.model_name = hparams['model_name']
        self.dataset = KLAID_dataset(split='all', tokenizer_name=self.model_name)
        self.num_classes = len(self.dataset.get_class_num())

        self.classification_model = Classifier(model_name=self.model_name, num_classes=self.num_classes)
        self.env = ClassifyEnv(run_mode=run_mode, dataset=self.dataset)
        self.agent = Agent()
        self.policy = Policy()
        self.processor = ClassifyProcessor()
        self.memory = SequentialMemory()

    def train_dataloader(self):
        dataset = KLAID_dataset(split='train', tokenizer_name=self.model_name)  # model name is for pretraiend tokenizer
        train_dataloader = DataLoader(dataset, batch_size=self.hparams['batch_size'],
                                      num_workers=self.hparams['num_workers'],
                                      shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        dataset = KLAID_dataset(split='test', tokenizer_name=self.model_name)
        val_dataloader = DataLoader(dataset, batch_size=self.hparams['batch_size'],
                                      num_workers=self.hparams['num_workers'],
                                      shuffle=True)
        return val_dataloader

    def configure_optimizers(self):
        optimizer = AdamW( lr=self.hparams['lr'])

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        pass

    def validation_step(self):
        # is there a validation step in RL????
        pass

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        pass

    def infer(self):
        pass
