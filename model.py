import os
import argparse
from typing import Dict, Tuple
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import pytorch_lightning as pl

from dataset import KLAID_dataset
from network import Classifier
from environment import ClassifyEnv
from agents import ValueAgent
from memory import SequentialMemory


class DQNClassification(pl.LightningModule):
    def __init__(self, hparams: Dict, run_mode: str):
        super(DQNClassification, self).__init__()
        self.hparams = hparams

        self.model_name = hparams['model_name']

        self.dataset = KLAID_dataset(model_name=self.model_name, split='all')
        self.num_classes = len(self.dataset.get_class_num())

        self.env = ClassifyEnv(run_mode=run_mode, dataset=self.dataset)
        self.env.seed(42)

        self.net = None
        self.target_net = None

        self.agent = ValueAgent(self.net, self.num_classes, )
        self.policy = Policy()
        self.processor = ClassifyProcessor()
        # do we need processor?
        # Explanation about processor from keras-rl
        # A processor acts as a coupling mechanism between an `Agent` and its `Env`. This can
        #     be necessary if your agent has different requirements with respect to the form of the
        #     observations, actions, and rewards of the environment. By implementing a custom processor,
        #     you can effectively translate between the two without having to change the underlaying
        #     implementation of the agent or environment.
        self.memory = SequentialMemory(limit=hparams['limit'], **hparams)

        self.total_reward = 0
        self.episode_reward = 0

        self.total_episode_step = 0
        self.episode_step = 0

        self.reward_list = []

    def populate(self, warm_start: int) -> None:
        # Populates the buffer with initial experience
        if warm_start > 0:
            for _ in range(warm_start):
                self.source.agent.epsilon = 1.0
                exp, _, _ = self.source.step()
                self.buffer.append(exp)

    def build_networks(self):
        # Initializing the DQN network and the target network
        self.classification_model = Classifier(model_name=self.model_name, num_classes=self.num_classes)
        self.target_model = Classifier(model_name=self.model_name, num_classes=self.num_classes)


    def train_dataloader(self):
        dataset = KLAID_dataset(split='train', model_name=self.model_name)  # model name is for pre-traiend tokenizer
        train_dataloader = DataLoader(dataset, batch_size=self.hparams['batch_size'],
                                      num_workers=self.hparams['num_workers'],
                                      shuffle=True)
        return train_dataloader

    def test_dataloader(self):
        dataset = KLAID_dataset(split='test', model_name=self.model_name)
        test_dataloader = DataLoader(dataset, batch_size=self.hparams['batch_size'],
                                      num_workers=self.hparams['num_workers'],
                                      shuffle=True)
        return test_dataloader

    def configure_optimizers(self):
        optimizer = AdamW(self.net.parameters(), lr=self.hparams['lr'])
        return [optimizer]

    def forward(self, batch):
        # first, classification model => get logits
        logits = self.classification_model(batch['encoded_output'], batch['encoded_attention_mask'])
        # second, get action from logits (Agent)
        return logits

    def loss(self, batch):


    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> OrderedDict:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved
        Args:
            batch: current mini batch of replay data
            _: batch number, not used
        Returns:
            Training loss and log metrics
        """
        self.agent.update_epsilon(self.global_step)

        # step through environment with agent and add to buffer
        exp, reward, done = self.source.step()
        self.buffer.append(exp)

        self.episode_reward += reward
        self.episode_steps += 1

        # calculates training loss
        loss = self.loss(batch)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        if done:
            self.total_reward = self.episode_reward
            self.reward_list.append(self.total_reward)
            self.avg_reward = sum(self.reward_list[-100:]) / 100
            self.episode_count += 1
            self.episode_reward = 0
            self.total_episode_steps = self.episode_steps
            self.episode_steps = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {'total_reward': torch.tensor(self.total_reward).to(self.device),
               'avg_reward': torch.tensor(self.avg_reward),
               'train_loss': loss,
               'episode_steps': torch.tensor(self.total_episode_steps)
               }
        status = {'steps': torch.tensor(self.global_step).to(self.device),
                  'avg_reward': torch.tensor(self.avg_reward),
                  'total_reward': torch.tensor(self.total_reward).to(self.device),
                  'episodes': self.episode_count,
                  'episode_steps': self.episode_steps,
                  'epsilon': self.agent.epsilon
                  }

        return OrderedDict({'loss': loss, 'avg_reward': torch.tensor(self.avg_reward),
                            'log': log, 'progress_bar': status})

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        pass
        # get metrics and log the loss

    def test_step(self):
        # is there a validation step in RL????
        pass

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        pass

    def infer(self):
        pass
