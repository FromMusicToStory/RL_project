import os
import argparse
from typing import Dict, Tuple
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
import pytorch_lightning as pl

from dataset import KLAID_dataset
from network import Classifier
from environment import ClassifyEnv
from agents import ValueAgent
from buffer import ReplayBuffer, RLDataset


class DQNClassification(pl.LightningModule):
    def __init__(self, hparams: Dict, run_mode: str):
        super(DQNClassification, self).__init__()
        self.hparams = hparams
        self.model_name = hparams['model_name']
        self.dataset = KLAID_dataset(model_name=self.model_name, split='all')
        self.num_classes = len(self.dataset.get_class_num())
        self.criterion = nn.MSELoss()

        self.env = ClassifyEnv(run_mode=run_mode, dataset=self.dataset)
        self.env.seed(42)

        self.net = None
        self.target_net = None
        self.build_networks()

        self.capacity = len(self.dataset)
        self.buffer = ReplayBuffer(self.capacity)
        self.agent = ValueAgent(self.env, self.buffer, batch_size=hparams['batch_size'])
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams['warm_start'])


    def populate(self, warm_start: int) -> None:
        # Populates the buffer with initial trasnsition
        # warm_start: number of episodes to populate the buffer
        if warm_start > 0:
            for _ in range(warm_start):
                self.agent.step(self.classification_model, epsilon=1.0)

    def build_networks(self):
        # Initializing the DQN network and the target network
        self.classification_model = Classifier(model_name=self.model_name, num_classes=self.num_classes)
        self.target_model = Classifier(model_name=self.model_name, num_classes=self.num_classes)

    def forward(self, batch):
        # InputL environment state
        # Output: Q values
        logits = self.classification_model(batch['encoded_output'], batch['encoded_attention_mask'])
        predictions = torch.argmax(logits, dim=1)
        return predictions

    def loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        state_action_values = self.classification_model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_values = self.target_model(next_states),max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()
        expected_state_action_values = next_state_values * self.hparams['gamma'] + rewards

        return self.criterion(state_action_values, expected_state_action_values)

    def configure_optimizers(self):
        optimizer = AdamW(self.classification_model.parameters(), lr=self.hparams['lr'])
        return [optimizer]

    def get_epsilon(self, start, end, frames):
        # epsilon for Exploration or Exploitation
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> OrderedDict:
        epsilon = self.get_epsilon(self.hparams['epsilon_start'], self.hparams['epsilon_final'], self.hparams['epsilon_frames'])
        self.log('epsilon', epsilon)

        exp, reward, done = self.source.step()
        reward, done = self.agent.play
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

    def _dataloader(self):
        self.buffer = ReplayBuffer(self.capacity)
        self.populate(self.hparams['warm_start'])

        dataset = RLDataset(self.buffer, self.hparams['batch_size'])
        dataloader = DataLoader(dataset,
                                batch_size=self.hparams['batch_size'], shuffle=True, num_workers=4)
        return dataloader


    def train_dataloader(self):
        """Get train loader."""
        return self._dataloader()

    def test_dataloader(self):
        """Get test loader."""
        return self._dataloader()

    def test_step(self):
        # is there a validation step in RL????
        pass

    def test_epoch_end(self, outputs) -> None:
        pass

    def infer(self):
        pass
