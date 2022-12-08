import os
import argparse
from typing import Dict, Tuple
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
import pytorch_lightning as pl
import wandb
from tqdm import tqdm

from dataset import KLAID_dataset
from network import Classifier
from environment import ClassifyEnv
from agents import ValueAgent
from buffer import ReplayBuffer, RLDataset

os.environ['TOKENIZERS_PARALLELISM']='FALSE'

class DQNClassification(pl.LightningModule):
    def __init__(self, hparams: Dict, run_mode: str):
        super(DQNClassification, self).__init__()
        self.save_hyperparameters(hparams)
        self.model_name = hparams['model_name']

        if run_mode == 'train':
            self.dataset = KLAID_dataset(model_name=self.model_name, split='train')
        elif run_mode == 'test':
            self.dataset = KLAID_dataset(model_name=self.model_name, split='test')

        self.num_classes = len(self.dataset.get_class_num())
        self.criterion = nn.MSELoss()

        print("\nInitializing the environment...")
        self.env = ClassifyEnv(run_mode=run_mode, dataset=self.dataset)
        self.env.seed(42)

        self.net = None
        self.target_net = None
        self.build_networks()

        self.capacity = len(self.dataset)
        self.buffer = ReplayBuffer(self.capacity)
        self.agent = ValueAgent(self.env, self.buffer)

        self.total_reward = 0
        self.avg_reward = 0
        self.reward_list = []

        self.episode_reward = 0
        self.episode_count = 0
        self.episode_steps = 0
        self.total_episode_steps = 0

        self.populate(self.hparams)


    def get_device(self, batch):
        return batch[0][0].device if torch.cuda.is_available() else 'cpu'

    def build_networks(self):
        # Initializing the DQN network and the target network
        self.classification_model = Classifier(model_name=self.model_name, num_classes=self.num_classes).to(self.device)
        self.target_model = Classifier(model_name=self.model_name, num_classes=self.num_classes).to(self.device)

    def populate(self, hparams) -> None:
        # steps: number of steps to populate the replay buffer
        print("\nPopulating the replay buffer...")
        device = hparams['gpu'][0]
        for _ in tqdm(range(len(self.env.env_data))):
            self.agent.step(self.classification_model, hparams['initial_eps'], 'cuda:{}'.format(device))

    def forward(self, batch):
        # Input: environment state
        # Output: Q values
        logits = self.classification_model(batch[0], batch[1])
        predictions = torch.argmax(logits, dim=1)
        return predictions

    def loss(self, batch):
        # Input: current batch (states, actions, rewards, next states, terminals) of replay buffer
        # Output: loss
        states, actions, rewards, next_states, terminals = batch
        state_action_values = self.classification_model(input_ids=states[0], attention_mask=states[1]).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_values = self.target_model(input_ids=states[0], attention_mask=states[1]).max(1)[0]
            next_state_values[terminals] = 0.0
            next_state_values = next_state_values.detach()
        expected_state_action_values = next_state_values * self.hparams['gamma'] + rewards

        return self.criterion(state_action_values, expected_state_action_values)

    def configure_optimizers(self):
        optimizer = AdamW(self.classification_model.parameters(), lr=float(self.hparams['lr']))
        return [optimizer]

    def get_epsilon(self, start, end, frames):
        # epsilon for Exploration or Exploitation
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> OrderedDict:
        device = self.get_device(batch)
        epsilon = self.get_epsilon(self.hparams['eps_start'], self.hparams['eps_end'], self.capacity)
        wandb.log({'train/epsilon': epsilon})

        reward, terminal = self.agent.step(self.classification_model, epsilon, device)
        self.episode_reward += reward
        self.episode_steps += 1
        # wandb.log({'train/episode_reward': self.episode_reward})

        # calculates training loss
        loss = self.loss(batch)
        wandb.log({'train/loss': loss})

        if terminal:
            self.total_reward = self.episode_reward
            self.reward_list.append(self.total_reward)
            self.avg_reward = sum(self.reward_list[-100:]) / 100
            self.episode_count += 1
            self.episode_reward = 0
            self.total_episode_steps = self.episode_steps
            self.episode_steps = 0

        # Soft update of target network
        if self.global_step % self.hparams['sync_rate'] == 0:
            self.target_model.load_state_dict(self.classification_model.state_dict())


        log = {'total_reward': self.total_reward,
               'avg_reward': self.avg_reward,
               'train_loss': loss,
               'episode_steps': self.total_episode_steps
               }
        status = {'steps': self.global_step,
                  'avg_reward': self.avg_reward,
                  'total_reward': self.total_reward,
                  'episodes': self.episode_count,
                  'episode_steps': self.episode_steps,
                  'epsilon': epsilon
                  }

        return {
            'loss' : loss,
            'avg_reward' : torch.tensor(self.avg_reward, dtype=float),
            'total_reward' : torch.tensor(self.total_reward, dtype=float),
            'episode_steps' : torch.tensor(self.episode_steps, dtype=float),
            'total_reward' : torch.tensor(self.total_reward, dtype=float),
            'log' : log,
            'progress_bar' : status
        }

    def training_epoch_end(self, outputs) -> None:
        collect = lambda key: torch.stack([x[key] for x in outputs]).mean()
        loss = collect('loss')
        avg_reward = collect('avg_reward')
        episode_steps = collect('episode_steps')
        total_reward = collect('total_reward')

        wandb.log({"train/loss": loss})

        wandb.log({"train/avg_reward": avg_reward})

        wandb.log({"train/episode_steps": episode_steps})
        wandb.log({"train/total_reward": total_reward})

        # reset episode related variables
        self.episode_reward = 0
        self.episode_count = 0
        self.episode_steps = 0


    def __dataloader(self):
        dataset = RLDataset(replay_buffer=self.buffer, buffer_size=self.capacity)
        dataloader = DataLoader(dataset,
                                batch_size=self.hparams['batch_size'], num_workers=self.hparams['num_workers'])

        return dataloader

    def train_dataloader(self):
        """Get train loader."""
        return self.__dataloader()

    def test_dataloader(self):
        """Get test loader."""
        return self.__dataloader()

    def test_step(self):
        # is there a validation step in RL????
        pass

    def test_epoch_end(self, outputs) -> None:
        pass

    def infer(self):
        pass
