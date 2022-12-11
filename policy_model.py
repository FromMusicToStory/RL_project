import os
import math

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
import pytorch_lightning as pl

import wandb
from tqdm import tqdm
from hydra.utils import instantiate

from dataset import KLAID_dataset
from network import Classifier, DuelingClassifier, PolicyNet
from environment import ClassifyEnv
from agents import ValueAgent, PolicyAgent
from buffer import ReplayBuffer, RLDataset

import numpy as np

os.environ['TOKENIZERS_PARALLELISM']='FALSE'

class PolicyGradientClassification(pl.LightningModule):
    def __init__(self, hparams, run_mode):
        super(PolicyGradientClassification, self).__init__()
        self.automatic_optimization = False

        self.save_hyperparameters(hparams)
        self.model_name = hparams['model_name']
        self.model = hparams['net']

        if run_mode == 'train':
            self.dataset = KLAID_dataset(model_name=self.model_name, split='train')
        elif run_mode == 'test':
            self.dataset = KLAID_dataset(model_name=self.model_name, split='test')

        self.num_classes = len(self.dataset.get_class_num())

        print("\nInitializing the environment...")
        self.env = ClassifyEnv(run_mode=run_mode, dataset=self.dataset)
        self.env.seed(42)

        self.build_networks(self.hparams)

        self.capacity = len(self.dataset)
        self.buffer = ReplayBuffer(self.capacity)
        self.agent = PolicyAgent(env=self.env, replay_buffer=self.buffer)

        self.total_reward = 0
        self.avg_reward = 0
        self.reward_list = []

        self.episode_reward = 0
        self.episode_count = 0
        self.episode_steps = 0
        self.total_episode_steps = 0

        self.gamma = hparams['gamma']
        self.returns = []
        self.probs = []

        self.p_criterion = None
        self.v_criterion = nn.MSELoss()

        self.populate(self.hparams)

    def populate(self, hparams):
        print("\nPopulating the replay buffer...")
        print()
        device = hparams['gpu'][0]
        for _ in tqdm(range(len(self.env.env_data))):
            self.agent.step(self.p_net, device)

    def get_device(self, batch):
        return batch[0][0].device if torch.cuda.is_available() else 'cpu'

    def build_networks(self, hparams):
        device = hparams['gpu'][0]
        self.p_net = instantiate(self.model['policy_net'])
        self.p_net.to(device)
        self.v_net = instantiate(self.model['value_net'])
        self.v_net.to(device)

    def configure_optimizers(self):
        p_opt = AdamW(self.p_net.parameters(), lr=float(self.hparams['lr']))
        v_opt = AdamW(self.v_net.parameters(), lr=float(self.hparams['lr']))
        return p_opt, v_opt

    def calculate_returns(self, rewards):
        discounted_rewards = [ math.pow(self.gamma,i) * r for i, r in enumerate(rewards)]
        return [ sum(discounted_rewards[i:]) for i in range(len(discounted_rewards))]

    def forward(self, batch):
        logits = self.v_net(batch[0], batch[1])
        predictions = self.p_net(logits, dim=1)
        return predictions

    def training_step(self, batch):
        p_opt, v_opt = self.optimizers()
        device = self.get_device(batch)

        terminal = False
        probs, rewards = [], []

        v_net_output = self.v_net(batch[0][0], batch[0][1]).max(1)[0]
        v_net_output = v_net_output * 0.9
        while terminal is not True:
            reward, terminal, prob = self.agent.step(self.p_net, device)
            self.episode_reward += reward
            self.episode_steps += 1

            rewards.append(reward)
            probs.append(prob)

        returns = self.calculate_returns(rewards)
        returns = torch.as_tensor(returns)

        probs = torch.as_tensor(probs)
        rewards = torch.as_tensor(rewards)
        p_loss = - (returns * probs).mean()


        v_loss = self.v_criterion(returns, v_net_output[0].detach().cpu() * rewards)

        p_loss.requires_grad_(True)
        v_loss.requires_grad_(True)


        p_opt.zero_grad()
        self.manual_backward(p_loss)
        p_opt.step()

        v_opt.zero_grad()
        self.manual_backward(v_loss)
        v_opt.zero_grad()

        self.log('policy_loss', p_loss)


        if terminal:
            self.total_reward = self.episode_reward
            self.reward_list.append(self.total_reward)
            self.avg_reward = sum(self.reward_list[-100:]) / 100
            self.episode_count += 1
            self.episode_reward = 0
            self.total_episode_steps = self.episode_steps
            self.episode_steps = 0

        log = {'policy_loss': p_loss,
               'value_loss' : v_loss,
               'total_reward': self.total_reward,
               'avg_reward': self.avg_reward,
               'episode_steps': self.total_episode_steps
               }
        status = {'steps': self.global_step,
                  'avg_reward': self.avg_reward,
                  'total_reward': self.total_reward,
                  'episodes': self.episode_count,
                  'episode_steps': self.episode_steps,
                  }

        return {
            'policy_loss': p_loss,
            'value_loss' : v_loss,
            'avg_reward' : torch.tensor(self.avg_reward, dtype=float),
            'total_reward' : torch.tensor(self.total_reward, dtype=float),
            'episode_steps' : torch.tensor(self.episode_steps, dtype=float),
            'total_reward' : torch.tensor(self.total_reward, dtype=float),
            'log' : log,
            'progress_bar' : status
        }

    def training_epoch_end(self, outputs):
        collect = lambda key: torch.stack([x[key] for x in outputs]).mean()
        p_loss = collect('policy_loss')
        v_loss = collect('value_loss')
        wandb.log({'train/policy_loss': p_loss})
        wandb.log({'train/value_loss': v_loss})



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