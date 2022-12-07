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

        self.env = ClassifyEnv(run_mode=run_mode, dataset=self.dataset)
        self.env.seed(42)

        self.net = None
        self.target_net = None
        self.build_networks()

        self.capacity = len(self.dataset)
        self.buffer = ReplayBuffer(self.capacity)
        self.agent = ValueAgent(self.classification_model, self.env, self.buffer)

        self.total_reward = 0
        self.avg_reward = 0
        self.reward_list = []

        self.episode_reward = 0
        self.episode_count = 0
        self.episode_steps = 0
        self.total_episode_steps = 0

        self.populate(self.hparams)


    def get_device(self, batch):
        return batch[0].device if torch.cuda.is_available() else 'cpu'

    def build_networks(self):
        # Initializing the DQN network and the target network
        self.classification_model = Classifier(model_name=self.model_name, num_classes=self.num_classes).to(self.device)
        self.target_model = Classifier(model_name=self.model_name, num_classes=self.num_classes).to(self.device)

    def populate(self, hparams) -> None:
        # steps: number of steps to populate the replay buffer
        warm_up_data = self.env.env_data[:hparams['warm_start_steps']]
        for _ in range(hparams['warm_start_steps']):
            self.agent.step(warm_up_data[0], warm_up_data[1], hparams['eps'])

    def forward(self, batch):
        # Input: environment state
        # Output: Q values
        logits = self.classification_model(batch[0], batch[1])
        predictions = torch.argmax(logits, dim=1)
        return predictions

    def get_attention_mask(self, batched_input):
        mask = []
        for x in batched_input:
            print(x)


    def loss(self, batch):
        # Input: current batch (states, actions, rewards, next states, terminals) of replay buffer
        # Output: loss
        states, actions, rewards, next_states, terminals, cur_attn, next_attn = batch
        state_action_values = self.classification_model(input_ids=states, attention_mask=cur_attn).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_values = self.target_model(input_ids=next_states, attention_mask=next_attn).max(1)[0]
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
        wandb.log({'train/epsilon':epsilon })

        # Training
        # batch로 부터 state를 받아서 agent에 넘기고
        # agent로 부터 prediction result, reward를 받음
        # agent로부터 받음 result로 MSE loss 를 계산하도록 수정
        #   env 로부터 answer를 받아와서 -> agent에서 true answer 받아서 MSE loss 계산
        states, _, _, _, _, attention_mask, _ = batch
        reward, terminal = self.agent.step(states, attention_mask, epsilon, device)
        self.episode_reward += reward


        # calculates training loss
        loss = self.loss(batch)

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
                  'epsilon': epsilon
                  }

        return OrderedDict({'loss': loss, 'avg_reward': torch.tensor(self.avg_reward),
                            'log': log, 'progress_bar': status})

    def training_epoch_end(self, outputs) -> None:
        wandb.log({"train/loss": outputs['loss']})

        wandb.log({"train/avg_reward": outputs['avg_reward']})

        wandb.log({"train/episode_steps": outputs['progress_bar']['episode_steps']})
        wandb.log({"train/total_reward": outputs['progress_bar']['total_reward'].detach().cpu().numpy()})



    def __dataloader(self):
        dataset = RLDataset(replay_buffer=self.buffer, batch_size=self.hparams['batch_size'])
        dataloader = DataLoader(dataset,
                                batch_size=self.hparams['batch_size'], num_workers=4)
        # shuffle =True에서 오류 남 (ValueError: DataLoader with IterableDataset: expected unspecified shuffle option, but got shuffle=True)
        # 어차피 env에서 random으로 data 불러오니까 여기서는 그냥 가져와도 될 듯?

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
