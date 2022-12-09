# code reference : keras-rl/keras-rl/blob/master/rl/agents/dqn.py

from random import randint
from typing import Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from environment import ClassifyEnv
from buffer import Transition, ReplayBuffer, RLDataset


class Agent:
    """Basic agent that always returns 0"""

    def __init__(self, model: nn.Module):
        self.model = model

    def __call__(self, state: torch.Tensor, device: str) -> int:
        """
        Using the given network, decide what action to carry
        Args:
            state: current state of the environment
            device: device used for current batch
        Returns:
            action
        """
        return 0


class ValueAgent(Agent):
    def __init__(self, env: ClassifyEnv, replay_buffer: ReplayBuffer):
        self.env = env
        self.reset()
        self.buffer = replay_buffer
        self.state = self.env.reset()

    def get_action(self, model: nn.Module, state: torch.Tensor, epsilon: float, device: str) -> int:
        if np.random.random() < epsilon:
            action = self.get_random_action()
        else:
            action = self.get_normal_action(model, state, device)
        return action

    def get_random_action(self) -> int:
        return randint(0, self.env.action_space.n - 1)

    def get_normal_action(self, model: nn.Module, state: torch.Tensor, device: str) -> int:
        input_id, attention_mask = state[0].unsqueeze(0), state[1].unsqueeze(0)

        if not isinstance(input_id, torch.Tensor):
            input_id = torch.Tensor(input_id).float()
            attention_mask = torch.Tensor(attention_mask).float()
        if device != 'cpu':
            input_id = input_id.to(device)
            attention_mask = attention_mask.to(device)

        logits = model(input_ids=input_id, attention_mask=attention_mask)    # classification model
        action = torch.argmax(logits, dim=1)
        return int(action)

    @torch.no_grad()
    def step(self, model: nn.Module, epsilon: float, device: str = "cuda:0") -> Tuple[float, bool]:
        action = self.get_action(model, self.state, epsilon, device)
        new_state, reward, terminal, _ = self.env.step(action)
        trans = Transition(self.state, action, reward, new_state, terminal)

        self.buffer.append(trans)

        self.state = new_state
        if terminal:
            self.reset()
        return reward, terminal

    def reset(self):
        self.state = self.env.reset()


class PolicyAgent(Agent):
    def __init__(self, env: ClassifyEnv, replay_buffer: ReplayBuffer):
        self.env = env
        self.reset()
        self.buffer = replay_buffer
        self.state = self.env.reset()

    def step(self, policy : nn.Module, device):
        action, prob = policy.get_action_and_prob(self.state, device)
        new_state, reward, terminal, _ = self.env.step(action)
        trans = Transition(self.state, action, reward, new_state, terminal)

        self.buffer.append(trans)

        self.state = new_state
        if terminal:
            self.reset()
        return reward, terminal, prob

    def reset(self):
        self.state = self.env.reset()