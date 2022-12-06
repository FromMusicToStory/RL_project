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

    def get_action(self, state: torch.Tensor, epsilon: float, device: str) -> int:
        if np.random.random() < epsilon:
            action = self.get_random_action()
        else:
            action = self.get_normal_action(state, device)
        return action

    def get_random_action(self) -> int:
        return randint(0, self.env.action_space.n - 1)

    def get_normal_action(self, state: torch.Tensor, device: str) -> int:
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor([state]).float()
        if device != 'cpu':
            state = state.to(device)

        q_values = self.model(state)
        _, action = torch.max(q_values, dim=1)
        return int(action.item())

    @torch.no_grad()
    def step(self, model: nn.Module,
                   epsilon: float,
                   device: str = "cuda:0") -> Tuple[float, bool]:
        action = self.get_action(model, device)
        new_state, reward, terminal, _, _  = self.env.step(action)
        trans = Transition(self.state, action, reward, new_state, terminal)

        self.buffer.append(trans)

        self.state = new_state
        if terminal:
            self.reset()
        return reward, terminal

    def reset(self):
        self.state = self.env.reset()


class PolicyAgent(Agent):
    def __call__(self, state: torch.Tensor, device: str) -> int:
        """
        Takes in the current state and returns the action based on the agents policy
        Args:
            state: current state of the environment
            device: the device used for the current batch
        Returns:
            action defined by policy
        """
        if device.type != 'cpu':
            state = state.cuda(device)

        # get the logits and pass through softmax for probability distribution
        probabilities = F.softmax(self.net(state))
        prob_np = probabilities.data.cpu().numpy()

        # take the numpy values and randomly select action based on prob distribution
        action = np.random.choice(len(prob_np), p=prob_np)

        return action